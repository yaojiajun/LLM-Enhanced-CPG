from __future__ import annotations

from typing import Optional, List, Tuple
import logging
import subprocess
import numpy as np
import os
import sys
import difflib
import signal
import torch
from omegaconf import DictConfig

from utils.utils import (
    file_to_string,
    extract_code_from_generator,
    filter_traceback,
    filter_code,
    block_until_running,
)
from utils.llm_client.base import BaseClient


class Hercules:
    """Evolutionary LLM pipeline for generating and evaluating heuristic functions."""

    # -----------------------------
    # Constants / knobs
    # -----------------------------
    MAX_RETRIES_PER_INDIVIDUAL = 5
    TEST_TIMEOUT_SECONDS = 50

    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm: BaseClient,
        direction_llm: Optional[BaseClient] = None,
        cooperative_llms: Optional[List[BaseClient]] = None,
    ) -> None:
        self.cfg = cfg
        self.root_dir = root_dir
        self.generator_llm = generator_llm
        self.direction_llm = direction_llm or generator_llm
        self.cooperative_llms = cooperative_llms or [self.direction_llm]

        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.k = 5
        self.lamda = 0.7

        self.mediocrity = None
        self.elitist = None
        self.long_term_direction_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None

        self.predictor_population = []
        self.generated_codes = []
        self.generated_outputs: List[torch.Tensor] = []

        # Task prompts (deduplicated)
        self.task_prompts: List[str] = [
            (
                "Based on the seed code above, generate a modified version of the heuristics function. "
                "You must ONLY change the calculations related to 'current_distance_matrix', "
                "'delivery_node_demands', and 'current_load'. Do NOT change any calculations or logic related "
                "to other inputs like delivery_node_demands_open, current_load_open, time_windows, arrival_times, "
                "pickup_node_demands, or current_length. Keep the function signature, structure, variable names, "
                "and all other parts identical. Ensure the output remains a heuristic score matrix of shape "
                "(pomo_size, N+1) with positive scores for promising edges and negative for undesirable ones."
            ),
            (
                "Based on the seed code above, generate a modified version of the heuristics function. "
                "You must ONLY change the calculations related to 'delivery_node_demands_open' and 'current_load_open'. "
                "Do NOT change any calculations or logic related to other inputs like current_distance_matrix, "
                "delivery_node_demands, current_load, time_windows, arrival_times, pickup_node_demands, or current_length. "
                "Keep the function signature, structure, variable names, and all other parts identical. Ensure the output "
                "remains a heuristic score matrix of shape (pomo_size, N+1) with positive scores for promising edges and "
                "negative for undesirable ones."
            ),
            (
                "Based on the seed code above, generate a modified version of the heuristics function. "
                "You must ONLY change the calculations related to 'time_windows' and 'arrival_times'. "
                "Do NOT change any calculations or logic related to other inputs like current_distance_matrix, "
                "delivery_node_demands, current_load, delivery_node_demands_open, current_load_open, pickup_node_demands, "
                "or current_length. Keep the function signature, structure, variable names, and all other parts identical. "
                "Ensure the output remains a heuristic score matrix of shape (pomo_size, N+1) with positive scores for "
                "promising edges and negative for undesirable ones."
            ),
            (
                "Based on the seed code above, generate a modified version of the heuristics function. "
                "You must ONLY change the calculations related to 'pickup_node_demands'. Do NOT change any calculations or "
                "logic related to other inputs like current_distance_matrix, delivery_node_demands, current_load, "
                "delivery_node_demands_open, current_load_open, time_windows, arrival_times, or current_length. "
                "Keep the function signature, structure, variable names, and all other parts identical. Ensure the output "
                "remains a heuristic score matrix of shape (pomo_size, N+1) with positive scores for promising edges and "
                "negative for undesirable ones."
            ),
            (
                "Based on the seed code above, generate a modified version of the heuristics function. "
                "You must ONLY change the calculations related to 'current_length'. Do NOT change any calculations or logic "
                "related to other inputs like current_distance_matrix, delivery_node_demands, current_load, "
                "delivery_node_demands_open, current_load_open, time_windows, arrival_times, or pickup_node_demands. "
                "Keep the function signature, structure, variable names, and all other parts identical. Ensure the output "
                "remains a heuristic score matrix of shape (pomo_size, N+1) with positive scores for promising edges and "
                "negative for undesirable ones."
            ),
            (
                "Based on the seed code above, generate a modified version of the heuristics function. "
                "You may change the calculations related to all inputs: 'current_distance_matrix', 'delivery_node_demands', "
                "'current_load', 'delivery_node_demands_open', 'current_load_open', 'time_windows', 'arrival_times', "
                "'pickup_node_demands', and 'current_length'. Keep the function signature, structure, variable names, and "
                "all other parts identical. Ensure the output remains a heuristic score matrix of shape (pomo_size, N+1) "
                "with positive scores for promising edges and negative for undesirable ones."
            ),
        ]
        self.num_tasks = len(self.task_prompts)
        self.phase_budget = self.cfg.max_fe // max(1, self.num_tasks)

        self._init_prompts()  # sets self.current_task
        self._init_population()

        # Flags to optionally print prompts once
        self.print_crossover_prompt = False
        self.print_mutate_prompt = False
        self.print_short_term_direction_prompt = False
        self.print_long_term_direction_prompt = False

    # -------------------------------------------------------------------------
    # Initialization helpers
    # -------------------------------------------------------------------------
    def _init_prompts(self) -> None:
        problem = self.cfg.problem
        self.problem = problem.problem_name
        self.problem_desc = problem.description
        self.problem_size = problem.problem_size
        self.func_name = problem.func_name
        self.obj_type = problem.obj_type
        self.pro = problem.pro
        self.alg = problem.alg

        logging.info("Problem: %s", self.problem)
        logging.info("Description: %s", self.problem_desc)
        logging.info("Function name: %s", self.func_name)

        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"

        problem_prompt_path = f"{self.prompt_dir}/{self.problem}"
        self.seed_func = file_to_string(f"{problem_prompt_path}/seed_func.txt")
        self.func_signature = file_to_string(f"{problem_prompt_path}/func_signature.txt")
        self.func_desc = file_to_string(f"{problem_prompt_path}/func_desc.txt")

        ext_path = f"{problem_prompt_path}/external_knowledge.txt"
        if os.path.exists(ext_path):
            self.external_knowledge = file_to_string(ext_path)
            self.long_term_direction_str = self.external_knowledge
        else:
            self.external_knowledge = ""

        # Common prompts
        self.system_core_abstraction = file_to_string(f"{self.prompt_dir}/common/system_core_abstraction.txt")
        self.user_core_abstraction = file_to_string(f"{self.prompt_dir}/common/user_core_abstraction.txt")
        self.system_generator_prompt = file_to_string(f"{self.prompt_dir}/common/system_generator.txt")
        self.system_direction_prompt = file_to_string(f"{self.prompt_dir}/common/system_direction.txt")
        self.user_direction_st_prompt = file_to_string(f"{self.prompt_dir}/common/user_direction_st.txt")
        self.user_direction_lt_prompt = file_to_string(f"{self.prompt_dir}/common/user_direction_lt.txt")
        self.crossover_prompt = file_to_string(f"{self.prompt_dir}/common/crossover.txt")
        self.mutation_prompt = file_to_string(f"{self.prompt_dir}/common/mutation.txt")
        self.user_generator_prompt = file_to_string(f"{self.prompt_dir}/common/user_generator.txt").format(
            func_name=self.func_name,
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
        )
        self.seed_prompt = file_to_string(f"{self.prompt_dir}/common/seed.txt").format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        self.current_task = self.task_prompts[0]

    def _init_population(self) -> None:
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed code extracted.")

        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        if not self.seed_ind.get("exec_success"):
            raise RuntimeError("Seed function is invalid. See stdout files for details.")

        self._update_iter()

        # Build initial messages
        system = self.system_generator_prompt
        user = (
            self.user_generator_prompt
            + "\n"
            + self.seed_prompt
            + "\n"
            + self.long_term_direction_str
            + "\n\n[Specific Modification Task]\n"
            + self.current_task
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Generating initial population...")

        responses = self.generator_llm.multi_chat_completion(
            [messages], self.cfg.init_pop_size, temperature=self.generator_llm.temperature + 0.3
        )
        population = [
            self._response_to_individual(response, response_id, messages, self.generator_llm)
            for response_id, response in enumerate(responses)
        ]

        self.population = self.evaluate_population(population)
        self._update_iter()

    # -------------------------------------------------------------------------
    # Individual construction and evaluation
    # -------------------------------------------------------------------------
    def _write_text(self, path: str, text: str) -> None:
        with open(path, "w") as f:
            f.write(text + ("\n" if not text.endswith("\n") else ""))

    def _safe_write_code(self, code_path: str, code: str) -> None:
        with open(code_path, "w") as f:
            f.write(code)

    def _response_to_individual(
        self,
        response: str,
        response_id: int,
        original_messages: List[dict],
        llm: BaseClient,
        file_name: Optional[str] = None,
    ) -> dict:
        """Convert an LLM response into an executable individual, retrying with feedback if needed."""
        file_stub = file_name or f"problem_iter{self.iteration}_response{response_id}"
        self._write_text(f"{file_stub}.txt", response)

        code = extract_code_from_generator(response)
        individual = {
            "stdout_filepath": f"{file_stub}_stdout.txt",
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }

        retries = 0
        while retries <= self.MAX_RETRIES_PER_INDIVIDUAL:
            failed = False
            feedback_parts = []

            if not code:
                failed = True
                feedback_parts.append("No code block was extracted.")
            else:
                self._safe_write_code(individual["code_path"], code)
                success, output = self.test_code_on_sample(individual)
                if not success:
                    failed = True
                    feedback_parts.append(
                        f"Correctness failed: {individual.get('traceback_msg', 'Unknown error')}"
                    )
                else:
                    # Prevent exact-output duplicates
                    too_identical = any(
                        (output.shape == existing.shape) and torch.equal(output, existing)
                        for existing in self.generated_outputs
                    )
                    if too_identical:
                        failed = True
                        feedback_parts.append("Output identical to an existing candidate.")

            if not failed:
                self.generated_outputs.append(output)
                individual["code"] = code
                return individual

            # Retry: request a new response
            new_messages = list(original_messages)  # shallow copy; original_messages are immutable dicts
            new_response = llm.multi_chat_completion([new_messages], 1)[0]
            code = extract_code_from_generator(new_response)
            individual["code"] = code
            if code:
                self._safe_write_code(individual["code_path"], code)

            retries += 1
            logging.info("Retrying code synthesis (attempt %d/%d)...", retries, self.MAX_RETRIES_PER_INDIVIDUAL)

        return self._mark_invalid(individual, "Failed after retries on sample")

    def _mark_invalid(self, individual: dict, traceback_msg: str) -> dict:
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual

    def evaluate_population(self, population: List[dict]) -> List[dict]:
        """Run each individual's code, then parse objective from stdout."""
        inner_runs: List[Optional[subprocess.Popen]] = []

        for idx, ind in enumerate(population):
            self.function_evals += 1
            code = ind.get("code")
            if not code:
                population[idx] = self._mark_invalid(ind, "Invalid response: no code.")
                inner_runs.append(None)
                continue

            logging.info("Iteration %d: launching code %d", self.iteration, idx)
            try:
                process = self._run_code(ind, idx)
                inner_runs.append(process)
            except Exception as e:
                logging.info("Launch error for response_id %d: %s", idx, e)
                population[idx] = self._mark_invalid(ind, str(e))
                inner_runs.append(None)

        # Wait and collect
        for idx, process in enumerate(inner_runs):
            if process is None:
                continue

            try:
                process.communicate(timeout=self.cfg.timeout)
            except subprocess.TimeoutExpired as e:
                logging.info("Timeout for response_id %d: %s", idx, e)
                population[idx] = self._mark_invalid(population[idx], "Evaluation timeout.")
                process.kill()
                continue

            ind = population[idx]
            stdout_path = ind["stdout_filepath"]
            try:
                with open(stdout_path, "r") as f:
                    stdout_str = f.read()
            except Exception as e:
                population[idx] = self._mark_invalid(ind, f"Failed to read stdout: {e}")
                continue

            tb = filter_traceback(stdout_str)
            if tb:
                population[idx] = self._mark_invalid(ind, tb)
                continue

            obj = self._parse_objective(stdout_str, self.obj_type)
            if obj is None:
                population[idx] = self._mark_invalid(ind, "Invalid objective in stdout.")
            else:
                ind["obj"] = obj
                ind["exec_success"] = True

        return population

    def _run_code(self, individual: dict, response_idx: int) -> subprocess.Popen:
        """Write candidate code into the problem file and run the evaluator."""
        self._safe_write_code(self.output_file, individual["code"])
        stdout_fp = individual["stdout_filepath"]

        eval_file_path = f"{self.root_dir}/problems/{self.problem}/eval.py"
        f = open(stdout_fp, "w")
        process = subprocess.Popen(
            [sys.executable, "-u", eval_file_path, f"{self.problem_size}", self.root_dir, "train"],
            stdout=f,
            stderr=f,
        )
        block_until_running(stdout_fp, log_status=True, iter_num=self.iteration, response_id=response_idx)
        return process

    # -------------------------------------------------------------------------
    # Iteration / selection / directions
    # -------------------------------------------------------------------------
    def _update_iter(self) -> None:
        population = self.population or []
        for i, ind in enumerate(population):
            logging.info("Iteration %d, response_id %d: obj=%s", self.iteration, i, ind.get("obj"))

        valid = [(i, ind["obj"]) for i, ind in enumerate(population) if ind.get("exec_success")]
        if not valid:
            raise RuntimeError("No valid individuals after evaluation.")

        indices, objs = zip(*valid)
        best_obj = min(objs)
        best_idx = indices[int(np.argmin(objs))]

        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_idx]["code"]
            self.best_code_path_overall = population[best_idx]["code_path"]

        if self.elitist is None or best_obj < self.elitist.get("obj", float("inf")):
            self.elitist = population[best_idx]

        logging.info("Iteration %d: elitist obj=%s", self.iteration, self.elitist["obj"])
        logging.info("Best overall obj=%s, code=%s", self.best_obj_overall, self.best_code_path_overall)
        logging.info("Function evals=%s", self.function_evals)
        self.iteration += 1

    def rank_select(self, population: List[dict]) -> Optional[List[dict]]:
        """Rank-based selection. Returns 2 * pop_size individuals as parents."""
        valid = [ind for ind in population if ind.get("exec_success")]
        if len(valid) < 2:
            return None
        valid = sorted(valid, key=lambda x: x["obj"])

        ranks = list(range(len(valid)))
        probs = np.array([1.0 / (r + 1 + len(valid)) for r in ranks], dtype=np.float64)
        probs /= probs.sum()

        selected: List[dict] = []
        trial = 0
        while len(selected) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(valid, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected.extend(parents)
            if trial > 1000:
                return None
        return selected

    def core_abstraction(self) -> str:
        """Summarize core components from top-k unique individuals."""
        pop_sorted = sorted(self.population, key=lambda x: x["obj"])
        # Ensure elitist first, and collect unique by obj
        unique: List[dict] = []
        seen = set()
        if self.elitist is not None:
            unique.append(self.elitist)
            seen.add(self.elitist["obj"])
        for ind in pop_sorted:
            if ind["obj"] not in seen:
                unique.append(ind)
                seen.add(ind["obj"])
            if len(unique) == self.k:
                break
        # Pad if needed
        while len(unique) < min(self.k, len(pop_sorted)):
            unique.append(pop_sorted[len(unique)])

        if len(unique) < 5:
            # Ensure indices exist for prompt formatting
            unique = (unique + pop_sorted)[:5]

        system = self.system_core_abstraction
        user = self.user_core_abstraction.format(
            func_name=self.func_name,
            alg=self.alg,
            pro=self.pro,
            func_desc=self.func_desc,
            code_0=unique[0]["code"],
            code_1=unique[1]["code"],
            code_2=unique[2]["code"],
            code_3=unique[3]["code"],
            code_4=unique[4]["code"],
        )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        responses = []
        for llm in self.cooperative_llms:
            try:
                resp = llm.multi_chat_completion([message])[0]
            except Exception as e:
                resp = f"# Error from LLM: {str(e)}"
            responses.append(resp)

        merged = "\n\n".join([f"# LLM-{i + 1} Response:\n{r}" for i, r in enumerate(responses)])
        return merged

    def _gen_short_term_direction_prompt(
        self, ind1: dict, ind2: dict, core_abstraction: str
    ) -> Tuple[List[dict], str, str]:
        if ind1["obj"] == ind2["obj"]:
            raise ValueError("Two individuals to crossover have the same objective value.")

        better, worse = (ind1, ind2) if ind1["obj"] < ind2["obj"] else (ind2, ind1)

        system = self.system_direction_prompt
        user = self.user_direction_st_prompt.format(
            func_name=self.func_name,
            func_desc=self.func_desc,
            problem_desc=self.problem_desc,
            component=core_abstraction,
            worse_code=filter_code(worse["code"]),
            better_code=filter_code(better["code"]),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_short_term_direction_prompt:
            logging.info("Short-term direction prompt:\n%s\n%s", system, user)
            self.print_short_term_direction_prompt = False
        return messages, filter_code(worse["code"]), filter_code(better["code"])

    def short_term_direction(
        self, parents: List[dict], core_abstraction: str
    ) -> Tuple[List[str], List[str], List[str]]:
        msgs, worse_codes, better_codes = [], [], []
        for i in range(0, len(parents), 2):
            m, w, b = self._gen_short_term_direction_prompt(parents[i], parents[i + 1], core_abstraction)
            msgs.append(m)
            worse_codes.append(w)
            better_codes.append(b)
        responses = self.direction_llm.multi_chat_completion(msgs)
        return responses, worse_codes, better_codes

    def long_term_direction(self, short_term_directions: List[str], core_abstraction: str) -> None:
        system = self.system_direction_prompt
        user = self.user_direction_lt_prompt.format(
            problem_desc=self.problem_desc,
            prior_direction=self.long_term_direction_str,
            component=core_abstraction,
            func_name=self.func_name,
            new_direction="\n".join(short_term_directions),
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_long_term_direction_prompt:
            logging.info("Long-term direction prompt:\n%s\n%s", system, user)
            self.print_long_term_direction_prompt = False

        self.long_term_direction_str = self.direction_llm.multi_chat_completion([messages])[0]
        self._write_text(f"problem_iter{self.iteration}_short_term_directions.txt", "\n".join(short_term_directions))
        self._write_text(f"problem_iter{self.iteration}_long_term_direction.txt", self.long_term_direction_str)

    def crossover(self, st_tuple: Tuple[List[str], List[str], List[str]]) -> List[dict]:
        directions, worse_codes, better_codes = st_tuple
        messages_lst = []
        for direction, worse_code, better_code in zip(directions, worse_codes, better_codes):
            system = self.system_generator_prompt
            user = self.crossover_prompt.format(
                user_generator=self.user_generator_prompt,
                func_signature0=self.func_signature.format(version=0),
                func_signature1=self.func_signature.format(version=1),
                worse_code=worse_code,
                better_code=better_code,
                direction=direction,
                func_name=self.func_name,
            )
            if hasattr(self, "current_task"):
                user += "\n\n[Specific Modification Task]\n" + self.current_task
            messages_lst.append([{"role": "system", "content": system}, {"role": "user", "content": user}])

            if self.print_crossover_prompt:
                logging.info("Crossover prompt:\n%s\n%s", system, user)
                self.print_crossover_prompt = False

        response_lst = self.generator_llm.multi_chat_completion(messages_lst)
        crossed = [
            self._response_to_individual(resp, rid, messages_lst[rid], self.generator_llm)
            for rid, resp in enumerate(response_lst)
        ]
        assert len(crossed) == self.cfg.pop_size
        return crossed

    def mutate(self) -> List[dict]:
        """Elitist-based mutation."""
        system = self.system_generator_prompt
        user = self.mutation_prompt.format(
            user_generator=self.user_generator_prompt,
            direction=self.long_term_direction_str + self.external_knowledge,
            func_signature1=self.func_signature.format(version=1),
            elitist_code=filter_code(self.elitist["code"]),
            func_name=self.func_name,
        )
        if hasattr(self, "current_task"):
            user += "\n\n[Specific Modification Task]\n" + self.current_task
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        if self.print_mutate_prompt:
            logging.info("Mutation prompt:\n%s\n%s", system, user)
            self.print_mutate_prompt = False

        n = max(1, int(self.cfg.pop_size * self.mutation_rate))
        responses = self.generator_llm.multi_chat_completion([messages], n)
        return [
            self._response_to_individual(resp, rid, messages, self.generator_llm)
            for rid, resp in enumerate(responses)
        ]

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def evolve(self):
        for task_id, task in enumerate(self.task_prompts):
            self.current_task = task
            if task_id > 0:
                # Seed from previous best and reinitialize
                self.seed_func = self.best_code_overall
                self._init_population()

            phase_max_evals = self.function_evals + self.phase_budget
            while self.function_evals < self.cfg.max_fe:
                if all(not ind.get("exec_success") for ind in self.population):
                    raise RuntimeError("All individuals invalid. See stdout files.")

                population_to_select = (
                    self.population
                    if (self.elitist is None or self.elitist in self.population)
                    else [self.elitist] + self.population
                )
                selected = self.rank_select(population_to_select)
                if selected is None:
                    raise RuntimeError("Selection failed.")

                if self.function_evals <= self.lamda * self.cfg.max_fe:
                    core_abs = self.core_abstraction()
                    st_tuple = self.short_term_direction(selected, core_abs)
                    crossed = self.crossover(st_tuple)
                    self.population = self.evaluate_population(crossed)
                    self._update_iter()
                    self.long_term_direction(list(st_tuple[0]), core_abs)

                mutated = self.mutate()
                self.population.extend(self.evaluate_population(mutated))
                self._update_iter()

                # Optional budget per phase (kept for compatibility though not currently used)
                _ = phase_max_evals

            # Persist best as new seed
            if self.best_code_overall:
                modified_code = self.best_code_overall.replace("heuristics_v2", "heuristics_v1")
                seed_file = os.path.join(self.root_dir, "prompts", "mt_routefinder_unified", "seed_func.txt")
                self._safe_write_code(seed_file, modified_code)
                self.seed_func = modified_code

            self.function_evals = 0

        return self.best_code_overall, self.best_code_path_overall

    # -------------------------------------------------------------------------
    # Sample loading / testing
    # -------------------------------------------------------------------------
    def load_sample_data(self) -> dict:
        import torch as _torch

        sample_path = os.path.join(self.root_dir, "prompts/mt_pomo_tw/data_VRPTW_100_1.pt")
        if not os.path.exists(sample_path):
            raise FileNotFoundError(f"Sample data file not found: {sample_path}")

        data = _torch.load(sample_path, map_location="cpu")
        expected_keys = [
            "depot_xy",
            "node_xy",
            "node_demand",
            "node_earlyTW",
            "node_lateTW",
            "node_serviceTime",
            "route_open",
            "route_length_limit",
        ]
        missing = set(expected_keys) - set(data.keys())
        if missing:
            raise ValueError(f"Missing keys in data: {missing}")

        if data["depot_xy"].shape[1:] != torch.Size([1, 2]):
            raise ValueError(f"Invalid depot_xy shape: {data['depot_xy'].shape}")
        if data["node_xy"].shape[1:] != torch.Size([100, 2]):
            raise ValueError(f"Invalid node_xy shape: {data['node_xy'].shape}")
        return data

    def test_code_on_sample(self, individual: dict) -> Tuple[bool, Optional[torch.Tensor]]:
        code = individual["code"]

        def _timeout_handler(signum, frame):
            raise TimeoutError("Code execution timeout")

        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(self.TEST_TIMEOUT_SECONDS)
        try:
            sample = self.load_sample_data()
            ns = {}
            exec(code, ns)
            func = ns.get("heuristics_v2")
            if not callable(func):
                return False, None

            depot_xy = sample["depot_xy"].squeeze(0)  # (1, 2)
            node_xy = sample["node_xy"].squeeze(0)    # (100, 2)
            node_demand = sample["node_demand"].squeeze(0)
            node_earlyTW = sample["node_earlyTW"].squeeze(0)
            node_lateTW = sample["node_lateTW"].squeeze(0)

            all_xy = torch.cat([depot_xy, node_xy], dim=0)  # (101, 2)
            device = all_xy.device

            pomo_size = node_xy.shape[0]
            current_pos = depot_xy.repeat(pomo_size, 1)  # (P, 2)
            current_distance_matrix = torch.norm(
                current_pos.unsqueeze(1) - all_xy.unsqueeze(0), dim=-1
            )  # (P, N+1)

            # demands with depot=1 as neutral base to keep shapes consistent
            all_demands = torch.cat([torch.ones(1, device=device), node_demand], dim=0)
            delivery_node_demands = torch.where(all_demands > 0, all_demands, torch.zeros((), device=device))
            pickup_node_demands = torch.where(all_demands < 0, all_demands, torch.zeros((), device=device))

            max_late = node_lateTW.max().item()
            depot_tw = torch.tensor([0.0, max_late], device=device)
            time_windows = torch.stack([node_earlyTW, node_lateTW], dim=-1)
            time_windows = torch.cat([depot_tw.unsqueeze(0), time_windows], dim=0)  # (N+1, 2)

            current_load = torch.zeros(pomo_size, device=device)
            current_length = torch.zeros(pomo_size, device=device)
            current_time = torch.zeros(pomo_size, device=device)

            # Call function. We reuse delivery_node_demands for the "open" variant as a simple baseline.
            output = func(
                current_distance_matrix,
                delivery_node_demands,
                current_load,
                delivery_node_demands,
                current_load,
                time_windows,
                current_distance_matrix,
                pickup_node_demands,
                current_length,
            )
            if not isinstance(output, torch.Tensor):
                return False, None
            expected_shape = (pomo_size, all_xy.shape[0])
            if output.shape != expected_shape:
                return False, None
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                return False, None
            return True, output
        except TimeoutError:
            return False, None
        except Exception:
            return False, None
        finally:
            signal.alarm(0)

    # -------------------------------------------------------------------------
    # Utilities
    # -------------------------------------------------------------------------
    @staticmethod
    def _parse_objective(stdout_str: str, obj_type: str) -> Optional[float]:
        """Parse objective from the last non-empty line."""
        lines = [ln.strip() for ln in stdout_str.splitlines() if ln.strip()]
        if not lines:
            return None
        last = lines[-1]
        try:
            val = float(last)
            return val if obj_type == "min" else -val
        except Exception:
            # Fallback: try to find the last float anywhere
            import re

            floats = re.findall(r"[-+]?\d*\.\d+|\d+", last)
            if not floats:
                return None
            val = float(floats[-1])
            return val if obj_type == "min" else -val
