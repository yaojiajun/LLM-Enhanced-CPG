from typing import Optional
import logging
import subprocess
import numpy as np
import os
import sys
from omegaconf import DictConfig
import difflib
import signal
import math
from utils.utils import *
from utils.llm_client.base import BaseClient

class Hercules:
    def __init__(
        self,
        cfg: DictConfig,
        root_dir: str,
        generator_llm: BaseClient,
        direction_llm: Optional[BaseClient] = None,
        cooperative_llms: Optional[list[BaseClient]] = None,  # New: list of LLMs for collaboration
    ) -> None:
        self.cfg = cfg
        self.generator_llm = generator_llm
        self.direction_llm = direction_llm or generator_llm
        self.cooperative_llms = cooperative_llms or [self.direction_llm]  # fallback to single LLM
        self.root_dir = root_dir

        self.mutation_rate = cfg.mutation_rate
        self.iteration = 0
        self.function_evals = 0
        self.k = 2
        self.lamda = 0.7
        self.similarity_threshold = 1
        self.mediocrity = None
        self.elitist = None
        self.elitist1 = None
        self.elitist2 = None
        self.long_term_direction_str = ""
        self.best_obj_overall = None
        self.best_code_overall = None
        self.best_code_path_overall = None
        self.predictor_population = []
        self.generated_codes = []

        self.init_prompt()
        self.init_population()


    def init_prompt(self) -> None:
        self.problem = self.cfg.problem.problem_name
        self.problem_desc = self.cfg.problem.description
        self.problem_size = self.cfg.problem.problem_size
        self.func_name = self.cfg.problem.func_name
        self.obj_type = self.cfg.problem.obj_type
        self.pro = self.cfg.problem.pro
        self.alg = self.cfg.problem.alg

        logging.info("Problem: " + self.problem)
        logging.info("Problem description: " + self.problem_desc)
        logging.info("Function name: " + self.func_name)
        
        self.prompt_dir = f"{self.root_dir}/prompts"
        self.output_file = f"{self.root_dir}/problems/{self.problem}/gpt.py"
        
        # Loading all text prompts
        # Problem-specific prompt components
        prompt_path_suffix = ""
        problem_prompt_path = f'{self.prompt_dir}/{self.problem}{prompt_path_suffix}'
        self.seed_func = file_to_string(f'{problem_prompt_path}/seed_func.txt')
        self.func_signature = file_to_string(f'{problem_prompt_path}/func_signature.txt')
        self.func_desc = file_to_string(f'{problem_prompt_path}/func_desc.txt')
        if os.path.exists(f'{problem_prompt_path}/external_knowledge.txt'):
            self.external_knowledge = file_to_string(f'{problem_prompt_path}/external_knowledge.txt')
            self.long_term_direction_str = self.external_knowledge
        else:
            self.external_knowledge = ""
        
        
        # Common prompts
        self.system_core_abstraction = file_to_string(f'{self.prompt_dir}/common/system_core_abstraction.txt')
        self.user_core_abstraction  = file_to_string(f'{self.prompt_dir}/common/user_core_abstraction.txt')
        self.system_generator_prompt = file_to_string(f'{self.prompt_dir}/common/system_generator.txt')
        self.system_direction_prompt = file_to_string(f'{self.prompt_dir}/common/system_direction.txt')
        self.user_direction_st_prompt = file_to_string(f'{self.prompt_dir}/common/user_direction_st.txt')
        self.user_direction_lt_prompt = file_to_string(f'{self.prompt_dir}/common/user_direction_lt.txt') # long-term direction
        self.crossover_prompt = file_to_string(f'{self.prompt_dir}/common/crossover.txt')
        self.mutataion_prompt = file_to_string(f'{self.prompt_dir}/common/mutation.txt')
        self.user_generator_prompt = file_to_string(f'{self.prompt_dir}/common/user_generator.txt').format(
            func_name=self.func_name, 
            problem_desc=self.problem_desc,
            func_desc=self.func_desc,
            )
        self.seed_prompt = file_to_string(f'{self.prompt_dir}/common/seed.txt').format(
            seed_func=self.seed_func,
            func_name=self.func_name,
        )

        # Flag to print prompts
        self.print_crossover_prompt = False # Print crossover prompt for the first iteration
        self.print_mutate_prompt = False # Print mutate prompt for the first iteration
        self.print_short_term_direction_prompt = False # Print short-term direction prompt for the first iteration
        self.print_long_term_direction_prompt = False # Print long-term direction prompt for the first iteration


    def init_population(self) -> None:
        # Evaluate the seed function, and set it as Elite
        logging.info("Evaluating seed function...")
        code = extract_code_from_generator(self.seed_func).replace("v1", "v2")
        logging.info("Seed function code: \n" + code)
        seed_ind = {
            "stdout_filepath": f"problem_iter{self.iteration}_stdout0.txt",
            "code_path": f"problem_iter{self.iteration}_code0.py",
            "code": code,
            "response_id": 0,
        }
        self.seed_ind = seed_ind
        self.population = self.evaluate_population([seed_ind])

        # If seed function is invalid, stop
        if not self.seed_ind["exec_success"]:
            raise RuntimeError(f"Seed function is invalid. Please check the stdout file in {os.getcwd()}.")

        self.update_iter()
        
        # Generate responses
        system = self.system_generator_prompt
        user = self.user_generator_prompt + "\n" + self.seed_prompt + "\n" + self.long_term_direction_str
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        logging.info("Initial Population Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)

        responses = self.generator_llm.multi_chat_completion([messages], self.cfg.init_pop_size, temperature = self.generator_llm.temperature + 0.3) # Increase the temperature for diverse initial population
        population = [self.response_to_individual(response, response_id, messages, self.generator_llm) for response_id, response in enumerate(responses)]

        # Run code and evaluate population
        population = self.evaluate_population(population)

        # Update iteration
        self.population = population
        self.update_iter()


    def response_to_individual(self, response: str, response_id: int, original_messages: list[dict], llm: BaseClient, file_name: str=None) -> dict:
        """
        Convert response to individual
        """
        # Write response to file
        file_name = f"problem_iter{self.iteration}_response{response_id}.txt" if file_name is None else file_name + ".txt"
        with open(file_name, 'w') as file:
            file.writelines(response + '\n')

        code = extract_code_from_generator(response)

        # Extract code and description from response
        std_out_filepath = f"problem_iter{self.iteration}_stdout{response_id}.txt" if file_name is None else file_name + "_stdout.txt"

        individual = {
            "stdout_filepath": std_out_filepath,
            "code_path": f"problem_iter{self.iteration}_code{response_id}.py",
            "code": code,
            "response_id": response_id,
        }

        # 将初始代码写入文件
        with open(individual["code_path"], 'w') as f:
            f.write(code)

        max_retries = 5  # 减少重试次数，避免无限循环
        retries = 0

        while retries < max_retries:
            failed = False
            feedback_parts = []

            if code is None:
                failed = True

            else:
                # Write code only if not None
                with open(individual["code_path"], 'w') as f:
                    f.write(code)

                if not self.test_code_on_sample(individual):
                    failed = True
                    feedback_parts.append(f"Correctness failed: {individual.get('traceback_msg', 'Unknown error')}")

                too_similar = False
                similarity = 0.0
                for existing_code in self.generated_codes:
                    similarity = difflib.SequenceMatcher(None, code, existing_code).ratio()
                    if similarity > self.similarity_threshold:
                        too_similar = True
                        break
                if too_similar:
                    failed = True
                    feedback_parts.append(f"Code too similar to existing (similarity: {similarity:.3f}).")

            if not failed:
                self.generated_codes.append(code)
                return individual  

            # 重新生成代码
            feedback = "Previous code issues: " + " ".join(feedback_parts) + " Requirements: Function must be named 'heuristics_v2'. Must return torch.Tensor of shape (pomo_size, N+1) without NaN/Inf. Handle inputs correctly. Generate a different implementation."
            
            # 复制并附加反馈到最后一个消息（用户提示）
            new_messages = original_messages.copy()
            # new_messages[-1]["content"] += "\n" + feedback
            new_response = llm.multi_chat_completion([new_messages], 1)[0]

            # 提取新代码
            code = extract_code_from_generator(new_response)
            individual["code"] = code

            # 更新代码文件（覆盖当前 individual 的文件）
            with open(individual["code_path"], 'w') as f:
                f.write(code)

            retries += 1
            logging.info(f"Iteration {retries} code code code...")
        # 如果所有重试失败，标记为无效
        return self.mark_invalid_individual(individual, "Failed after retries on sample")

    def mark_invalid_individual(self, individual: dict, traceback_msg: str) -> dict:
        """
        Mark an individual as invalid.
        """
        individual["exec_success"] = False
        individual["obj"] = float("inf")
        individual["traceback_msg"] = traceback_msg
        return individual


    def evaluate_population(self, population: list[dict]) -> list[float]:
        """
        Evaluate population by running code in parallel and computing objective values.
        """
        inner_runs = []
        # Run code to evaluate
        for response_id in range(len(population)):
            self.function_evals += 1
            # Skip if response is invalid
            if population[response_id]["code"] is None:
                population[response_id] = self.mark_invalid_individual(population[response_id], "Invalid response!")
                inner_runs.append(None)
                continue
            
            logging.info(f"Iteration {self.iteration}: Running Code {response_id}")
            
            try:
                process = self._run_code(population[response_id], response_id)
                inner_runs.append(process)
            except Exception as e: # If code execution fails
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_runs.append(None)
        
        # Update population with objective values
        for response_id, inner_run in enumerate(inner_runs):
            if inner_run is None: # If code execution fails, skip
                continue
            try:
                inner_run.communicate(timeout=self.cfg.timeout) # Wait for code execution to finish
            except subprocess.TimeoutExpired as e:
                logging.info(f"Error for response_id {response_id}: {e}")
                population[response_id] = self.mark_invalid_individual(population[response_id], str(e))
                inner_run.kill()
                continue

            individual = population[response_id]
            stdout_filepath = individual["stdout_filepath"]
            with open(stdout_filepath, 'r') as f:  # read the stdout file
                stdout_str = f.read() 
            traceback_msg = filter_traceback(stdout_str)
            
            individual = population[response_id]
            # Store objective value for each individual
            if traceback_msg == '':
                try:
                    lines = stdout_str.split('\n')

                    def extract_average(lines, keyword):
                        for i in range(len(lines) - 1):
                            if keyword in lines[i]:
                                next_line = lines[i + 1].strip()
                                match = re.search(r'^[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?$', next_line)
                                if match:
                                    return float(match.group(0))
                        return None

                    cvrp_avg = extract_average(lines, "[*] Average for CVRP:")
                    ovrp_avg = extract_average(lines, "[*] Average for OVRP:")
                    overall_avg = extract_average(lines, "[*] Overall Average (sum of avg_obj / 5):")

                    # 设置默认值以避免缺失键
                    default_value = float("inf") if self.obj_type == "min" else float("-inf")
                    if overall_avg is not None:
                        individual["obj"] = overall_avg if self.obj_type == "min" else -overall_avg
                    else:
                        individual["obj"] = default_value
                        logging.warning(f"Overall Average not found for response_id {response_id}")
                    if cvrp_avg is not None:
                        individual["obj1"] = cvrp_avg if self.obj_type == "min" else -cvrp_avg
                    else:
                        individual["obj1"] = default_value
                        logging.warning(f"CVRP Average not found for response_id {response_id}")

                    if ovrp_avg is not None:
                        individual["obj2"] = ovrp_avg if self.obj_type == "min" else -ovrp_avg
                    else:
                        individual["obj2"] = default_value
                        logging.warning(f"OVRP Average not found for response_id {response_id}")

                    individual["exec_success"] = True
                except Exception as e:
                    population[response_id] = self.mark_invalid_individual(population[response_id],
                                                                           f"Invalid stdout / objective value: {str(e)}")
                    # 确保 obj1 存在
                    population[response_id]["obj1"] = default_value
            else:
                population[response_id] = self.mark_invalid_individual(population[response_id], traceback_msg)
                # 确保 obj1 存在
                population[response_id]["obj1"] = default_value

            #logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {individual['obj']}")
        return population


    def _run_code(self, individual: dict, response_id) -> subprocess.Popen:
        """
        Write code into a file and run eval script.
        """
        logging.debug(f"Iteration {self.iteration}: Processing Code Run {response_id}")
        
        with open(self.output_file, 'w') as file:
            file.writelines(individual["code"] + '\n')

        # Execute the python file with flags
        with open(individual["stdout_filepath"], 'w') as f:
            eval_file_path = f'{self.root_dir}/problems/{self.problem}/eval.py'
            process = subprocess.Popen([sys.executable, '-u', eval_file_path, f'{self.problem_size}', self.root_dir, "train"],
                                        stdout=f, stderr=f)

        block_until_running(individual["stdout_filepath"], log_status=True, iter_num=self.iteration, response_id=response_id)
        return process

    
    def update_iter(self) -> None:
        """
        Update after each iteration
        """
        population = self.population
        for response_id in range(len(population)):
            logging.info(f"Iteration {self.iteration}, response_id {response_id}: Objective value: {population[response_id]['obj']}")

        objs_with_indices = [(i, individual["obj"],individual["obj1"],individual["obj2"]) for i, individual in enumerate(population) if individual["exec_success"] == True]
        indices, objs, objs1, objs2 = zip(*objs_with_indices)
        progress = self.function_evals / self.cfg.max_fe
        self.similarity_threshold = min(1, 0.99 + progress)
        best_obj, best_sample_idx = min(objs), indices[np.argmin(objs)]
        best_obj1, best_sample_idx1 = min(objs1), indices[np.argmin(objs1)]
        best_obj2, best_sample_idx2 = min(objs2), indices[np.argmin(objs2)]
        # update best overall
        if self.best_obj_overall is None or best_obj < self.best_obj_overall:
            self.best_obj_overall = best_obj
            self.best_code_overall = population[best_sample_idx]["code"]
            self.best_code_path_overall = population[best_sample_idx]["code_path"]
        
        # update elitist
        if self.elitist is None or best_obj < self.elitist["obj"]:
            self.elitist = population[best_sample_idx]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist['obj']}")
        if self.elitist1 is None or best_obj1 < self.elitist1["obj1"]:
            self.elitist1 = population[best_sample_idx1]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist1['obj1']}")
        if self.elitist2 is None or best_obj2 < self.elitist2["obj2"]:
            self.elitist2 = population[best_sample_idx2]
            logging.info(f"Iteration {self.iteration}: Elitist: {self.elitist2['obj2']}")
        
        logging.info(f"Iteration {self.iteration} finished...")
        logging.info(f"Best obj: {self.best_obj_overall}, Best Code Path: {self.best_code_path_overall}")
        logging.info(f"Function Evals: {self.function_evals}")
        self.iteration += 1
        
    def rank_select(self, population: list[dict]) -> list[dict]:
        """
        Rank-based selection, select individuals with probability proportional to their rank.
        """

        population = [individual for individual in population if individual["exec_success"]]
        if len(population) < 2:
            return None
        # Sort population by objective value
        population = sorted(population, key=lambda x: x["obj"])
        ranks = [i for i in range(len(population))]
        probs = [1 / (rank + 1 + len(population)) for rank in ranks]
        # Normalize probabilities
        probs = [prob / sum(probs) for prob in probs]
        selected_population = []
        trial = 0
        while len(selected_population) < 2 * self.cfg.pop_size:
            trial += 1
            parents = np.random.choice(population, size=2, replace=False, p=probs)
            if parents[0]["obj"] != parents[1]["obj"]:
                selected_population.extend(parents)
            if trial > 1000:
                return None
        return selected_population

    def core_abstraction(self):
        #Sort according to fitness values
        population = sorted(self.population, key=lambda x: x["obj"])
        population1 = sorted(self.population, key=lambda x: x["obj1"])
        population2 = sorted(self.population, key=lambda x: x["obj2"])
        unique_population = []
        unique_population1 = []
        unique_population2 = []
        seen_objs = set()
        seen_objs1 = set()
        seen_objs2 = set()

        unique_population.append(self.elitist)
        unique_population1.append(self.elitist1)
        unique_population2.append(self.elitist2)

        seen_objs.add(self.elitist["obj"])
        seen_objs1.add(self.elitist1["obj1"])
        seen_objs2.add(self.elitist2["obj2"])

        for individual in population:
            if individual["obj"] not in seen_objs:
                unique_population.append(individual)
                seen_objs.add(individual["obj"])
            if len(unique_population) == self.k:
                break
        i = len(unique_population)
        while len(unique_population) < self.k:
            unique_population.append(population[i])
            i = i +1

        for individual in population1:
            if individual["obj1"] not in seen_objs1:
                unique_population1.append(individual)
                seen_objs1.add(individual["obj1"])
            if len(unique_population1) == self.k:
                break
        i = len(unique_population1)
        while len(unique_population1) < self.k:
            unique_population1.append(population1[i])
            i = i +1

        for individual in population2:
            if individual["obj2"] not in seen_objs2:
                unique_population2.append(individual)
                seen_objs2.add(individual["obj2"])
            if len(unique_population2) == self.k:
                break
        i = len(unique_population2)
        while len(unique_population2) < self.k:
            unique_population2.append(population2[i])
            i = i +1

        system = self.system_core_abstraction
        #self.k = 5
        user = self.user_core_abstraction.format(
            func_name=self.func_name,
            alg=self.alg,
            pro=self.pro,
            func_desc=self.func_desc,
            code_0 = unique_population[0]["code"],
            code_1 = unique_population[1]["code"],
            code_2 = unique_population1[0]["code"],
            code_3 = unique_population1[1]["code"],
            code_4 = unique_population2[0]["code"],
            code_5 = unique_population2[1]["code"],
            # code_3 = unique_population[3]["code"],
            # code_4 = unique_population[4]["code"],
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]

        # New: Use all cooperative LLMs
        responses = []
        for llm in self.cooperative_llms:
            try:
                response = llm.multi_chat_completion([message])[0]
            except Exception as e:
                response = f"# Error from LLM: {str(e)}"
            responses.append(response)

        # Merge responses
        merged = "\n\n".join([
            f"# LLM-{i+1} Response:\n{resp}" for i, resp in enumerate(responses)
        ])
        return merged

    def gen_short_term_direction_prompt(self, ind1: dict, ind2: dict, core_abstraction: str) -> tuple[list[dict], str, str]:
        """
        Short-term direction before crossovering two individuals.
        """
        if ind1["obj"] == ind2["obj"]:
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj"] < ind2["obj"]:
            better_ind, worse_ind = ind1, ind2
        elif ind1["obj"] > ind2["obj"]:
            better_ind, worse_ind = ind2, ind1

        if ind1["obj1"] == ind2["obj1"]:
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj1"] < ind2["obj1"]:
            better_ind1, worse_ind1 = ind1, ind2
        elif ind1["obj1"] > ind2["obj1"]:
            better_ind1, worse_ind1 = ind2, ind1

        if ind1["obj2"] == ind2["obj2"]:
            raise ValueError("Two individuals to crossover have the same objective value!")
        # Determine which individual is better or worse
        if ind1["obj2"] < ind2["obj2"]:
            better_ind2, worse_ind2 = ind1, ind2
        elif ind1["obj2"] > ind2["obj2"]:
            better_ind2, worse_ind2 = ind2, ind1


        worse_code = filter_code(worse_ind["code"])
        better_code = filter_code(better_ind["code"])

        worse_code1 = filter_code(worse_ind1["code"])
        better_code1 = filter_code(better_ind1["code"])

        worse_code2 = filter_code(worse_ind2["code"])
        better_code2 = filter_code(better_ind2["code"])

        system = self.system_direction_prompt
        user = self.user_direction_st_prompt.format(
            func_name = self.func_name,
            func_desc = self.func_desc,
            problem_desc = self.problem_desc,
            component=core_abstraction,
            worse_code=worse_code,
            better_code=better_code,
            worse_code1=worse_code1,
            better_code1=better_code1,
            worse_code2=worse_code2,
            better_code2=better_code2,
            )
        message = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        # Print direction prompt for the first iteration
        if self.print_short_term_direction_prompt:
                logging.info("Short-term direction Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_short_term_direction_prompt = False
        return message, worse_code, better_code


    def short_term_direction(self, population: list[dict], core_abstraction: str) -> tuple[list[str], list[str], list[str]]:
        """
        Short-term direction before crossovering two individuals.
        """
        messages_lst = []
        worse_code_lst = []
        better_code_lst = []
        for i in range(0, len(population), 2):
            # Select two individuals
            parent_1 = population[i]
            parent_2 = population[i+1]
            
            # Short-term direction
            messages, worse_code, better_code = self.gen_short_term_direction_prompt(parent_1, parent_2, core_abstraction)
            messages_lst.append(messages)
            worse_code_lst.append(worse_code)
            better_code_lst.append(better_code)
        
        # Asynchronously generate responses
        response_lst = self.direction_llm.multi_chat_completion(messages_lst)
        return response_lst, worse_code_lst, better_code_lst
    
    def long_term_direction(self, short_term_directions: list[str], core_abstraction: str) -> None:
        """
        Long-term direction before mutation.
        """
        system = self.system_direction_prompt
        user = self.user_direction_lt_prompt.format(
            problem_desc = self.problem_desc,
            prior_direction = self.long_term_direction_str,
            component=core_abstraction,
            func_name=self.func_name,
            new_direction = "\n".join(short_term_directions),
            )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        
        if self.print_long_term_direction_prompt:
            logging.info("Long-term direction Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_long_term_direction_prompt = False
        
        self.long_term_direction_str = self.direction_llm.multi_chat_completion([messages])[0]
        
        # Write directions to file
        file_name = f"problem_iter{self.iteration}_short_term_directions.txt"
        with open(file_name, 'w') as file:
            file.writelines("\n".join(short_term_directions) + '\n')
        
        file_name = f"problem_iter{self.iteration}_long_term_direction.txt"
        with open(file_name, 'w') as file:
            file.writelines(self.long_term_direction_str + '\n')


    def crossover(self, short_term_direction_tuple: tuple[list[str], list[str], list[str]]) -> list[dict]:
        direction_content_lst, worse_code_lst, better_code_lst = short_term_direction_tuple
        messages_lst = []
        progress = self.function_evals / self.cfg.max_fe
        self.similarity_threshold = min(1, 1 + progress)
        for direction, worse_code, better_code in zip(direction_content_lst, worse_code_lst, better_code_lst):
            # Crossover
            system = self.system_generator_prompt
            func_signature0 = self.func_signature.format(version=0)
            func_signature1 = self.func_signature.format(version=1)
            user = self.crossover_prompt.format(
                user_generator = self.user_generator_prompt,
                func_signature0 = func_signature0,
                func_signature1 = func_signature1,
                worse_code = worse_code,
                better_code = better_code,
                direction = direction,
                func_name = self.func_name,
            )
            messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
            messages_lst.append(messages)
            
            # Print crossover prompt for the first iteration
            if self.print_crossover_prompt:
                logging.info("Crossover Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
                self.print_crossover_prompt = False
        
        # Asynchronously generate responses
        response_lst = self.generator_llm.multi_chat_completion(messages_lst)
        crossed_population = [self.response_to_individual(response, response_id, messages_lst[response_id], self.generator_llm) for response_id, response in enumerate(response_lst)]

        assert len(crossed_population) == self.cfg.pop_size
        return crossed_population


    def mutate(self) -> list[dict]:
        """Elitist-based mutation. We only mutate the best individual to generate n_pop new individuals."""
        system = self.system_generator_prompt
        func_signature1 = self.func_signature.format(version=1) 
        user = self.mutataion_prompt.format(
            user_generator = self.user_generator_prompt,
            direction = self.long_term_direction_str + self.external_knowledge,
            func_signature1 = func_signature1,
            elitist_code = filter_code(self.elitist["code"]),
            func_name = self.func_name,
        )
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        if self.print_mutate_prompt:
            logging.info("Mutation Prompt: \nSystem Prompt: \n" + system + "\nUser Prompt: \n" + user)
            self.print_mutate_prompt = False
        responses = self.generator_llm.multi_chat_completion([messages], int(self.cfg.pop_size * self.mutation_rate))
        population = [self.response_to_individual(response, response_id, messages, self.generator_llm) for response_id, response in enumerate(responses)]
        return population


    def evolve(self):
        while self.function_evals < self.cfg.max_fe:
            core_abstraction = ""
            # If all individuals are invalid, stop
            if all([not individual["exec_success"] for individual in self.population]):
                raise RuntimeError(f"All individuals are invalid. Please check the stdout files in {os.getcwd()}.")
            # Select
            population_to_select = self.population if (self.elitist is None or self.elitist in self.population) else [self.elitist] + self.population # add elitist to population for selection
            selected_population = self.rank_select(population_to_select)
            if selected_population is None:
                raise RuntimeError("Selection failed. Please check the population.")
            if self.function_evals <= self.lamda*self.cfg.max_fe:
                # Abstract core components
                core_abstraction = self.core_abstraction()
            # Short-term direction
            short_term_direction_tuple = self.short_term_direction(selected_population, core_abstraction) # (response_lst, worse_code_lst, better_code_lst)
            # Crossover
            crossed_population = self.crossover(short_term_direction_tuple)
            # Evaluate
            self.population = self.evaluate_population(crossed_population)
            # Update
            self.update_iter()
            # Long-term direction
            self.long_term_direction([response for response in short_term_direction_tuple[0]], core_abstraction)
            # Mutate
            mutated_population = self.mutate()
            # Evaluate
            self.population.extend(self.evaluate_population(mutated_population))
            # Update
            self.update_iter()


        return self.best_code_overall, self.best_code_path_overall

    def load_sample_data(self) -> dict:
        """
        加载样本数据，添加更详细的错误处理
        """
        import torch
        import os

        sample_path = os.path.join(self.root_dir, "prompts/mt_pomo_tw/data_VRPTW_100_1.pt")

        try:
            if not os.path.exists(sample_path):
                raise FileNotFoundError(f"Sample data file not found: {sample_path}")

            data = torch.load(sample_path, map_location='cpu')  # 使用CPU避免GPU内存问题

            # 验证标准VRPTW结构
            expected_keys = ['depot_xy', 'node_xy', 'node_demand', 'node_earlyTW', 'node_lateTW', 'node_serviceTime',
                             'route_open', 'route_length_limit']
            missing_keys = set(expected_keys) - set(data.keys())
            if missing_keys:
                raise ValueError(f"Missing keys in data: {missing_keys}")

            # 验证数据形状
            if data['depot_xy'].shape[1:] != torch.Size([1, 2]):
                raise ValueError(f"Invalid depot_xy shape: {data['depot_xy'].shape}")
            if data['node_xy'].shape[1:] != torch.Size([100, 2]):
                raise ValueError(f"Invalid node_xy shape: {data['node_xy'].shape}")

            return data

        except Exception as e:
            raise RuntimeError(f"Failed to load VRPTW sample from {sample_path}: {str(e)}")

    def test_code_on_sample(self, individual: dict) -> bool:
        import torch  # 导入 torch 以在测试代码中使用
        code = individual["code"]

        # 设置简单的超时机制
        def timeout_handler(signum, frame):
            raise TimeoutError("Code execution timeout")

        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(50)  # 3秒超时

        try:
            sample = self.load_sample_data()

            exec_namespace = {}
            exec(code, exec_namespace)
            func = exec_namespace.get('heuristics_v2')
            if not callable(func):
                return False

            depot_xy = sample['depot_xy'].squeeze(0)  # (1,2)
            node_xy = sample['node_xy'].squeeze(0)
            node_demand = sample['node_demand'].squeeze(0)
            node_earlyTW = sample['node_earlyTW'].squeeze(0)
            node_lateTW = sample['node_lateTW'].squeeze(0)

            # Build all positions: depot + nodes
            all_xy = torch.cat([depot_xy, node_xy], dim=0)

            pomo_size = 100
            current_pos = depot_xy.repeat(pomo_size, 1)
            current_distance_matrix = torch.norm(current_pos.unsqueeze(1) - all_xy.unsqueeze(0), dim=-1)

            # delivery_node_demands: depot=0 + nodes
            all_demands = torch.cat([torch.ones(1, device=all_xy.device), node_demand], dim=0)
            delivery_node_demands = torch.where(all_demands > 0, all_demands, torch.tensor(1))
            pickup_node_demands = torch.where(all_demands < 0, all_demands, torch.tensor(-1))
            # time_windows: depot [0, max_late] + nodes
            max_late = node_lateTW.max().item()
            depot_tw = torch.tensor([0.0, max_late], device=all_xy.device)
            time_windows = torch.stack([node_earlyTW, node_lateTW], dim=-1)
            time_windows = torch.cat([depot_tw.unsqueeze(0), time_windows], dim=0)

            # current_load: assume capacity 1.0 for POMO samples
            current_load = torch.ones(pomo_size, device=all_xy.device)
            current_length = current_load
            # current_time: initial 0 for POMO samples
            current_time = torch.zeros(pomo_size, device=all_xy.device)

            # Call function
            output = func(current_distance_matrix,delivery_node_demands,current_load)

            # Check output: should be torch.Tensor, shape (pomo_size, len(all_xy)), finite values
            if not isinstance(output, torch.Tensor):
                return False
            expected_shape = (pomo_size, all_xy.shape[0])
            if output.shape != expected_shape:
                return False
            if torch.any(torch.isnan(output)) or torch.any(torch.isinf(output)):
                return False
            return True

        except TimeoutError:
            return False
        except Exception as e:
            return False
        finally:
            signal.alarm(0)  # 取消超时