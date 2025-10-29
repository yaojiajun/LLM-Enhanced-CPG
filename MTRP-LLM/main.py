import hydra
import logging 
import os
from pathlib import Path
import subprocess
from utils.utils import init_client


ROOT_DIR = os.getcwd()
logging.basicConfig(level=logging.INFO)

@hydra.main(version_base=None, config_path="cfg", config_name="config")
def main(cfg):
    workspace_dir = Path.cwd()
    # Set logging level
    logging.info(f"Workspace: {workspace_dir}")
    logging.info(f"Project Root: {ROOT_DIR}")
    logging.info(f"Using LLM: {cfg.get('model', cfg.llm_client.model)}")
    logging.info(f"Using Algorithm: {cfg.algorithm}")

    client = init_client(cfg)
    if cfg.algorithm == "hercules":
        from hercules import Hercules as LHH
    # elif cfg.algorithm == "hercules-p":
    #     from herculesp import Herculesp as LHH
    else:
        raise NotImplementedError

    # Create code directory
    code_dir = os.path.join(ROOT_DIR, 'code')
    os.makedirs(code_dir, exist_ok=True)

    # Run until 10 improvements are found
    max_improvements = 10
    logging.info(f"Running until {max_improvements} improvements are found")

    best_score_overall = float('inf')  # Assuming lower score is better
    improvement_count = 0
    run = 0

    while True:
        # Main algorithm
        lhh = LHH(cfg, ROOT_DIR, client)
        best_code, best_path = lhh.evolve()
        logging.info(f"Run {run}: Best Code: {best_code}")
        logging.info(f"Run {run}: Best Code Path: {best_path}")
        
        # Write the code to file
        code_file = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/gpt.py"
        with open(code_file, 'w') as file:
            file.write(best_code + '\n')
        
        # Run validation
        test_script = f"{ROOT_DIR}/problems/{cfg.problem.problem_name}/eval.py"
        logging.info(f"Run {run}: Running validation script...: {test_script}")
        result = subprocess.run(["python", test_script, "-1", ROOT_DIR, "val"], capture_output=True, text=True)
        output = result.stdout
        lines = output.splitlines()
        
        # Extract score assuming the last line is the numerical evaluation value
        score = float('inf')
        if lines:
            last_line = lines[-1].strip()
            try:
                # Parse the number after the colon
                score_str = last_line.split(':')[-1].strip()
                score = float(score_str)
            except ValueError:
                logging.error(f"Run {run}: Could not parse score from last line: {last_line}")
        
        logging.info(f"Run {run}: Evaluation Score: {score}")
        
        # Save the improved code with incremental numbering in ./code
        best_file = os.path.join(code_dir, f'best_code_{score}.py')
        with open(best_file, 'w') as f:
            f.write(best_code)
        logging.info(f"Saved improved code to {best_file}")
        
        # Check if this is an improvement
        if score < best_score_overall:
            best_score_overall = score
            best_code_overall = best_code
            best_path_overall = best_path
            best_output_lines = lines
            improvement_count += 1
            logging.info(f"Improvement {improvement_count}: New best score {best_score_overall}")

            # Modify the function name for seed_func.txt
            modified_code = best_code_overall.replace('heuristics_v2', 'heuristics_v1')

            # Replace the content in ./prompts/mt_pomo_tw/seed_func.txt with modified code
            seed_file = os.path.join(ROOT_DIR, 'prompts', 'mt_routefinder_unified', 'seed_func.txt')
            with open(seed_file, 'w') as f:
                f.write(modified_code)
            logging.info(f"Updated seed_func.txt with modified improved code")

            if improvement_count >= max_improvements:
                logging.info(f"Reached {max_improvements} improvements. Stopping.")
                break

        run += 1

    # Log the overall best
    logging.info(f"Best Score Overall: {best_score_overall}")
    logging.info(f"Best Code Overall: {best_code_overall}")
    logging.info(f"Best Code Path Overall: {best_path_overall}")
    
    # Print the full validation results for the best run
    if best_output_lines:
        logging.info("Best Run Validation Results:")
        for line in best_output_lines:
            logging.info(line.strip())

if __name__ == "__main__":
    main()