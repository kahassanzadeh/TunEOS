import os
import re
import subprocess
import tempfile
import optuna
import logging
from omegaconf import OmegaConf
import wandb
from optuna.exceptions import TrialPruned
from functools import partial
from pathlib import Path
import pandas
from optuna.samplers import GridSampler
import time

import preprocess_conf


def fetch_test_metric_from_wandb(run_id: str, entity: str, project: str,
                                 key: str) -> float:
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history(keys=[key], pandas=True)
    if len(history) == 0:
        raise ValueError(f"No data for key '{key}' in run {run_id}")
    test_val = history.iloc[-1][key]
    return float(test_val)


def update_sbatch_script(script_path: str, job_name: str, base_path:str, logs_base: str = "logs") -> None:
    with open(script_path, 'r') as f:
        content = f.read()

    config_path = f""" config_path="{base_path[2:]}" """
    temp_path = f""" temp_path="{job_name}.yaml" """

    content = re.sub(
        r'(^#SBATCH\s+--job-name=).*',
        rf"\1{job_name}",
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'(^#SBATCH\s+--output=).*',
        rf"\1{base_path}/{logs_base}/{job_name}/%x-%j.out",
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'(^#SBATCH\s+--error=).*',
        rf"\1{base_path}/{logs_base}/{job_name}/%x-%j.err",
        content,
        flags=re.MULTILINE
    )

    content = re.sub(
        r'^config_path.*',
        config_path,
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^temp_path.*',
        temp_path,
        content,
        flags=re.MULTILINE
    )

    with open(f'{base_path}/{job_name}.sh', 'w') as f:
        f.write(content)

    log_dir = os.path.join(base_path, logs_base, job_name)
    os.makedirs(log_dir, exist_ok=True)


def objective(trial: optuna.Trial, path_conf: str, model: str, dict_of_params: dict):
    cfg = OmegaConf.load(path_conf)
    name = f'{model}_{trial.number}_'
    for k, v in dict_of_params.items():
        # print(f"Suggesting value for {k} from {v}")
        value = trial.suggest_categorical(f'{k}', list(v))
        OmegaConf.update(cfg, k, value)
        name += f'{k}_{value}'
    
    if model == 'Nequip':
        cfg.trainer.logger.name = name
    elif model == 'Mace':
        cfg.name = name
        cfg.model_dir= name
        cfg.log_dir= name
        cfg.checkpoints_dir= name
        cfg.results_dir= name
        cfg.wandb_name = name

    base_path = f'./opt_{model}/{name}'
    os.makedirs(base_path, exist_ok=True)

    with open(f'{base_path}/out.txt', 'w') as f:
        f.write('')

    update_sbatch_script(f'./opt_{model}/job.sh', name, base_path)

    with open(f"{base_path}/{name}.yaml", 'w') as tmp:
        OmegaConf.save(cfg, tmp)

    flag = False
    entity = ''
    project = ''
    run_id = ''

    try:
        process = subprocess.Popen(["sbatch", f"{base_path}/{name}.sh"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        
        out, _ = process.communicate()
        m = re.search(r"Submitted batch job (\d+)", out)
        if not m:
            raise RuntimeError("sbatch failed to submit")
        job_id = m.group(1)
        print(f"Submitted job {job_id}")
        
        log_dir = os.path.join(base_path, "logs", name)
        log_file = os.path.join(log_dir, f"{name}-{job_id}.out")
        log_err_file = os.path.join(log_dir, f"{name}-{job_id}.err")
        
        while not os.path.exists(log_file):
            time.sleep(1.0)
        
        
        flag_time = False
        timeout_secs = 10 * 3600   
        start = time.time()  
        while True:
            if flag == False:
                with open(log_err_file, "r") as f:
                    m = re.search(r"https://wandb.ai/([^/]+)/([^/]+)/runs/([^/\s]+)", f.read())
                    if flag == False and m:
                        entity = m.group(1)
                        project = m.group(2)
                        run_id = m.group(3)
                        print("************************************************************")
                        print(f"Detected WandB run: entity={entity}, project={project}, run_id={run_id}")
                        print("************************************************************")
                        flag = True
            
            sacct = subprocess.run(
            ["sacct", "--jobs", job_id, "--format=State", "--noheader"],
            capture_output=True, text=True
            )
            state = sacct.stdout.strip().split()[0]
            # print(state)
            if state in ("COMPLETED", "FAILED", "CANCELLED"):
                break
            
            time.sleep(2)
            # print(time.time() - start)
            if time.time() - start > timeout_secs:
                print(f"\nTimeout of {timeout_secs}s reached, killing process.")
                scancel = subprocess.run(
                ["scancel", job_id],
                capture_output=True, text=True
                )
                flag_time = True
                break
                
        # if process.returncode != 0:
        #     logging.error("Training command failed with return code %d", process.returncode)
        #     raise RuntimeError("Training command failed")
        if flag_time:
            raise RuntimeError("Training command timed out and was killed")
        
        test_metric = fetch_test_metric_from_wandb(run_id, entity, project, 'test0_epoch/weighted_sum')
        print(f"Fetched test metric from WandB: {test_metric}")
        return test_metric
    except Exception as e:
        print(f"An error occurred: {e}")
        raise TrialPruned()


def tuning(model: str, study_name: str):
    study = optuna.create_study(
        study_name=study_name,
        storage="postgresql+psycopg2://postgres:kamyar_1378@nequip.c70eu6gaclqm.eu-north-1.rds.amazonaws.com:5432/test",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=10)
    )


    file_name = f'./opt_{model}'
    dict_of_files = preprocess_conf.file_handler('.')
    dict_of_params = dict_of_files[file_name]
    # print(dict_of_params)

    bound_objective = partial(objective, path_conf=f'{file_name}/config.yaml', model=model,
                              dict_of_params=dict_of_params)

    study.optimize(bound_objective, n_trials=100)


if __name__ == "__main__":
    tuning('Mace', study_name='final_mace_test')

