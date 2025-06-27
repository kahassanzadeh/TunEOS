import ast
import yaml
import pprint
import os


def extract_optuna_args(optuna_str):
    """
    Given a string like "optuna(0.001, 0.01, [0.1, 0.2])",
    parse it with AST and return a list of literal values.
    """
    fake_src = f"_param = {optuna_str.strip()}"
    node = ast.parse(fake_src).body[0]
    call = node.value

    values = []
    for arg in call.args:
        if isinstance(arg, ast.Constant):
            values.append(arg.value)
        elif isinstance(arg, (ast.List, ast.Tuple)):
            for elt in arg.elts:
                if isinstance(elt, ast.Constant):
                    values.append(elt.value)
    return values


def find_optuna_params(cfg, prefix=""):
    """
    Recursively traverse cfg (dicts/lists) and find any string values starting with optuna(...).
    Returns a dict mapping each config path (dot/list-index notation) to its candidate values.
    """
    tuned = {}

    if isinstance(cfg, dict):
        for key, val in cfg.items():
            path = f"{prefix}.{key}" if prefix else key
            tuned.update(find_optuna_params(val, path))
    elif isinstance(cfg, list):
        for idx, val in enumerate(cfg):
            path = f"{prefix}[{idx}]"
            tuned.update(find_optuna_params(val, path))
    elif isinstance(cfg, str) and cfg.strip().lower().startswith("optuna("):
        try:
            values = extract_optuna_args(cfg)
            tuned[prefix] = values
        except Exception as e:
            pass

    return tuned


def file_handler(path):
    dict_of_files = dict()
    matches = [fname for fname in os.listdir(path)
               if fname.startswith('opt')]
    print(f'Detected Files')
    for fname in matches:
        print(os.path.join(path, fname))
        dict_of_files[os.path.join(path, fname)] = dict()

    for file in dict_of_files.keys():
        model_name = file.split('_')[-1]
        print(f'-------------------------{model_name}-------------------------')
        print(f'Detected tunable parameters and their candidate values:')
        with open(f"{file}/config.yaml") as f:
            cfg = yaml.safe_load(f)

        params = find_optuna_params(cfg)
        pprint.pprint(params)
        dict_of_files[file] = params

    return dict_of_files

if __name__ == "__main__":
    test = file_handler('.')
    print(test)
