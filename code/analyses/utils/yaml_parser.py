import yaml
import itertools

def grid_from_yaml(yaml_file):
    with open(yaml_file) as f:
        all_params = yaml.load(f)
    keys = all_params.keys()
    values = (all_params[key] for key in keys)
    hyperparams_grid = [dict(zip(keys, combination)) for combination in itertools.product(*values)]
    return hyperparams_grid

def hierarchize_one(hp_dict):
    hierarchized_param_dict = {}
    for key, value in hp_dict.items():
        branches = key.split('.')
        h = hierarchized_param_dict
        for node in branches[:-1]:
            if node not in h.keys():
                h[node] = {}
            h = h[node]
        h[branches[-1]] = value
    return hierarchized_param_dict

def hierarchize(hyperparams_grid):
    return [hierarchize_one(hp_dict) for hp_dict in hyperparams_grid]

if __name__ == '__main__':
    hyperparams_grid = grid_from_yaml('example.yaml')
    print(hyperparams_grid)
    print('')
    print(hierarchize(hyperparams_grid))
