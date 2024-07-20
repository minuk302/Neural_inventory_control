#!/user/ml4723/.conda/envs/neural_inventory_control/bin/python

import yaml
from trainer import *
import sys
from ray import train, tune # pip install "ray[tune]"
import matplotlib.pyplot as plt
from ray.tune import Stopper
import ray

# HDPO w/o context = symmetry_aware
# HDPO w/ context = symmetry_aware

# Check if command-line arguments for setting and hyperparameter filenames are provided (which corresponds to third and fourth parameters)
if len(sys.argv) == 3:
    setting_name = sys.argv[1]
    hyperparams_name = sys.argv[2]

n_stores = 3

print(f'Setting file name: {setting_name}')
print(f'Hyperparams file name: {hyperparams_name}\n')
config_setting_file = f'config_files/settings/{setting_name}.yml'
config_hyperparams_file = f'config_files/policies_and_hyperparams/{hyperparams_name}.yml'

with open(config_setting_file, 'r') as file:
    config_setting = yaml.safe_load(file)
with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)

def run(tuning_configs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    setting_keys = 'seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'echelon_params', 'sample_data_params'
    hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
    seeds, test_seeds, problem_params, params_by_dataset, observation_params, store_params, warehouse_params, echelon_params, sample_data_params = [
        config_setting[key] for key in setting_keys
        ]
    trainer_params, optimizer_params, nn_params = [config_hyperparams[key] for key in hyperparams_keys]
    observation_params = DefaultDict(lambda: None, observation_params)

    # apply tuning configs
    if "master_neurons" in tuning_configs:
        nn_params['neurons_per_hidden_layer']['master'] = [tuning_configs['master'] for _ in nn_params['neurons_per_hidden_layer']['master']]

    if 'n_stores' in tuning_configs:
        problem_params['n_stores'] = tuning_configs['n_stores']
    
    if 'learning_rate' in tuning_configs:
        optimizer_params['learning_rate'] = tuning_configs['learning_rate']

    nets = ['context', 'context_store']
    for net in nets:
        if net in tuning_configs:
            nn_params['neurons_per_hidden_layer'][net] = [tuning_configs[net] for _ in nn_params['neurons_per_hidden_layer'][net]]
            nn_params['output_sizes'][net] = tuning_configs[net]
    
    dataset_creator = DatasetCreator()
    if sample_data_params['split_by_period']:
        scenario = Scenario(
            periods=None,  # period info for each dataset is given in sample_data_params
            problem_params=problem_params, 
            store_params=store_params, 
            warehouse_params=warehouse_params, 
            echelon_params=echelon_params, 
            num_samples=params_by_dataset['train']['n_samples'],  # in this case, num_samples=number of products, which has to be the same across all datasets
            observation_params=observation_params, 
            seeds=seeds
            )
        train_dataset, dev_dataset, test_dataset = dataset_creator.create_datasets(
            scenario, 
            split=True, 
            by_period=True, 
            periods_for_split=[sample_data_params[k] for  k in ['train_periods', 'dev_periods', 'test_periods']],)
    else:
        scenario = Scenario(
            periods=params_by_dataset['train']['periods'], 
            problem_params=problem_params, 
            store_params=store_params, 
            warehouse_params=warehouse_params, 
            echelon_params=echelon_params, 
            num_samples=params_by_dataset['train']['n_samples'] + params_by_dataset['dev']['n_samples'], 
            observation_params=observation_params, 
            seeds=seeds
            )

        train_dataset, dev_dataset = dataset_creator.create_datasets(scenario, split=True, by_sample_indexes=True, sample_index_for_split=params_by_dataset['dev']['n_samples'])
        scenario = Scenario(
            params_by_dataset['test']['periods'], 
            problem_params, 
            store_params, 
            warehouse_params, 
            echelon_params, 
            params_by_dataset['test']['n_samples'], 
            observation_params, 
            test_seeds
            )
        test_dataset = dataset_creator.create_datasets(scenario, split=False)

    train_loader = DataLoader(train_dataset, batch_size=params_by_dataset['train']['batch_size'], shuffle=True)
    dev_loader = DataLoader(dev_dataset, batch_size=params_by_dataset['dev']['batch_size'], shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=params_by_dataset['test']['batch_size'], shuffle=False)
    data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

    neural_net_creator = NeuralNetworkCreator
    model = neural_net_creator().create_neural_network(scenario, nn_params, device=device)

    loss_function = PolicyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'])

    simulator = Simulator(device=device)
    trainer = Trainer(device=device)

    trainer_params['base_dir'] = train.get_context().get_trial_dir()
    trainer_params['save_model_folders'] = []
    trainer_params['save_model_filename'] = "model"
    trainer.train(trainer_params['epochs'], loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params)

ray.init(object_store_memory=4000000000)

if 'symmetry_aware' in hyperparams_name:
    search_space = {
        'n_stores': n_stores,
        "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        'context': tune.grid_search([1, 256]),
        "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]),
    }
elif 'GNN' in hyperparams_name:
    search_space = {
        "learning_rate": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),
        "context": tune.grid_search([16, 32, 64, 128, 256]),
        "context_store": tune.grid_search([16, 32, 64, 128, 256]),
        "samples": tune.grid_search([0, 1]),
    }
else:
    search_space = {
        "learning_rate": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),
        "master_neurons": tune.grid_search([32, 64, 128, 256]),
        "samples": tune.grid_search([0, 1]),
    }
trainable_with_resources = tune.with_resources(run, {"cpu": 1, "gpu": 1})
tuner = tune.Tuner(trainable_with_resources
, param_space=search_space
, run_config=train.RunConfig(storage_path=os.path.join(os.getcwd(), f'ray_results/perf/{n_stores}')))
results = tuner.fit()
ray.shutdown()