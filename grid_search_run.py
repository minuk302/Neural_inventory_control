import yaml
import pandas as pd
from trainer import *
import sys
import os
from ray import train, tune # pip install "ray[tune]"
device = "cuda:0" if torch.cuda.is_available() else "cpu"

# this grid search is specifically for symmetry aware setup
hyperparams_name = "symmetry_aware"
if len(sys.argv) == 2:
    setting_name = sys.argv[1]
else:
    print(f'Number of parameters provided including script name: {len(sys.argv)}')
    print(f'Number of parameters should be 1.')
    assert False


print(f'Default setting file name: {setting_name}')
print(f'Default hyperparams file name: {hyperparams_name}\n')
config_setting_file = f'config_files/settings/{setting_name}.yml'
config_hyperparams_file = f'config_files/policies_and_hyperparams/{hyperparams_name}.yml'

with open(config_setting_file, 'r') as file:
    config_setting = yaml.safe_load(file)
with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)

def run(tuning_configs):
    setting_keys = 'seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'echelon_params', 'sample_data_params'
    hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
    seeds, test_seeds, problem_params, params_by_dataset, observation_params, store_params, warehouse_params, echelon_params, sample_data_params = [
        config_setting[key] for key in setting_keys
        ]
    trainer_params, optimizer_params, nn_params = [config_hyperparams[key] for key in hyperparams_keys]
    observation_params = DefaultDict(lambda: None, observation_params)

    # apply tuning configs
    problem_params['n_stores'] = tuning_configs['n_stores']
    nn_params['neurons_per_hidden_layer']['context'] = [tuning_configs['context_size'] for _ in nn_params['neurons_per_hidden_layer']['context']]
    nn_params['output_sizes']['context'] = tuning_configs['context_size']
    optimizer_params['learning_rate'] = tuning_configs['learning_rate']
    seeds['demand'] = tuning_configs['demand_seed']
    
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

    trainer_params['base_dir'] = 'saved_models'
    trainer_params['save_model_folders'] = [trainer.get_year_month_day(), nn_params['name']]
    # TODO: If want to save model, modify the file name to be based on the values of tuning_configs.
    trainer_params['save_model_filename'] = trainer.get_time_stamp()

    trainer.train(trainer_params['epochs'], loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params)
    average_test_loss, average_test_loss_to_report = trainer.test(
        loss_function, 
        simulator, 
        model, 
        data_loaders, 
        optimizer, 
        problem_params, 
        observation_params, 
        params_by_dataset, 
        discrete_allocation=store_params['demand']['distribution'] == 'poisson'
        )

    return {"test_loss": average_test_loss_to_report}

search_space = {
    "n_stores": tune.grid_search([3, 5, 10, 20, 30, 50]),
    "context_size": tune.grid_search([1, 2, 4, 8, 16, 32, 64, 128]),
    "learning_rate": tune.grid_search([0.01, 0.001]),
    "demand_seed": tune.grid_search([57, 58, 59]),
}

tuner = tune.Tuner(run, param_space=search_space)

# Code for restoring/resuming grid search.
# https://docs.ray.io/en/latest/tune/api/doc/ray.tune.Tuner.restore.html#ray.tune.Tuner.restore
# change restart_errored accordingly.
# tuner = tune.Tuner.restore(os.getcwd() + "/ray_results/run_2024-06-27_07-48-18", run, restart_errored=True)

results = tuner.fit()