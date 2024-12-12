#!/user/ml4723/.conda/envs/neural_inventory_control/bin/python

import yaml
from trainer import *
import sys
from ray import train, tune # pip install "ray[tune]"
import matplotlib.pyplot as plt
from ray.tune import Stopper
import ray
optimal_test_losses_per_stores = {
    3: 5.61,
    5: 5.24,
    10: 5.71,
    20: 5.82,
    30: 5.55,
    50: 5.36,
}
results_dir = os.path.join(os.getcwd(), 'grid_search/results')
# n_store = 50
# maximum_context_size = 256
# context_search_count = 7
# change tune_rate too

n_store = 20
maximum_context_size = 2
context_search_count = 1

# this grid search is specifically for symmetry aware setup
search_or_visualize = sys.argv[1]
if search_or_visualize not in ["search", "visualize"]:
    raise ValueError("Invalid argument. Must be 'search' or 'visualize'.")

if search_or_visualize == "visualize":
    csv_files = [f for f in os.listdir(results_dir) if f.endswith('.csv')]
    store_results = pd.DataFrame(columns=['Store', 'Min Context Size'])
    for file in csv_files:
        store_number = file.split('_')[0]
        df = pd.read_csv(os.path.join(results_dir, file))
        successful_df = df[df['Success']]
        if not successful_df.empty:
            min_context_size = successful_df['Context Size'].min()
        else:
            min_context_size = maximum_context_size
        new_row = pd.DataFrame({'Store': [store_number], 'Min Context Size': [min_context_size]})
        store_results = pd.concat([store_results, new_row], ignore_index=True)

    store_results['Store'] = pd.to_numeric(store_results['Store'])
    store_results.sort_values('Store', inplace=True)
    plt.plot(store_results['Store'], store_results['Min Context Size'], marker='o', linestyle='-', color='blue')
    plt.xticks(store_results['Store'])  # Set x-axis ticks to only the existing store numbers
    plt.xlabel('# of Stores')
    plt.ylabel('Context Size')
    plt.title('Smallest Context Size Achieved 1% Test Gap VS # of Stores')
    plt.grid(True)
    plt.show()
    plt.savefig(os.path.join(results_dir, 'summary_results_plot.png'))
    exit()

hyperparams_name = "symmetry_aware_grid_search"
setting_name = sys.argv[2]

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
    problem_params['n_stores'] = tuning_configs['n_stores']
    nn_params['neurons_per_hidden_layer']['context'] = [tuning_configs['context_size'] for _ in nn_params['neurons_per_hidden_layer']['context']]
    nn_params['output_sizes']['context'] = tuning_configs['context_size']
    optimizer_params['learning_rate'] = tuning_configs['learning_rate']
    
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
            seeds=seeds,
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
            test_seeds,
            True
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
    
    with torch.no_grad():
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
    return {'test_loss': average_test_loss}

def is_success(test_loss):
    return test_loss <= optimal_test_losses_per_stores[n_store] * 1.005

class CustomStopper(Stopper):
    def __init__(self):
        self.should_stop = False

    def __call__(self, trial_id: str, result: dict) -> bool:
        if self.should_stop:
            return True

        if is_success(result["test_loss"]) == False:
            return False

        self.should_stop = True
        return True

    def stop_all(self) -> bool:
        return self.should_stop

ray.init(object_store_memory=4000000000)
minimum_context_size = 1
context_size = (minimum_context_size + maximum_context_size) // 2
results_df = pd.DataFrame(columns=['Context Size', 'Success'])
for _ in range(context_search_count):
    search_space = {
        "learning_rate": tune.grid_search([0.001]),
        "samples": tune.grid_search([0, 1, 2, 3, 4,5,6,7,8,9,10]),
        # "learning_rate": tune.grid_search([0.1, 0.01, 0.001, 0.0005]),
        # "samples": tune.grid_search([0, 1, 2, 3, 4,5,6,7,8,9,10]),
        "n_stores": n_store,
        "context_size": context_size,
    }
    stopper = CustomStopper()
    trainable_with_resources = tune.with_resources(run, {"cpu": 1, "gpu": 1})
    tuner = tune.Tuner(trainable_with_resources
    , param_space=search_space
    , run_config=train.RunConfig(stop=stopper, storage_path=os.path.join(os.getcwd(), f'ray_results_{n_store}')))
    results = tuner.fit()
    best_result = results.get_best_result("test_loss", "min")

    success = is_success(best_result.metrics['test_loss'])
    new_row = pd.DataFrame({'Context Size': [context_size], 'Success': [success]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    if success:
        maximum_context_size = context_size
        context_size = (minimum_context_size + context_size) // 2
    else:
        minimum_context_size = context_size
        context_size = (context_size + maximum_context_size) // 2
    print(f"context_size updated: {context_size}")

os.makedirs(results_dir, exist_ok=True)
results_path = os.path.join(results_dir, f'{n_store}_stores_context_search_results.csv')
results_df.to_csv(results_path, index=False)
ray.shutdown()