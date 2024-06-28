import yaml
from trainer import *
import sys
from ray import train, tune # pip install "ray[tune]"
from ray.tune import Stopper
import seaborn as sns
import json
import time
from matplotlib.ticker import FuncFormatter
device = "cuda:0" if torch.cuda.is_available() else "cpu"

optimal_test_losses_per_stores = {
    3: 5.61,
    5: 5.24,
    10: 5.71,
    20: 5.82,
    30: 5.55,
    50: 5.36,
}

# this grid search is specifically for symmetry aware setup
search_or_visualize = sys.argv[1]
if search_or_visualize not in ["search", "visualize"]:
    raise ValueError("Invalid argument. Must be 'search' or 'visualize'.")

if search_or_visualize == "visualize":
    ray_results_folder_name = sys.argv[2]
    base_path = os.getcwd() + "/ray_results/" + ray_results_folder_name

    data = []
    for folder_name in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder_name)
        if os.path.isdir(folder_path) == False:
            continue
        params_path = os.path.join(folder_path, 'params.json')
        result_path = os.path.join(folder_path, 'result.json')
        
        with open(params_path, 'r') as file:
            params = json.load(file)
        with open(result_path, 'r') as file:
            results = json.load(file)
        
        data.append({
            'n_stores': params['n_stores'],
            'context_size': params['context_size'],
            'learning_rate': params['learning_rate'],
            'samples': params['samples'],
            'test_loss': results['test_loss']
        })
    df = pd.DataFrame(data)
    df = df.sort_values(by='test_loss')
    df = df.drop_duplicates(subset=['n_stores', 'context_size', 'learning_rate'], keep='first')
    df.drop(columns=['samples'], inplace=True)

    figure_result_directory_path = os.path.join(os.getcwd(), 'grid_search/results')
    os.makedirs(figure_result_directory_path, exist_ok=True)
    for n_store in df['n_stores'].unique():
        df_plot = df[df['n_stores'] == n_store].copy()
        optimal_test_loss = optimal_test_losses_per_stores[n_store]
        df_plot['test_loss_gap_percentage'] = ((df_plot['test_loss'] - optimal_test_loss) / optimal_test_loss)
        df_pivot = df_plot.pivot(index='learning_rate', columns='context_size', values='test_loss_gap_percentage')
        
        def percentage_formatter(x, pos):
            return '{:.2%}'.format(x)

        # Create heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(df_pivot, annot=True, cmap="coolwarm", fmt=".2%", vmin=0, cbar_kws={'format': FuncFormatter(percentage_formatter)})
        plt.title(f'Heatmap of Test Loss Gap % for {n_store} Stores')
        plt.xlabel('Context Net/Dimension')
        plt.ylabel('Learning Rate')
        
        # Save heatmap
        plt.savefig(f'{figure_result_directory_path}/heatmap_n_stores_{n_store}.png')
        plt.close()
    exit()

hyperparams_name = "symmetry_aware"
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

    training_losses = trainer.train(trainer_params['epochs'], loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params)
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
    
    training_losses['test_loss'] = average_test_loss_to_report
    return training_losses

n_store = 3
def is_success(test_loss):
    return test_loss <= optimal_test_losses_per_stores[n_store] * 1.01

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

context_search_count = 7
minimum_context_size = 1
maximum_context_size = 256
context_size = 128
for _ in range(context_search_count):
    search_space = {
        "learning_rate": tune.grid_search([0.01, 0.001]),
        "samples": tune.grid_search([0, 1, 2]),
        "n_stores": n_store,
        "context_size": context_size,
    }
    stopper = CustomStopper()
    tuner = tune.Tuner(run, param_space=search_space, run_config=train.RunConfig(stop=stopper))
    results = tuner.fit()
    best_result = results.get_best_result("test_loss", "min")

    if is_success(best_result.metrics['test_loss']):
        maximum_context_size = context_size
        context_size = (minimum_context_size + context_size) // 2
    else:
        minimum_context_size = context_size
        context_size = (context_size + maximum_context_size) // 2
    print(f"context_size updated: {context_size}")