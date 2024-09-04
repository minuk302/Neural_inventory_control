#!/user/ml4723/.conda/envs/neural_inventory_control/bin/python

import yaml
from trainer import *
import sys
from ray import train, tune # pip install "ray[tune]"
import matplotlib.pyplot as plt
from ray.tune import Stopper
import ray
import json
import os
# for debugging
# os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# HDPO w/o context = symmetry_aware
# HDPO w/ context = symmetry_aware

# Check if command-line arguments for setting and hyperparameter filenames are provided (which corresponds to third and fourth parameters)

setting_name = sys.argv[1]
hyperparams_name = sys.argv[2]
n_stores = None
if len(sys.argv) >= 4:
    n_stores = int(sys.argv[3])
    
gpus_in_machine = torch.cuda.device_count()
if len(sys.argv) >= 5:
    gpus_to_use = [int(gpu) for gpu in sys.argv[4:]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus_to_use))
else:
    gpus_to_use = list(range(torch.cuda.device_count()))
total_cpus = os.cpu_count()
num_instances_per_gpu = 1
n_cpus_per_instance = max(16, total_cpus // (gpus_in_machine * num_instances_per_gpu) if gpus_in_machine > 0 else total_cpus)

load_model = False

print(f'Setting file name: {setting_name}')
print(f'Hyperparams file name: {hyperparams_name}\n')
print(f'Using GPUs: {gpus_to_use}')
print(f'Using CPUs per instance: {n_cpus_per_instance}')
config_setting_file = f'config_files/settings/{setting_name}.yml'
config_hyperparams_file = f'config_files/policies_and_hyperparams/{hyperparams_name}.yml'

with open(config_setting_file, 'r') as file:
    config_setting = yaml.safe_load(file)
with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)


def find_model_path_for(tuning_configs):
    paths = {
        3: "/user/ml4723/Prj/NIC/ray_results/perf/3/run_2024-07-19_23-43-56",
        5: "/user/ml4723/Prj/NIC/ray_results/perf/5/run_2024-07-19_23-45-43",
        10: "/user/ml4723/Prj/NIC/ray_results/perf/10/run_2024-07-19_23-53-23",
        20: "/user/ml4723/Prj/NIC/ray_results/perf/20/run_2024-07-19_23-53-28",
        30: "/user/ml4723/Prj/NIC/ray_results/perf/30/run_2024-07-19_23-51-47",
        50: "/user/ml4723/Prj/NIC/ray_results/perf/50/run_2024-07-19_23-53-27"
    }
    for num_stores, base_path in paths.items():
        if tuning_configs['n_stores'] != num_stores:
            continue
        for subfolder in os.listdir(base_path):
            subfolder_path = os.path.join(base_path, subfolder)
            params_path = os.path.join(subfolder_path, 'params.json')

            if not os.path.isfile(params_path):
                continue

            with open(params_path, 'r') as f:
                params = json.load(f)
                if all(tuning_configs[key] == params.get(key) for key in tuning_configs):
                    return os.path.join(subfolder_path, 'model.pt')
    return None

def run(tuning_configs):
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    import research_utils
    config_setting_overrided, config_hyperparams_overrided = research_utils.override_configs(tuning_configs, config_setting, config_hyperparams)
    recorder = research_utils.Recorder(config_setting_overrided, config_hyperparams_overrided)

    setting_keys = 'seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'echelon_params', 'sample_data_params'
    hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
    seeds, test_seeds, problem_params, params_by_dataset, observation_params, store_params, warehouse_params, echelon_params, sample_data_params = [
        config_setting_overrided[key] for key in setting_keys
        ]
    trainer_params, optimizer_params, nn_params = [config_hyperparams_overrided[key] for key in hyperparams_keys]
    observation_params = DefaultDict(lambda: None, observation_params)
    
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

    train_loader = DataLoader(train_dataset, batch_size=params_by_dataset['train']['batch_size'], shuffle=True, num_workers=n_cpus_per_instance, pin_memory=True, persistent_workers=True, prefetch_factor=8)
    dev_loader = DataLoader(dev_dataset, batch_size=params_by_dataset['dev']['batch_size'], shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=1)
    test_loader = DataLoader(test_dataset, batch_size=params_by_dataset['test']['batch_size'], shuffle=False, num_workers=2, pin_memory=True, prefetch_factor=1)
    data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

    neural_net_creator = NeuralNetworkCreator
    model = neural_net_creator().create_neural_network(scenario, nn_params, device=device)

    loss_function = PolicyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=optimizer_params['learning_rate'])

    simulator = Simulator(recorder, device=device)
    trainer = Trainer(device=device)

    if load_model == True:
        model, optimizer = trainer.load_model(model, optimizer, find_model_path_for(tuning_configs))

    trainer_params['base_dir'] = train.get_context().get_trial_dir()
    trainer_params['save_model_folders'] = []
    trainer_params['save_model_filename'] = "model"
    trainer.train(trainer_params['epochs'], loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params)

num_gpus = len(gpus_to_use)
num_instances = num_gpus * num_instances_per_gpu

gpus_per_instance = num_gpus / num_instances
ray.init(num_cpus = num_instances * n_cpus_per_instance, num_gpus = num_gpus, object_store_memory=4000000000, address='local')


if 'symmetry_aware_grid_search' == hyperparams_name:
    search_space = {
        'n_stores': n_stores,
        "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        'context': tune.grid_search([8, 16]),
        "overriding_networks": ["context"],
        "overriding_outputs": ["context"],
        "samples": tune.grid_search([1, 2, 3]),
    }
    save_path = 'ray_results/stable_bench/ctx'
elif 'vanilla_one_warehouse' == hyperparams_name:
    search_space = {
        'n_stores': n_stores,
        "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        "master": tune.grid_search([128, 512]),
        "overriding_networks": ["master"],
        "samples": tune.grid_search([1, 2, 3]),
    }
    save_path = 'ray_results/stable_bench/vanilla'
elif 'data_driven_net_real_data' == hyperparams_name:
    search_space = {
        "learning_rate": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),
        "master": tune.grid_search([32, 64, 128, 256]),
        "overriding_networks": ["master"],
        "samples": tune.grid_search([0, 1]),
    }
    save_path = 'ray_results/real/vanilla'
elif 'symmetry_aware_transshipment' == hyperparams_name:
        search_space = {
            'n_stores': n_stores,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            'context': tune.grid_search([0, 1, 256]),
            "overriding_networks": ["context"],
            "overriding_outputs": ["context"],
            "samples": tune.grid_search([1, 2, 3]),
        }
        save_path = 'ray_results/transshipment'
elif 'vanilla_transshipment' == hyperparams_name:
    search_space = {
        'n_stores': n_stores,
        "learning_rate": tune.grid_search([0.0001]),
        "samples": tune.grid_search([0, 1, 2]),
    }
    save_path = 'ray_results/transshipment/vanilla'
elif 'symmetry_aware_real_data' == hyperparams_name:
    search_space = {
        "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        "samples": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    save_path = 'ray_results/real/ctx'
elif 'symmetry_GNN_message_passing' == hyperparams_name:
    search_space = {
        "n_stores": n_stores,
        "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        "for_all_networks": tune.grid_search([1, 16, 64]),
        "overriding_networks": ["context", "store_embedding", "warehouse_embedding", "store_embedding_update"],
        "overriding_outputs": ["context", "store_embedding", "warehouse_embedding", "store_embedding_update"],
        "samples": tune.grid_search([0]),
    }
    save_path = 'ray_results/perf/GNN_message_passing'
elif 'GNN' in hyperparams_name:
    search_space = {
        "n_stores": n_stores,
        "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        "for_all_networks": tune.grid_search([1, 16, 64]),
        "overriding_networks": ["context", "store_embedding"],
        "overriding_outputs": ["context", "store_embedding"],
        "samples": tune.grid_search([0]),
    }
    if 'symmetry_GNN_real_data' == hyperparams_name:    
        save_path = 'ray_results/real/GNN'
    else:
        save_path = 'ray_results/perf/GNN'
elif 'transformed_nv_one_warehouse' == hyperparams_name:
    # for real data, n_stores = 16 is fixed!
    search_space = {
        "learning_rate": tune.grid_search([0.5, 0.03, 0.005]),
        "samples": tune.grid_search([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
    }
    save_path = 'ray_results/real/bench'
elif 'data_driven_net_real_data' == hyperparams_name:
    search_space = {
        "learning_rate": tune.grid_search([0.1, 0.01, 0.001, 0.0001]),
        "master": tune.grid_search([32, 64, 128, 256]),
        "overriding_networks": ["master"],
        "samples": tune.grid_search([0, 1]),
    }
    save_path = 'ray_results/real/vanilla'
trainable_with_resources = tune.with_resources(run, {"cpu": n_cpus_per_instance, "gpu": gpus_per_instance})
if n_stores != None:
    save_path += f'/{n_stores}'
    search_space['n_stores'] = n_stores

tuner = tune.Tuner(trainable_with_resources
, param_space=search_space
, run_config=train.RunConfig(storage_path=os.path.join(os.getcwd(), save_path)))

results = tuner.fit()
ray.shutdown()