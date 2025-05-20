#!/user/ml4723/.conda/envs/neural_inventory_control/bin/python

import yaml
from trainer import *
import sys
from ray import train, tune # pip install "ray[tune]"
import matplotlib.pyplot as plt
from ray.tune import Stopper
import ray
import os
from main_run import MainRun

testset_name = sys.argv[1]
hyperparams_name = sys.argv[2]
num_instances_per_gpu = 1
if len(sys.argv) >= 4:
    num_instances_per_gpu = int(sys.argv[3])

n_stores = None
if len(sys.argv) >= 5:
    n_stores = int(sys.argv[4])
    if n_stores == -1:
        n_stores = None
    
gpus_in_machine = torch.cuda.device_count()
if len(sys.argv) >= 6:
    gpus_to_use = [int(gpu) for gpu in sys.argv[5:]]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpus_to_use))
else:
    gpus_to_use = list(range(torch.cuda.device_count()))
total_cpus = os.cpu_count()
n_cpus_per_instance = max(1, min(16, total_cpus // (gpus_in_machine * num_instances_per_gpu) if gpus_in_machine > 0 else total_cpus))

load_model = False

print(f'Hyperparams file name: {hyperparams_name}\n')
print(f'Using GPUs: {gpus_to_use}')
print(f'Using CPUs per instance: {n_cpus_per_instance}')

# Load all settings from config files
config_settings_dir = 'config_files/settings'
config_settings = {}
for root, dirs, files in os.walk(config_settings_dir):
    for filename in files:
        if filename.endswith('.yml'):
            setting_name = os.path.relpath(os.path.join(root, filename[:-4]), config_settings_dir)
            with open(os.path.join(root, filename), 'r') as file:
                config_settings[setting_name] = yaml.safe_load(file)

config_hyperparams_file = f'config_files/policies_and_hyperparams/{hyperparams_name}.yml'
with open(config_hyperparams_file, 'r') as file:
    config_hyperparams = yaml.safe_load(file)

def run(tuning_configs):
    # for debugging
    # os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    # os.environ["TORCH_USE_CUDA_DSA"] = "1"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tuning_configs['n_cpus_per_instance'] = n_cpus_per_instance
    tuning_configs['base_dir_for_ray'] = train.get_context().get_trial_dir()
    main_run = MainRun("train", tuning_configs['config'], config_settings[tuning_configs['config']], config_hyperparams, tuning_configs)
    main_run.run()

num_gpus = len(gpus_to_use)
num_instances = num_gpus * num_instances_per_gpu

gpus_per_instance = num_gpus / num_instances
ray.init(num_cpus = num_instances * n_cpus_per_instance, num_gpus = num_gpus, object_store_memory=4000000000, address='local')

save_path = f'ray_results/{testset_name}/{hyperparams_name}'

if "finals_one_store_sample_efficiency" == testset_name:
    config = "one_store_lost"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([5]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "store_lead_time": tune.grid_search([3, 4]),
        "store_underage_cost": tune.grid_search([9, 19]),

        "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),

        'train_dev_sample_and_batch_size': tune.grid_search([16, 32, 64, 128, 256, 512, 1024]),

        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
    }
    if 'vanilla_one_store' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }

if "finals_serial" == testset_name:
    configs = ["serial_system_7", "serial_system_3", "serial_system_4", "serial_system_5", "serial_system_6"]
    common_setups = {
        "config": tune.grid_search(configs),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "store_lead_time": tune.grid_search([4]),
        "store_underage_cost": tune.grid_search([9]),
        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),
        
        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192, 1024, 128]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'vanilla_serial' == hyperparams_name:
        search_space = { **common_setups,
            "master_echelon": tune.grid_search([32, 64, 128]),
            "overriding_networks": ["master_echelon"],
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if 'vanilla_serial_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_echelon_selfloop": tune.grid_search([32, 64, 128]),
            "overriding_networks": ["master_echelon_selfloop"],
        }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if 'GNN_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
        search_space['train_dev_sample_and_batch_size'] = tune.grid_search([128])
        search_space['repeats'] = tune.grid_search([3])
        search_space['learning_rate'] = tune.grid_search([0.001, 0.0001])
    if 'GNN_bottleneck_small' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if 'echelon_stock_hard' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.5, 0.1, 0.03]),
        }
        search_space['train_dev_sample_and_batch_size'] = tune.grid_search([8192])
        search_space['repeats'] = tune.grid_search([4, 5, 6, 7, 8, 9, 10])

if "serial_paper_comparison" == testset_name:
    config = "serial_system_4"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "store_lead_time": tune.grid_search([1, 2, 3, 4]),
        "store_underage_cost": tune.grid_search([4, 9, 19, 39]),
        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),
        
        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([32768]),
        "train_batch_size": tune.grid_search([8192]),

        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
    }
    if 'vanilla_serial' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01]),
        }
    if 'echelon_stock_hard' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.5, 0.1, 0.03]),
        }
        search_space["repeats"] = tune.grid_search([1])

if "serial_paper_comparison_8K" == testset_name:
    config = "serial_system_4"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "store_lead_time": tune.grid_search([1, 2, 3, 4]),
        "store_underage_cost": tune.grid_search([4, 9, 19, 39]),
        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),
        
        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'vanilla_serial' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01]),
        }
    if 'echelon_stock_hard' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.5, 0.1, 0.03]),
        }
        search_space["repeats"] = tune.grid_search([1])

if "finals_transshipment" == testset_name:
    config = "transshipment_backlogged"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "n_stores": tune.grid_search([50, 30, 20, 10, 5, 3]),
        "store_underage_cost": tune.grid_search([9]),
        "store_lead_time": tune.grid_search([4]),

        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192, 1024, 128]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'GNN_bottleneck_transshipment' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        # search_space["n_stores"] = tune.grid_search([50, 30, 20, 10, 5, 3])
        search_space["repeats"] = tune.grid_search([1])
    if 'GNN_transshipment' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space["repeats"] = tune.grid_search([2, 3])
    if 'vanilla_transshipment' == hyperparams_name:  
        search_space = {**common_setups,
            "master": tune.grid_search([512, 256, 128]),
            'overriding_networks': ['master'],
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if 'vanilla_transshipment_selfloop' == hyperparams_name:
        search_space = {**common_setups,
            "master": tune.grid_search([512, 256, 128]),
            'overriding_networks': ['master'],
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }

if "finals_one_warehouse_n_stores_debug" == testset_name:
    config = "one_warehouse_lost_demand"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "n_stores": tune.grid_search([3]),
        "store_underage_cost": tune.grid_search([9]),
        "store_lead_time": tune.grid_search([[2, 6]]),

        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),

        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1]),
        'train_dev_sample_and_batch_size': tune.grid_search([1024, 128]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }

if "finals_one_warehouse_n_stores" == testset_name:
    config = "one_warehouse_lost_demand"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "n_stores": tune.grid_search([50, 30, 20, 10, 5, 3]),
        "store_underage_cost": tune.grid_search([9]),
        "store_lead_time": tune.grid_search([[2, 6]]),

        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),

        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192, 1024, 128]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
    if 'GNN_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }
        search_space["repeats"] = tune.grid_search([2, 3])
    if 'GNN_bottleneck' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space["repeats"] = tune.grid_search([2, 3])
    if 'vanilla_one_warehouse' == hyperparams_name:  
        search_space = {**common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }
    if 'vanilla_one_warehouse_selfloop' == hyperparams_name:
        search_space = {**common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_selfloop": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master_selfloop"],
        }

if "finals_real_n_warehouses_n_stores" == testset_name:
    configs = ["n_warehouse_21_2_real_lost_demand", "n_warehouse_21_3_real_lost_demand", "n_warehouse_21_4_real_lost_demand",  "n_warehouse_21_5_real_lost_demand"]
    common_setups = {
        "config": tune.grid_search(configs),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([288]),
        "train_batch_size": tune.grid_search([72]),

        "test_n_samples": tune.grid_search([288]),
        "test_batch_size": tune.grid_search([288]),
    }
    if 'GNN_real' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
    if 'GNN_real_best' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
        search_space['config'] = tune.grid_search([config + "_best" for config in configs])
    if 'GNN_real_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
    if 'GNN_real_bottleneck' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
    if 'GNN_real_fastest' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space['train_dev_sample_and_batch_size'] = tune.grid_search([288])
    if 'GNN_real_cheapest' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space['train_dev_sample_and_batch_size'] = tune.grid_search([288])
    if "data_driven_net_n_warehouses_real" == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_n_warehouses": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master_n_warehouses"],
        }
    if "just_in_time_real" == hyperparams_name:
        search_space = { **common_setups,
            "all_edges_have_lead_time_one": tune.grid_search([True]),
        }
        search_space["repeats"] = tune.grid_search([1])

if "finals_real_one_warehouse_n_stores" == testset_name:
    config = "one_warehouse_21_real_lost_demand"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "n_stores": tune.grid_search([3, 5, 10, 15, 21]),

        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([64]),
        "train_batch_size": tune.grid_search([72]),

        "test_n_samples": tune.grid_search([64]),
        "test_batch_size": tune.grid_search([64]),
    }
    if 'GNN_real' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
    if 'GNN_real_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
    if 'GNN_real_bottleneck' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
    if "data_driven_net_real" == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }
    if "just_in_time_real" == hyperparams_name:
        search_space = { **common_setups,
        }
        search_space["repeats"] = tune.grid_search([1])
        search_space["train_dev_sample_and_batch_size"] = tune.grid_search([64])

if "finals_n_warehouses_n_stores" == testset_name:
    configs = ["n_warehouse_50_6_lost_demand", "n_warehouse_40_5_lost_demand", "n_warehouse_30_4_lost_demand", "n_warehouse_20_3_lost_demand", "n_warehouse_10_2_lost_demand"]
    common_setups = {
        "config": tune.grid_search(configs),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "store_underage_cost": tune.grid_search([9]),

        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),

        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192, 1024, 128]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
        search_space["repeats"] = tune.grid_search([2, 3])
        # search_space["config"] = tune.grid_search(["n_warehouse_50_6_lost_demand", "n_warehouse_40_5_lost_demand"])
        search_space["config"] = tune.grid_search(["n_warehouse_30_4_lost_demand", "n_warehouse_20_3_lost_demand", "n_warehouse_10_2_lost_demand"])
        
    if 'GNN_bottleneck' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
        search_space["config"] = tune.grid_search(["n_warehouse_50_6_lost_demand", "n_warehouse_40_5_lost_demand"])
        search_space["repeats"] = tune.grid_search([1])
    if 'GNN_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
        search_space["config"] = tune.grid_search(["n_warehouse_50_6_lost_demand", "n_warehouse_40_5_lost_demand"])
        # search_space["config"] = tune.grid_search(["n_warehouse_30_4_lost_demand", "n_warehouse_20_3_lost_demand", "n_warehouse_10_2_lost_demand"])
        search_space["repeats"] = tune.grid_search([3])
    if 'GNN_cheapest' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space["repeats"] = tune.grid_search([1, 2, 3])
        search_space["train_dev_sample_and_batch_size"] = tune.grid_search([8192])
    if 'GNN_fastest' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space["repeats"] = tune.grid_search([1, 2, 3])
        search_space["train_dev_sample_and_batch_size"] = tune.grid_search([8192])
    if 'vanilla_n_warehouses' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_n_warehouses": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master_n_warehouses"],
        }
    if 'vanilla_n_warehouses_selfloop' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_n_warehouses_selfloop": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master_n_warehouses_selfloop"],
        }


if "finals_n_warehouses_n_stores_no_edge_cost" == testset_name:
    configs = ["n_warehouse_50_6_lost_demand_no_edge_cost", "n_warehouse_40_5_lost_demand_no_edge_cost", "n_warehouse_30_4_lost_demand_no_edge_cost", "n_warehouse_20_3_lost_demand_no_edge_cost", "n_warehouse_10_2_lost_demand_no_edge_cost"]
    common_setups = {
        "config": tune.grid_search(configs),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "store_underage_cost": tune.grid_search([9]),

        "dev_periods": tune.grid_search([100]),
        "dev_ignore_periods": tune.grid_search([60]),

        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([128, 1024, 8192]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'GNN_bottleneck' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
        search_space["repeats"] = tune.grid_search([1])
        search_space["train_dev_sample_and_batch_size"] = tune.grid_search([8192])

        search_space["learning_rate"] = tune.grid_search([0.0001])
        search_space["config"] = tune.grid_search(["n_warehouse_50_6_lost_demand_no_edge_cost"])
    if 'GNN_cheapest_holding' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space["repeats"] = tune.grid_search([1])
        search_space["train_dev_sample_and_batch_size"] = tune.grid_search([8192])
    if 'GNN_fastest' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([1]),
        }
        search_space["repeats"] = tune.grid_search([1])
        search_space["train_dev_sample_and_batch_size"] = tune.grid_search([8192])

        search_space["learning_rate"] = tune.grid_search([0.0001])
        search_space["config"] = tune.grid_search(["n_warehouse_50_6_lost_demand_no_edge_cost"])
    if 'vanilla_n_warehouses' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_n_warehouses": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master_n_warehouses"],
        }

if "separate_store" == testset_name:
    import os
    import glob
    yml_files = []
    folder_paths = [
        # "/user/ml4723/Prj/NIC/config_files/settings/separate/n_warehouse_21_2_real_lost_demand"
        # "/user/ml4723/Prj/NIC/config_files/settings/separate/n_warehouse_21_3_real_lost_demand", 
        "/user/ml4723/Prj/NIC/config_files/settings/separate/n_warehouse_21_5_real_lost_demand", 
        # "/user/ml4723/Prj/NIC/config_files/settings/separate/n_warehouse_21_4_real_lost_demand", 
    ]
    
    for folder_path in folder_paths:
        if os.path.isdir(folder_path):
            for yml_file in glob.glob(os.path.join(folder_path, '*.yml')):
                yml_files.append(yml_file.split('settings/')[1].replace('.yml', ''))
    common_setups = {
        "config": tune.grid_search(yml_files),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "samples": tune.grid_search([1]),
        "repeats": tune.grid_search([1, 2, 3]),
    }
    if "data_driven_net_n_warehouses_real" == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master_n_warehouses": tune.grid_search([64, 128]),
            "overriding_networks": ["master_n_warehouses"],
        }

if "separated_networks_demands_signal_oneuniverse_highcap" == testset_name:
    common_setups = {
        "config": tune.grid_search([testset_name]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "samples": tune.grid_search([1]),
        "repeats": tune.grid_search([1, 2, 3]),
        'different_for_each_sample': tune.grid_search([True]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192]),
        "train_batch_size": tune.grid_search([1024]),
        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),

        "store_underage_cost": tune.grid_search([[2, 6, 10]]),
        "warehouse_demands_cap": tune.grid_search([20, 22.5, 25, 27.5, 30, 32.5]),
    }
    if 'GNN_transshipment' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if 'GNN_decentralized_transshipment' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }

if "n_warehouse_1_2_edge_cost" == testset_name:
    config = "n_warehouse_1_2_lost_demand_edge_cost"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),

        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192]),#, 1024, 128]),
        "train_batch_size": tune.grid_search([2048]),
        "dev_periods": tune.grid_search([100]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }
    if 'GNN_cheapest' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "n_MP": tune.grid_search([2]),
        }

if "censored_demands" == testset_name:
    config = "one_store_lost_censored"
    common_setups = {
        "config": tune.grid_search([config]),
        "repeats": tune.grid_search([1, 2, 3]),
        "store_lead_time": tune.grid_search([2]),
        "censor_demands_for_train_and_dev": tune.grid_search(["weibull"]),
        "censoring_threshold": tune.grid_search([5, 6, 7, 8]),
        "weibull_fixed_lambda": tune.grid_search([1.0, 2.0, 3.0]),
        "weibull_k": tune.grid_search([0.8, 1.0, 1.2]),
    }
    if 'vanilla_one_store' == hyperparams_name:
        search_space = { **common_setups }
    if 'capped_base_stock' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.1, 0.01]),
        }

if "censored_demands_kaplanmeier" == testset_name:
    config = "one_store_lost_censored"
    common_setups = {
        "config": tune.grid_search([config]),
        "samples": tune.grid_search([17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]),
        "repeats": tune.grid_search([1, 2, 3]),
        "store_lead_time": tune.grid_search([2]),
        "censor_demands_for_train_and_dev": tune.grid_search(["kaplanmeier"]),
        "kaplanmeier_n_fit": tune.grid_search([10**2, 10**3, 10**4, 10**5]),
    }
    if 'vanilla_one_store' == hyperparams_name:
        search_space = { **common_setups }
    if 'capped_base_stock' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([1.0, 0.5, 0.1]),
        }

if "finals_weight_sharing" == testset_name:
    config = "n_stores_lost_demand_optimal"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        
        "samples": tune.grid_search([1]),
        'different_for_each_sample': tune.grid_search([True]),
        "repeats": tune.grid_search([1, 2, 3]),
        'train_dev_sample_and_batch_size': tune.grid_search([8192, 1024, 128]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),

        "n_stores": tune.grid_search([50, 30, 20, 10, 5, 3]),
    }
    if 'vanilla_n_stores' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }
    if 'n_stores_shared_net' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }
    if 'n_stores_per_store_net' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }

if 'generic_architecture_real' == testset_name:
    config = "one_warehouse_lost_demand_real"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "store_underage_cost": tune.grid_search([2, 5, 8, 11]),
        "samples": tune.grid_search([1, 2, 3]),
    }
    if 'GNN_real' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'GNN_skip_connection_real' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if "symmetry_aware_real" == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "apply_normalization": tune.grid_search([False]),
            "store_orders_for_warehouse": tune.grid_search([False]),
            "omit_context_from_store_input": tune.grid_search([False]),
            # "store_orders_for_warehouse": tune.grid_search([False]),
        }
    if 'decentralized_real' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "apply_normalization": tune.grid_search([False]),
            "store_orders_for_warehouse": tune.grid_search([False]),
            "omit_context_from_store_input": tune.grid_search([True]),
        }
    if "data_driven_net_real" == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }
    if 'transformed_nv_one_warehouse_real' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.1, 0.03, 0.01, 0.003, 0.001]),
        }
    if "just_in_time_real" == hyperparams_name:
        search_space = { **common_setups,
        }

else:
    if 'pretrained_store' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "include_context_for_warehouse_input": tune.grid_search([True]),
            "training_batch_size": tune.grid_search([4096]),
            "samples": tune.grid_search([1, 2, 3, 4]),
        }
        save_path = 'ray_results/warehouse_varying_underage_cost/pretrained_store'

    if 'vanilla_one_store_for_warehouse' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            # "training_batch_size": tune.grid_search([4096]),
            "store_underage_cost": tune.grid_search([9]),
            "samples": tune.grid_search([1, 2, 3, 4]),
        }
        save_path = 'ray_results/one_store/'
    elif 'transformed_nv_no_quantile_one_warehouse' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.03, 0.01, 0.003, 0.001]), # 0.03, 0.01, 0.003, 0.001
            "training_n_samples": tune.grid_search([1, 2, 4, 8]),
            "repeats": tune.grid_search([1, 2, 3]),
            "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        }
        save_path = 'ray_results/sample_efficiency/transformed_nv_no_quantile_one_warehouse'
    elif 'transformed_nv_no_quantile_no_upper_bound_one_warehouse' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.03, 0.01, 0.003, 0.001]),
            "training_n_samples": tune.grid_search([16, 256, 8192]),
            "repeats": tune.grid_search([1, 2, 3]),
            "samples": tune.grid_search([1]), # just one sample when training_n_samples = 16, 256, 8192
        }
        save_path = 'ray_results/sample_efficiency/transformed_nv_no_quantile_no_upper_bound_one_warehouse'
    elif 'transformed_nv_calculated_quantile_one_warehouse' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.03, 0.01, 0.003, 0.001]),
            "training_n_samples": tune.grid_search([16, 256, 8192]),
            "repeats": tune.grid_search([1, 2, 3]),
            "samples": tune.grid_search([1]), # just one sample when training_n_samples = 16, 256, 8192
        }
        save_path = 'ray_results/sample_efficiency/transformed_nv_calculated_quantile_one_warehouse'
    elif 'transformed_nv_no_quantile_sep_stores_one_warehouse' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.03, 0.01, 0.003, 0.001, 0.0003, 0.0001]),
            "training_n_samples": tune.grid_search([16, 256, 8192]),
            "repeats": tune.grid_search([1, 2, 3]),
            "samples": tune.grid_search([1]), # just one sample when training_n_samples = 16, 256, 8192
        }
        save_path = 'ray_results/sample_efficiency/transformed_nv_no_quantile_sep_stores_one_warehouse'

    elif 'cons_weekly_forecast_NN' == hyperparams_name:
        search_space = {
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "samples": tune.grid_search([0, 1, 2, 3, 4, 5]),
        }
    elif 'cons_data_driven_net' == hyperparams_name:
        search_space = {
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "samples": tune.grid_search([0, 1, 2, 3, 4, 5]),
        }
        save_path = 'ray_results/cons/data_driven_net'
    elif 'cons_fixed_quantile' == hyperparams_name:
        search_space = {
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "samples": tune.grid_search([0, 1, 2, 3, 4, 5]),
        }
        save_path = 'ray_results/cons/fixed_quantile'
    elif 'cons_quantile_nv' == hyperparams_name:
        search_space = {
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "samples": tune.grid_search([0]),
        }
        save_path = 'ray_results/cons/quantile_nv'
    elif 'cons_just_in_time' == hyperparams_name:
        search_space = {
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "samples": tune.grid_search([0]),
        }
        save_path = 'ray_results/cons/just_in_time'
trainable_with_resources = tune.with_resources(run, {"cpu": n_cpus_per_instance, "gpu": gpus_per_instance})
if n_stores != None:
    save_path += f'/{n_stores}'
    search_space['n_stores'] = n_stores

tuner = tune.Tuner(trainable_with_resources
, param_space=search_space
, run_config=train.RunConfig(storage_path=os.path.join(os.getcwd(), save_path)))

results = tuner.fit()
ray.shutdown()