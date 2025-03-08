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
for filename in os.listdir(config_settings_dir):
    if filename.endswith('.yml'):
        setting_name = filename[:-4]  # Remove .yml extension
        with open(os.path.join(config_settings_dir, filename), 'r') as file:
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
    main_run = MainRun("train", config_settings[tuning_configs['config']], config_hyperparams, tuning_configs)
    main_run.run()

num_gpus = len(gpus_to_use)
num_instances = num_gpus * num_instances_per_gpu

gpus_per_instance = num_gpus / num_instances
ray.init(num_cpus = num_instances * n_cpus_per_instance, num_gpus = num_gpus, object_store_memory=4000000000, address='local')

save_path = f'ray_results/{testset_name}/{hyperparams_name}'
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

if "models_overfit_in_many_stores_test" == testset_name:
    config = "n_stores_lost_demand"
    common_setups = {
        "config": tune.grid_search([config]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "samples": tune.grid_search([1, 2, 3]),

        'train_dev_sample_and_batch_size': tune.grid_search([8192, 256]),
        "train_batch_size": tune.grid_search([1024]),

        "test_n_samples": tune.grid_search([8192]),
        "test_batch_size": tune.grid_search([8192]),

        # "dev_periods": tune.grid_search([150]),
        # "trian_periods": tune.grid_search([100]),
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
            "master": tune.grid_search([32]),
            "overriding_networks": ["master"],
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
    if 'GNN_MP_real' == hyperparams_name:
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


if "generic_architecture_n_warehouse" == testset_name:
    config = "n_warehouse_lost_demand"
    common_setups = {
        "config": tune.grid_search([config]),
        "store_underage_cost": tune.grid_search([8]),
        "train_n_samples": tune.grid_search([16]),
        "train_batch_size": tune.grid_search([16]),
        "dev_n_samples": tune.grid_search([16]),
        "dev_batch_size": tune.grid_search([16]),
        # "train_n_samples": tune.grid_search([32768]),
        # "train_batch_size": tune.grid_search([4096]),
        # "dev_n_samples": tune.grid_search([32768]),
        # "dev_batch_size": tune.grid_search([32768]),
        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "samples": tune.grid_search([1, 2, 3]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "n_MP": tune.grid_search([2]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'GNN_longer_dev' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "dev_periods": tune.grid_search([100]),
            "n_MP": tune.grid_search([3]),
            # "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'GNN_pna' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "n_MP": tune.grid_search([3]),
        }
    if 'GNN_NN_per_layer' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "n_MP": tune.grid_search([3]),
        }
    if 'GNN_edge_embedding' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "n_MP": tune.grid_search([3]),
        }
    if 'GNN_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'vanilla_n_warehouses' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }
    if 'vanilla_n_warehouses_longer_dev' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "dev_periods": tune.grid_search([100]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }

if "generic_architecture" == testset_name:
    config = "one_warehouse_lost_demand"
    common_setups = {
        "config": tune.grid_search([config]),
        # "train_n_samples": tune.grid_search([16]),
        # "train_batch_size": tune.grid_search([16]),
        # "dev_n_samples": tune.grid_search([16]),
        # "dev_batch_size": tune.grid_search([16]),
        "train_n_samples": tune.grid_search([32768]),
        "train_batch_size": tune.grid_search([4096]),
        "dev_n_samples": tune.grid_search([32768]),
        "dev_batch_size": tune.grid_search([32768]),
        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        "samples": tune.grid_search([1, 2, 3]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'GNN_MP_shared_weights' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'GNN_MP_individual_weights' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'CBS_one_warehouse' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([1.0, 0.5, 0.1, 0.05, 0.01, 0.005]),
        }
    if 'GNN_MP' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'vanilla_one_warehouse' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }
    if 'symmetry_aware' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
        }

if "generic_architecture_hard" == testset_name:
    config = "one_warehouse_lost_demand_hard"
    common_setups = {
        "config": tune.grid_search([config]),
        "train_n_samples": tune.grid_search([8192]),
        "dev_n_samples": tune.grid_search([4096]),
        "test_n_samples": tune.grid_search([4096]),
        "train_batch_size": tune.grid_search([1024]),
        "dev_batch_size": tune.grid_search([4096]),
        "test_batch_size": tune.grid_search([4096]),
        "stop_if_no_improve_for_epochs": tune.grid_search([1000]),
        "samples": tune.grid_search([1, 2, 3]),
    }
    if 'GNN_MP' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'GNN_MP_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'GNN_MP_NN_per_layer' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'GNN_MP_attention' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.001, 0.0001, 0.00003]),
        }
    # also can try separated layers
    # multi-head of attention
if "generic_architecture_transshipment" == testset_name:
    config = "transshipment_backlogged"
    common_setups = {
        "config": tune.grid_search([config]),
        # "store_lead_time": tune.grid_search([2, 6]),
        # "store_underage_cost": tune.grid_search([4, 9]),
        # "stores_correlation": tune.grid_search([0.0, 0.5]),
        "store_lead_time": tune.grid_search([6]),
        "store_underage_cost": tune.grid_search([9]),
        "stores_correlation": tune.grid_search([0.5]),
        "samples": tune.grid_search([1, 2, 3]),
        # "train_n_samples": tune.grid_search([32768]),
        # "train_batch_size": tune.grid_search([1024]),
        # "dev_n_samples": tune.grid_search([32768]),
        # "dev_batch_size": tune.grid_search([32768]),
        "train_n_samples": tune.grid_search([16]),
        "train_batch_size": tune.grid_search([2048]),
        "dev_n_samples": tune.grid_search([16]),
        "dev_batch_size": tune.grid_search([16]),
        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
    }
    if 'GNN_MP_transshipment' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'GNN_MP_transshipment_varying_training_primitives' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
        search_space['config'] = tune.grid_search(["transshipment_backlogged_varying_training_primitives"])
    if 'vanilla_transshipment' == hyperparams_name:  
        search_space = {**common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([512, 256, 128]),
            "overriding_networks": ["master"],
        }

if "generic_architecture_serial_hard" == testset_name:
    config = "serial_system_hard"
    common_setups = {
        "config": tune.grid_search([config]),
        # "store_lead_time": tune.grid_search([1, 2, 3, 4]),
        # "store_underage_cost": tune.grid_search([4, 9, 19, 39]),
        "store_lead_time": tune.grid_search([2]),
        "store_underage_cost": tune.grid_search([9]),


        "train_n_samples": tune.grid_search([16]),
        "train_batch_size": tune.grid_search([16]),
        "dev_n_samples": tune.grid_search([16]),
        "dev_batch_size": tune.grid_search([16]),
        "stop_if_no_improve_for_epochs": tune.grid_search([1000]),
        "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8]),
        # "train_n_samples": tune.grid_search([32768]),
        # "train_batch_size": tune.grid_search([4096]),
        # "dev_n_samples": tune.grid_search([32768]),
        # "dev_batch_size": tune.grid_search([32768]),
        # "stop_if_no_improve_for_epochs": tune.grid_search([500]),
        # "samples": tune.grid_search([1, 2, 3]),


        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
        "early_stop_check_epochs": tune.grid_search([50]),
    }
    if 'GNN_MP' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "save_model_for_all_epochs": tune.grid_search([True]),
            "initial_bias_output": tune.grid_search([5.0]),
        }
    if 'GNN_MP_bottleneck_loss' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'GNN_MP_bottleneck_loss_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'GNN_MP_bottleneck_loss_stop_gradient' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'GNN_MP_bottleneck_loss_stop_gradient_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'GNN_MP_layer_normalization' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'vanilla_serial_hard' == hyperparams_name:
        search_space = { **common_setups,
            "master": tune.grid_search([128]),
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
            "overriding_networks": ["master"],
            # "save_model_for_all_epochs": tune.grid_search([True]),
        }
    if 'vanilla_serial_hard_proportional' == hyperparams_name:
        search_space = { **common_setups,
            "master": tune.grid_search([128]),
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
            "overriding_networks": ["master"],
        }
    if 'GNN_MP_NN_per_layer' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.001, 0.0001]),
        }
    if 'GNN_MP_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'GNN_MP_NN_per_layer_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'echelon_stock_hard' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.5, 0.1, 0.03]),
        }

if "generic_architecture_serial_hard_initialize" == testset_name:
    config = "serial_system_hard_initialize"
    common_setups = {
        "config": tune.grid_search([config]),
        # "store_lead_time": tune.grid_search([1, 2, 3, 4]),
        # "store_underage_cost": tune.grid_search([4, 9, 19, 39]),
        "store_lead_time": tune.grid_search([2]),
        "store_underage_cost": tune.grid_search([9]),
        "train_n_samples": tune.grid_search([16]),
        "train_batch_size": tune.grid_search([16]),
        "dev_n_samples": tune.grid_search([16]),
        "dev_batch_size": tune.grid_search([16]),
        "stop_if_no_improve_for_epochs": tune.grid_search([1000]),
        "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8]),
        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
        "early_stop_check_epochs": tune.grid_search([50]),
    }
    if 'GNN_MP' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
            "save_model_for_all_epochs": tune.grid_search([True]),
            "initial_bias_output": tune.grid_search([0.0]),
        }

if "generic_architecture_serial" == testset_name:
    config = "serial_system"
    common_setups = {
        "config": tune.grid_search([config]),
        # "store_lead_time": tune.grid_search([1, 2, 3, 4]),
        # "store_underage_cost": tune.grid_search([4, 9, 19, 39]),
        # "train_n_samples": tune.grid_search([32768]),
        # "train_batch_size": tune.grid_search([1024]),
        # "dev_n_samples": tune.grid_search([32768]),
        # "dev_batch_size": tune.grid_search([32768]),
        "store_lead_time": tune.grid_search([4]),
        "store_underage_cost": tune.grid_search([4, 9, 19, 39]),
        "train_n_samples": tune.grid_search([16]),
        "train_batch_size": tune.grid_search([8192]),
        "dev_n_samples": tune.grid_search([16]),
        "dev_batch_size": tune.grid_search([16]),
        "test_n_samples": tune.grid_search([32768]),
        "test_batch_size": tune.grid_search([32768]),
        "samples": tune.grid_search([1, 2, 3]),
        "early_stop_check_epochs": tune.grid_search([10]),
        "stop_if_no_improve_for_epochs": tune.grid_search([500]),
    }
    if 'GNN' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'GNN_MP' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
    if 'GNN_MP_varying_training_primitives' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
        search_space['config'] = tune.grid_search(["serial_system_varying_training_primitives"])
    if 'GNN_MP_skip_connection_varying_training_primitives' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
        search_space['config'] = tune.grid_search(["serial_system_varying_training_primitives"])
    if 'GNN_MP_skip_connection_varying_training_primitives_normal' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "gradient_clipping_norm_value": tune.grid_search([1.0]),
        }
        search_space['config'] = tune.grid_search(["serial_system_varying_training_primitives"])
    if 'vanilla_serial' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            "master": tune.grid_search([128, 256, 512]),
            "overriding_networks": ["master"],
        }
    if 'GNN_MP_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.001, 0.0001]),
        }
    if 'GNN_MP_NN_per_layer_skip_connection' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
        }
    if 'GNN_MP_NN_per_layer' == hyperparams_name:
        search_space = { **common_setups,
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001, 0.00003]),
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
    elif 'symmetry_aware_decentralized_real' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            'context': tune.grid_search([0]),
            "overriding_networks": ["context"],
            "overriding_outputs": ["context"],
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "training_n_samples": tune.grid_search([416]),
            "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        }
        save_path = 'ray_results/warehouse_real/symmetry_aware_decentralized'
    elif 'symmetry_aware_decentralized_real_normalized' == hyperparams_name:
        search_space = {
            "learning_rate": tune.grid_search([0.01, 0.001, 0.0001]),
            'context': tune.grid_search([0]),
            "overriding_networks": ["context"],
            "overriding_outputs": ["context"],
            "training_n_samples": tune.grid_search([416]),
            "store_underage_cost": tune.grid_search([4, 6, 9, 13]),
            "samples": tune.grid_search([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]),
        }
        save_path = 'ray_results/warehouse_real/symmetry_aware_decentralized_normalized'
trainable_with_resources = tune.with_resources(run, {"cpu": n_cpus_per_instance, "gpu": gpus_per_instance})
if n_stores != None:
    save_path += f'/{n_stores}'
    search_space['n_stores'] = n_stores

tuner = tune.Tuner(trainable_with_resources
, param_space=search_space
, run_config=train.RunConfig(storage_path=os.path.join(os.getcwd(), save_path)))

results = tuner.fit()
ray.shutdown()