from shared_imports import *

def override_configs(overriding_params, config_setting, config_hyperparams):
    config_setting = copy.deepcopy(config_setting)
    config_hyperparams = copy.deepcopy(config_hyperparams)
    if 'n_stores' in overriding_params:
        config_setting['problem_params']['n_stores'] = overriding_params['n_stores']
    
    if 'learning_rate' in overriding_params:
        config_hyperparams['optimizer_params']['learning_rate'] = overriding_params['learning_rate']

    if 'warehouse_holding_cost' in overriding_params:
        config_setting['warehouse_params']['holding_cost'] = overriding_params['warehouse_holding_cost']

    if 'warehouse_lead_time' in overriding_params:
        config_setting['warehouse_params']['lead_time'] = overriding_params['warehouse_lead_time']

    if 'stores_correlation' in overriding_params:
        config_setting['store_params']['demand']['correlation'] = overriding_params['stores_correlation']

    for net in overriding_params['overriding_networks']:
        if 'for_all_networks' in overriding_params:
            size = overriding_params['for_all_networks']
        else:
            size = overriding_params[net]
        config_hyperparams['nn_params']['neurons_per_hidden_layer'][net] = [size for _ in config_hyperparams['nn_params']['neurons_per_hidden_layer'][net]]

    for net in overriding_params['overriding_outputs']:
        if 'for_all_networks' in overriding_params:
            size = overriding_params['for_all_networks']
        else:
            size = overriding_params[net]
        if net not in config_hyperparams['nn_params']['output_sizes']:
            continue
        if size == 0:
            del config_hyperparams['nn_params']['output_sizes'][net]
            continue
        config_hyperparams['nn_params']['output_sizes'][net] = size
    
    return config_setting, config_hyperparams