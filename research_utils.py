from shared_imports import *

def override_configs(overriding_params, config_setting, config_hyperparams):
    config_setting = copy.deepcopy(config_setting)
    config_hyperparams = copy.deepcopy(config_hyperparams)
    if 'n_stores' in overriding_params:
        config_setting['problem_params']['n_stores'] = overriding_params['n_stores']

    if 'different_for_each_sample' in config_setting['seeds'] and config_setting['seeds']['different_for_each_sample'] == True:
        config_setting['seeds']['demand'] = config_setting['seeds']['demand'] + overriding_params['samples'] * 100

    if 'training_n_samples' in overriding_params:
        config_setting['params_by_dataset']['train']['n_samples'] = overriding_params['training_n_samples']
        config_setting['params_by_dataset']['train']['batch_size'] = min(1024, overriding_params['training_n_samples'])
    
    if 'learning_rate' in overriding_params:
        config_hyperparams['optimizer_params']['learning_rate'] = overriding_params['learning_rate']

    if 'warehouse_holding_cost' in overriding_params:
        config_setting['warehouse_params']['holding_cost'] = overriding_params['warehouse_holding_cost']

    if 'warehouse_lead_time' in overriding_params:
        config_setting['warehouse_params']['lead_time'] = overriding_params['warehouse_lead_time']

    if 'stores_correlation' in overriding_params:
        config_setting['store_params']['demand']['correlation'] = overriding_params['stores_correlation']

    if 'n_sub_sample_for_context' in overriding_params:
        config_hyperparams['nn_params']['n_sub_sample_for_context'] = overriding_params['n_sub_sample_for_context']

    if 'apply_normalization' in overriding_params:
        config_hyperparams['nn_params']['apply_normalization'] = overriding_params['apply_normalization']

    if 'store_orders_for_warehouse' in overriding_params:
        config_hyperparams['nn_params']['store_orders_for_warehouse'] = overriding_params['store_orders_for_warehouse']
        if overriding_params['store_orders_for_warehouse']:
            config_hyperparams['nn_params']['output_sizes']['store'] = 2
            del config_hyperparams['nn_params']['output_sizes']['warehouse']
    
    if 'omit_context_from_store_input' in overriding_params:
        config_hyperparams['nn_params']['omit_context_from_store_input'] = overriding_params['omit_context_from_store_input']

    if 'store_underage_cost' in overriding_params:
        if 'range' in config_setting['store_params']['underage_cost']:
            current_range = config_setting['store_params']['underage_cost']['range']
            current_mean = sum(current_range) / 2
            current_low_deviation_ratio = (current_mean - current_range[0]) / current_mean
            current_high_deviation_ratio = (current_range[1] - current_mean) / current_mean

            new_mean = overriding_params['store_underage_cost']
            new_lower = new_mean * (1 - current_low_deviation_ratio)
            new_upper = new_mean * (1 + current_high_deviation_ratio)
            
            config_setting['store_params']['underage_cost']['range'] = [new_lower, new_upper]
        else:
            config_setting['store_params']['underage_cost']['value'] = overriding_params['store_underage_cost']

    if 'overriding_networks' in overriding_params:
        for net in overriding_params['overriding_networks']:
            if 'for_all_networks' in overriding_params:
                size = overriding_params['for_all_networks']
            else:
                size = overriding_params[net]
            config_hyperparams['nn_params']['neurons_per_hidden_layer'][net] = [size for _ in config_hyperparams['nn_params']['neurons_per_hidden_layer'][net]]

    if 'overriding_outputs' in overriding_params:
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


class Recorder():
    def __init__(self, config_setting, config_hyperparams, start_recording = False, recorder_identifier=None):
        self.is_recording = start_recording
        self.config_setting = config_setting
        self.config_hyperparams = config_hyperparams
        self.recorder_identifier = recorder_identifier

    def start_recording(self):
        self.is_recording = True

    def on_step_store(self, s_underage_costs, s_holding_costs):
        if self.is_recording == False:
            return
        underage_cost = self.config_setting['store_params']['underage_cost']['value']
        filename = f"analysis/results/cons/{self.config_hyperparams['nn_params']['name']}/{underage_cost}.csv"
        def append_tensors_to_csv():
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df_new = pd.DataFrame({'s_underage_costs': s_underage_costs, 's_holding_costs': s_holding_costs})
            if os.path.exists(filename):
                df_new.to_csv(filename, mode='a', header=False, index=False)
            else:
                df_new.to_csv(filename, index=False)
        append_tensors_to_csv()

    def on_step(self, s_underage_costs, s_holding_costs, w_holding_costs, warehouse_orders):
        if self.is_recording == False:
            return

        underage_cost_range = self.config_setting['store_params']['underage_cost']['range']
        underage_cost = round(sum(underage_cost_range) / 2)
        file_name = f"analysis/results/one_warehouse_real/{self.recorder_identifier}/{underage_cost}.csv"
        def append_tensors_to_csv(filename, s_underage_costs, s_holding_costs, w_holding_costs, warehouse_orders):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df_new = pd.DataFrame({'s_underage_costs': s_underage_costs, 's_holding_costs': s_holding_costs
                                   , 'w_holding_costs': w_holding_costs, 'warehouse_orders': warehouse_orders})
            if os.path.exists(filename):
                df_new.to_csv(filename, mode='a', header=False, index=False)
            else:
                df_new.to_csv(filename, index=False)
        append_tensors_to_csv(file_name, s_underage_costs, s_holding_costs, w_holding_costs, warehouse_orders)