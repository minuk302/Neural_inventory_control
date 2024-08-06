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
    def __init__(self, config_setting, config_hyperparams, start_recording = False):
        self.is_recording = start_recording
        self.config_setting = config_setting
        self.config_hyperparams = config_hyperparams

    def start_recording(self):
        self.is_recording = True
    
    def on_step(self, s_underage_costs, s_holding_costs, w_holding_costs, warehouse_orders):
        if self.is_recording == False:
            return
        
        holding_cost = self.config_setting['warehouse_params']['holding_cost']
        lead_time = self.config_setting['warehouse_params']['lead_time']
        correlation = self.config_setting['store_params']['demand']['correlation']
        context = self.config_hyperparams['nn_params']['neurons_per_hidden_layer']['context']
        file_name = f"analysis/results/primitive/{holding_cost}_{lead_time}_{correlation}_{context}.csv"
        def append_tensors_to_csv(filename, s_underage_costs, s_holding_costs, w_holding_costs, warehouse_orders):
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            df_new = pd.DataFrame({'s_underage_costs': s_underage_costs, 's_holding_costs': s_holding_costs
                                   , 'w_holding_costs': w_holding_costs, 'warehouse_orders': warehouse_orders})
            if os.path.exists(filename):
                df_new.to_csv(filename, mode='a', header=False, index=False)
            else:
                df_new.to_csv(filename, index=False)
        append_tensors_to_csv(file_name, s_underage_costs, s_holding_costs, w_holding_costs, warehouse_orders)