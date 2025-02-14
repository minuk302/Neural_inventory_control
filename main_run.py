#!/user/ml4723/.conda/envs/neural_inventory_control/bin/python

import yaml
from trainer import *
import sys
import research_utils
import json

class MainRun:
    def __init__(self, train_or_test, config_setting, config_hyperparams, tuning_configs = None, recorder_config_path=None, recorder_identifier=None):
        self.train_or_test = train_or_test
        self.config_setting = config_setting
        self.config_hyperparams = config_hyperparams
        self.tuning_configs = tuning_configs
        self.recorder_config_path = recorder_config_path

        if self.tuning_configs is not None:
            self.config_setting, self.config_hyperparams = research_utils.override_configs(self.tuning_configs, self.config_setting, self.config_hyperparams)

        if recorder_config_path is not None:
            self.override_configs()
            start_record = True
            if recorder_identifier is None:
                start_record = False
            self.recorder = research_utils.Recorder(self.config_setting, self.config_hyperparams, start_record, recorder_identifier)
        else:
            self.recorder = research_utils.Recorder(self.config_setting, self.config_hyperparams)

        self.extract_configs()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.dataset_creator = DatasetCreator()

        self.create_scenario_and_datasets()
        self.create_data_loaders()
        self.create_model_and_optimizer()
        self.simulator = Simulator(self.recorder, device=self.device)
        self.trainer = Trainer(device=self.device)

        self.setup_trainer_params()

    def override_configs(self):
        # Use recorder_config_path directly as model path since it points to model_index.pt
        model_path = self.recorder_config_path
        # Get params.json from parent directory
        params_path = os.path.join(os.path.dirname(self.recorder_config_path), 'params.json')
        with open(params_path, 'r') as file:
            params = json.load(file)
            self.config_setting, self.config_hyperparams = research_utils.override_configs(params, self.config_setting, self.config_hyperparams)
        self.config_hyperparams['trainer_params']['load_model_path'] = model_path
        self.config_hyperparams['trainer_params']['load_previous_model'] = os.path.exists(model_path)

        self.config_hyperparams['nn_params']['is_debugging'] = True
        
        # Extract relevant parts from model path to construct debug identifier
        path_parts = model_path.split('/')
        # Find ray_results index and take parts after it
        ray_results_idx = path_parts.index('ray_results')
        relevant_parts = path_parts[ray_results_idx+1:ray_results_idx+3] # Get architecture and model folders
        # Get date from run folder
        run_date = path_parts[ray_results_idx+3].split('_')[1:3] # Get date parts
        run_date = '_'.join(run_date) # Join with /
        # Get run number from next folder
        run_num = path_parts[ray_results_idx+4].split('_')[2] # Get run number
        # Get model name
        model_name = path_parts[-1]
        # Construct debug identifier path
        debug_path = f"/{'/'.join(relevant_parts)}/{run_date}/run_{run_num}/{model_name}"
        self.config_hyperparams['nn_params']['debug_identifier'] = debug_path

    def extract_configs(self):
        setting_keys = 'seeds', 'dev_seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'echelon_params', 'sample_data_params', 'store_training_params', 'warehouse_training_params', 'echelon_training_params'
        self.seeds, self.dev_seeds, self.test_seeds, self.problem_params, self.params_by_dataset, self.observation_params, self.store_params, self.warehouse_params, self.echelon_params, self.sample_data_params, self.store_training_params, self.warehouse_training_params, self.echelon_training_params = [
            self.config_setting[key] if key in self.config_setting else None for key in setting_keys
        ]

        hyperparams_keys = 'trainer_params', 'optimizer_params', 'nn_params'
        self.trainer_params, self.optimizer_params, self.nn_params = [self.config_hyperparams[key] for key in hyperparams_keys]
        self.observation_params = DefaultDict(lambda: None, self.observation_params)

        # temporary for debugging
        if 'range' in self.store_params['underage_cost']:
            self.problem_params['underage_cost'] = sum(self.store_params['underage_cost']['range']) / 2 
        else:
            self.problem_params['underage_cost'] = self.store_params['underage_cost']['value']

    def create_scenario_and_datasets(self):
        if self.sample_data_params['split_by_period']:
            self.scenario = Scenario(
                periods=None,
                problem_params=self.problem_params,
                store_params=self.store_params,
                warehouse_params=self.warehouse_params,
                echelon_params=self.echelon_params,
                num_samples=self.params_by_dataset['train']['n_samples'],
                observation_params=self.observation_params,
                seeds=self.seeds
            )

            self.train_dataset, self.dev_dataset, self.test_dataset = self.dataset_creator.create_datasets(
                self.scenario,
                split=True,
                periods_for_split=[self.sample_data_params[k] for k in ['train_periods', 'dev_periods', 'test_periods']]
            )
        else:
            self.training_scenario = Scenario(
                periods=self.params_by_dataset['train']['periods'],
                problem_params=self.problem_params,
                store_params=self.store_training_params if self.store_training_params else self.store_params,
                warehouse_params=self.warehouse_training_params if self.warehouse_training_params else self.warehouse_params,
                echelon_params=self.echelon_training_params if self.echelon_training_params else self.echelon_params,
                num_samples=self.params_by_dataset['train']['n_samples'],
                observation_params=self.observation_params,
                seeds=self.seeds
            )
            self.train_dataset = self.dataset_creator.create_datasets(self.training_scenario, split=False)

            self.dev_scenario = Scenario(
                periods=self.params_by_dataset['dev']['periods'],
                problem_params=self.problem_params,
                store_params=self.store_params,
                warehouse_params=self.warehouse_params,
                echelon_params=self.echelon_params,
                num_samples=self.params_by_dataset['dev']['n_samples'],
                observation_params=self.observation_params,
                seeds=self.dev_seeds
            )
            self.dev_dataset = self.dataset_creator.create_datasets(self.dev_scenario, split=False)

            self.test_scenario = Scenario(
                self.params_by_dataset['test']['periods'],
                self.problem_params,
                self.store_params,
                self.warehouse_params, 
                self.echelon_params, 
                self.params_by_dataset['test']['n_samples'],
                self.observation_params,
                self.test_seeds,
                True
            )
            self.test_dataset = self.dataset_creator.create_datasets(self.test_scenario, split=False)

    def create_data_loaders(self):
        if self.tuning_configs is None:
            train_loader = DataLoader(self.train_dataset, batch_size=self.params_by_dataset['train']['batch_size'], shuffle=True)
            dev_loader = DataLoader(self.dev_dataset, batch_size=self.params_by_dataset['dev']['batch_size'], shuffle=False)
            test_loader = DataLoader(self.test_dataset, batch_size=self.params_by_dataset['test']['batch_size'], shuffle=False)
        else:
            train_loader = DataLoader(self.train_dataset, batch_size=self.params_by_dataset['train']['batch_size'], shuffle=True, num_workers=self.tuning_configs['n_cpus_per_instance'], pin_memory=True, persistent_workers=True, prefetch_factor=8)
            dev_loader = DataLoader(self.dev_dataset, batch_size=self.params_by_dataset['dev']['batch_size'], shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=1)
            test_loader = DataLoader(self.test_dataset, batch_size=self.params_by_dataset['test']['batch_size'], shuffle=False, num_workers=2, pin_memory=False, prefetch_factor=1)
        self.data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

    def create_model_and_optimizer(self):
        neural_net_creator = NeuralNetworkCreator
        self.model = neural_net_creator().create_neural_network(self.problem_params, self.nn_params, device=self.device)
        self.loss_function = PolicyLoss()

        weight_decay = 0.0
        if 'weight_decay' in self.optimizer_params:
            weight_decay = self.optimizer_params['weight_decay']
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['learning_rate'], weight_decay=weight_decay)

    def setup_trainer_params(self):
        if self.tuning_configs is not None:
            self.trainer_params['base_dir'] = self.tuning_configs['base_dir_for_ray']
            self.trainer_params['save_model_folders'] = []
            self.trainer_params['save_model_filename'] = "model"
        else:
            self.trainer_params['base_dir'] = 'saved_models'
            self.trainer_params['save_model_folders'] = [self.trainer.get_year_month_day(), self.nn_params['name']]
            self.trainer_params['save_model_filename'] = self.trainer.get_time_stamp()
            if self.trainer_params['load_previous_model']:
                self.model, self.optimizer = self.trainer.load_model(self.model, self.optimizer, self.trainer_params['load_model_path'])

    def run(self):
        if self.train_or_test == 'train':
            self.trainer.train(
                self.trainer_params['epochs'],
                self.loss_function,
                self.simulator,
                self.model,
                self.data_loaders,
                self.optimizer,
                self.problem_params,
                self.observation_params,
                self.params_by_dataset,
                self.trainer_params,
                self.store_training_params if self.store_training_params else self.store_params
            )
        elif self.train_or_test == 'test':
            with torch.no_grad():
                average_test_loss, average_test_loss_to_report = self.trainer.test(
                    self.loss_function,
                    self.simulator,
                    self.model,
                    self.data_loaders,
                    self.optimizer,
                    self.problem_params,
                    self.observation_params,
                    self.params_by_dataset,
                    discrete_allocation=self.store_params['demand']['distribution'] == 'poisson'
                )
            print(f'Average per-period test loss: {average_test_loss_to_report}')
        elif self.train_or_test == 'test_on_dev':
            with torch.no_grad():
                average_dev_loss, average_dev_loss_to_report = self.trainer.test_on_dev(
                    self.loss_function,
                    self.simulator,
                    self.model,
                    self.data_loaders,
                    self.optimizer,
                    self.problem_params,
                    self.observation_params,
                    self.params_by_dataset,
                    discrete_allocation=self.store_params['demand']['distribution'] == 'poisson'
                )
            print(f'Average per-period dev loss: {average_dev_loss_to_report}')
        else:
            print(f'Invalid argument: {self.train_or_test}')
            assert False

if __name__ == "__main__":
    train_or_test = sys.argv[1]
    setting_name = sys.argv[2]
    hyperparams_name = sys.argv[3]
    recorder_config_path = sys.argv[4] if len(sys.argv) > 4 else None
    recorder_identifier = sys.argv[5] if recorder_config_path is not None and len(sys.argv) > 5 else None

    def load_yaml(file_path):
        with open(file_path, 'r') as file:
            return yaml.safe_load(file)
    config_setting = load_yaml(f'config_files/settings/{setting_name}.yml')
    config_hyperparams = load_yaml(f'config_files/policies_and_hyperparams/{hyperparams_name}.yml')
    
    main_run = MainRun(train_or_test, config_setting, config_hyperparams, None, recorder_config_path, recorder_identifier)
    import time

    start_time = time.time()
    main_run.run()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")