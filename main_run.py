#!/user/ml4723/.conda/envs/neural_inventory_control/bin/python

import yaml
from trainer import *
import sys
import research_utils
import json

class MainRun:
    def __init__(self, train_or_test, setting_name, hyperparams_name, recorder_config_path=None, recorder_identifier=None):
        self.train_or_test = train_or_test
        self.setting_name = setting_name
        self.hyperparams_name = hyperparams_name
        self.recorder_config_path = recorder_config_path

        self.config_setting_file = f'config_files/settings/{setting_name}.yml'
        self.config_hyperparams_file = f'config_files/policies_and_hyperparams/{hyperparams_name}.yml'

        def load_yaml(file_path):
            with open(file_path, 'r') as file:
                return yaml.safe_load(file)
        self.config_setting = load_yaml(self.config_setting_file)
        self.config_hyperparams = load_yaml(self.config_hyperparams_file)

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
        path = os.path.join(self.recorder_config_path, 'params.json')
        with open(path, 'r') as file:
            params = json.load(file)
            self.config_setting, self.config_hyperparams = research_utils.override_configs(params, self.config_setting, self.config_hyperparams)
        model_path = os.path.join(self.recorder_config_path, 'model.pt')
        self.config_hyperparams['trainer_params']['load_model_path'] = model_path
        self.config_hyperparams['trainer_params']['load_previous_model'] = os.path.exists(model_path)

    def extract_configs(self):
        setting_keys = 'seeds', 'test_seeds', 'problem_params', 'params_by_dataset', 'observation_params', 'store_params', 'warehouse_params', 'echelon_params', 'sample_data_params'
        self.seeds, self.test_seeds, self.problem_params, self.params_by_dataset, self.observation_params, self.store_params, self.warehouse_params, self.echelon_params, self.sample_data_params = [
            self.config_setting[key] for key in setting_keys
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
                by_period=True,
                periods_for_split=[self.sample_data_params[k] for k in ['train_periods', 'dev_periods', 'test_periods']]
            )
        else:
            self.scenario = Scenario(
                periods=self.params_by_dataset['train']['periods'],
                problem_params=self.problem_params,
                store_params=self.store_params,
                warehouse_params=self.warehouse_params,
                echelon_params=self.echelon_params,
                num_samples=self.params_by_dataset['train']['n_samples'] + self.params_by_dataset['dev']['n_samples'],
                observation_params=self.observation_params,
                seeds=self.seeds
            )

            self.train_dataset, self.dev_dataset = self.dataset_creator.create_datasets(
                self.scenario,
                split=True,
                by_sample_indexes=True,
                sample_index_for_split=self.params_by_dataset['dev']['n_samples']
            )

            self.scenario = Scenario(
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

            self.test_dataset = self.dataset_creator.create_datasets(self.scenario, split=False)

    def create_data_loaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.params_by_dataset['train']['batch_size'], shuffle=True)
        dev_loader = DataLoader(self.dev_dataset, batch_size=self.params_by_dataset['dev']['batch_size'], shuffle=False)
        test_loader = DataLoader(self.test_dataset, batch_size=self.params_by_dataset['test']['batch_size'], shuffle=False)
        # train_loader = DataLoader(self.train_dataset, batch_size=self.params_by_dataset['train']['batch_size'], shuffle=True, num_workers=8, pin_memory=True, persistent_workers=True)
        # dev_loader = DataLoader(self.dev_dataset, batch_size=self.params_by_dataset['dev']['batch_size'], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        # test_loader = DataLoader(self.test_dataset, batch_size=self.params_by_dataset['test']['batch_size'], shuffle=False, num_workers=8, pin_memory=True, persistent_workers=True)
        self.data_loaders = {'train': train_loader, 'dev': dev_loader, 'test': test_loader}

    def create_model_and_optimizer(self):
        neural_net_creator = NeuralNetworkCreator
        self.model = neural_net_creator().create_neural_network(self.scenario, self.nn_params, device=self.device)
        self.loss_function = PolicyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.optimizer_params['learning_rate'])

    def setup_trainer_params(self):
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
                self.store_params
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

    main_run = MainRun(train_or_test, setting_name, hyperparams_name, recorder_config_path, recorder_identifier)
    import time

    start_time = time.time()
    main_run.run()
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"Execution time: {execution_time:.2f} seconds")