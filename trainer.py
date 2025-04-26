from shared_imports import *
from environment import *
from loss_functions import *
from ray import train
import time

class Trainer():
    """
    Trainer class
    """

    def __init__(self,  device='cpu'):
        
        self.device = device
        self.time_stamp = self.get_time_stamp()
        self.best_performance_data = {'train_loss': np.inf, 'dev_loss': np.inf, 'last_epoch_saved': -1000, 'model_params_to_save': None}
        self.best_train_loss = np.inf
        self.best_dev_loss = np.inf
    
    def reset(self):
        """
        Reset the losses
        """

    def train(self, epochs, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, trainer_params, store_params):
        """
        Train a parameterized policy

        Parameters:
        ----------
        epochs: int
            Number of epochs to train the policy
        loss_function: LossFunction
            Loss function to use for training.
            In our case, we will use PolicyLoss, that directly calculates the loss as sum of the rewards/costs
        simulator: Simulator(gym.Env)
            Differentiable Gym environment to use for simulating.
            In our experiments, this will simulate a specific inventory problem for a number of periods
        model: nn.Module
            Neural network model to train
        data_loaders: dict
            Dictionary containing the DataLoader objects for train, dev and test datasets
        optimizer: torch.optim
            Optimizer to use for training the model.
            In our experiments we use Adam optimizer
        problem_params: dict
            Dictionary containing the problem parameters, specifying the number of warehouses, stores, whether demand is lost
            and whether to maximize profit or minimize underage+overage costs
        observation_params: dict
            Dictionary containing the observation parameters, specifying which features to include in the observations
        params_by_dataset: dict
            Dictionary containing the parameters for each dataset, such as the number of periods, number of samples, batch size
        trainer_params: dict
            Dictionary containing the parameters for the trainer, such as the number of epochs between saving the model, the base directory
            where to save the model, the filename for the model, whether to save the model, the number of epochs between saving the model
            and the metric to use for choosing the best model
        """

        # trace_handler = tensorboard_trace_handler(dir_name="/user/ml4723/Prj/NIC/analysis/HTA", use_gzip=False)
        # with profile(
        # activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        # on_trace_ready = trace_handler,
        # with_stack = True
        # ) as prof:
        n_passed_epochs_without_improvement = 0
        # torch.cuda.memory._record_memory_history(max_entries=1000000)
        for epoch in range(epochs): # Make multiple passes through the dataset
            # prof.step()
            # if model.is_debugging and epoch == 2:
            #     try:
            #         torch.cuda.memory._dump_snapshot(f"/user/ml4723/Prj/NIC/tensorboard_traces/memory_snapshot.pickle")
            #     except Exception as e:
            #         print(f"Failed to capture memory snapshot {e}")
            #     exit()


            if 'stop_if_no_improve_for_epochs' in trainer_params and n_passed_epochs_without_improvement >= trainer_params['stop_if_no_improve_for_epochs']:
                break
            
            n_passed_epochs_without_improvement += 1
            # Do one epoch of training, including updating the model parameters
            start_time = time.time()
            average_train_loss, average_train_loss_to_report = self.do_one_epoch(
                optimizer, 
                data_loaders['train'], 
                loss_function, 
                simulator, 
                model, 
                params_by_dataset['train']['periods'], 
                problem_params, 
                observation_params, 
                train=True, 
                ignore_periods=params_by_dataset['train']['ignore_periods']
                )
            train_time = time.time() - start_time
            if model.is_debugging:
                print(f"Training time: {train_time:.2f} seconds")
            
            if epoch % trainer_params['do_dev_every_n_epochs'] == 0:
                start_time = time.time()
                # torch.cuda.empty_cache()
                average_dev_loss, average_dev_loss_to_report = self.do_one_epoch(
                    optimizer, 
                    data_loaders['dev'], 
                    loss_function, 
                    simulator, 
                    model, 
                    params_by_dataset['dev']['periods'], 
                    problem_params, 
                    observation_params, 
                    train=False, 
                    ignore_periods=params_by_dataset['dev']['ignore_periods']
                    )
                # torch.cuda.empty_cache()
                dev_time = time.time() - start_time
                if model.is_debugging:
                    print(f"Dev time: {dev_time:.2f} seconds")

                # Check if the current model is the best model so far, and save the model parameters if so.
                # Save the model if specified in the trainer_params
                if_save_model_for_all_epochs = "save_model_for_all_epochs" in trainer_params and trainer_params['save_model_for_all_epochs']
                self.update_best_params_and_save(epoch, average_train_loss_to_report, average_dev_loss_to_report, trainer_params, model, optimizer, if_save_model_for_all_epochs)
                
                if self.update_best_train_or_dev_loss(average_train_loss_to_report, average_dev_loss_to_report, trainer_params):
                    n_passed_epochs_without_improvement = 0

                if 'ray_report_loss' in trainer_params:
                    report_dict = {'dev_loss': average_dev_loss_to_report, 'train_loss': average_train_loss_to_report}
                    if 'report_test_loss' in problem_params and problem_params['report_test_loss'] == True:
                        # torch.cuda.empty_cache()
                        with torch.no_grad():
                            start_time = time.time()
                            average_test_loss, average_test_loss_to_report = self.do_one_epoch(
                                optimizer, 
                                data_loaders['test'], 
                                loss_function, 
                                simulator, 
                                model, 
                                params_by_dataset['test']['periods'], 
                                problem_params, 
                                observation_params,
                                train=False, 
                                ignore_periods=params_by_dataset['test']['ignore_periods'],
                                discrete_allocation=store_params['demand']['distribution'] == 'poisson'
                                )
                            test_time = time.time() - start_time
                            if model.is_debugging:
                                print(f"Test time: {test_time:.2f} seconds")
                            report_dict['test_loss'] = average_test_loss_to_report
                        # torch.cuda.empty_cache()
                    train.report(report_dict)
                    if math.isnan(average_train_loss_to_report):
                        break
            else:
                average_dev_loss, average_dev_loss_to_report = 0, 0

            # Print epoch number and average per-period loss every 10 epochs
            if epoch % trainer_params['print_results_every_n_epochs'] == 0:
                print(f'epoch: {epoch + 1}')
                print(f'Average per-period train loss: {average_train_loss_to_report}')
                print(f'Average per-period dev loss: {average_dev_loss_to_report}')
                print(f'Best per-period dev loss: {self.best_performance_data["dev_loss"]}')
    
    def test(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, discrete_allocation=False):

        if model.trainable and self.best_performance_data['model_params_to_save'] is not None:
            # Load the parameter weights that gave the best performance on the specified dataset
            model.load_state_dict(self.best_performance_data['model_params_to_save'])

        average_test_loss, average_test_loss_to_report = self.do_one_epoch(
                optimizer, 
                data_loaders['test'], 
                loss_function, 
                simulator, 
                model, 
                params_by_dataset['test']['periods'], 
                problem_params, 
                observation_params, 
                train=False, 
                ignore_periods=params_by_dataset['test']['ignore_periods'],
                discrete_allocation=discrete_allocation
                )
        
        return average_test_loss, average_test_loss_to_report
    
    def test_on_dev(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, discrete_allocation=False):

        if model.trainable and self.best_performance_data['model_params_to_save'] is not None:
            # Load the parameter weights that gave the best performance on the specified dataset
            model.load_state_dict(self.best_performance_data['model_params_to_save'])

        average_dev_loss, average_dev_loss_to_report = self.do_one_epoch(
            optimizer, 
            data_loaders['dev'], 
            loss_function, 
            simulator, 
            model, 
            params_by_dataset['dev']['periods'], 
            problem_params, 
            observation_params, 
            train=False, 
            ignore_periods=params_by_dataset['dev']['ignore_periods'],
            discrete_allocation=discrete_allocation
            )
        
        return average_dev_loss, average_dev_loss_to_report
    
    def test_on_train(self, loss_function, simulator, model, data_loaders, optimizer, problem_params, observation_params, params_by_dataset, discrete_allocation=False):

        if model.trainable and self.best_performance_data['model_params_to_save'] is not None:
            # Load the parameter weights that gave the best performance on the specified dataset
            model.load_state_dict(self.best_performance_data['model_params_to_save'])

        average_train_loss, average_train_loss_to_report = self.do_one_epoch(
            optimizer, 
            data_loaders['train'], 
            loss_function, 
            simulator, 
            model, 
            params_by_dataset['train']['periods'], 
            problem_params, 
            observation_params, 
            train=False, 
            ignore_periods=params_by_dataset['train']['ignore_periods'],
            discrete_allocation=discrete_allocation
            )
        
        return average_train_loss, average_train_loss_to_report

    def do_one_epoch(self, optimizer, data_loader, loss_function, simulator, model, periods, problem_params, observation_params, train=True, ignore_periods=0, discrete_allocation=False):
        """
        Do one epoch of training or testing
        """
        def process_batch():
            epoch_loss = 0
            epoch_loss_to_report = 0  # Loss ignoring the first 'ignore_periods' periods
            total_samples = len(data_loader.dataset)
            periods_tracking_loss = periods - ignore_periods  # Number of periods for which we report the loss

            amp_enabled = torch.cuda.is_bf16_supported()
            if any(x in torch.cuda.get_device_name() for x in ('V100', 'T4')):
                amp_enabled = False
            if 'disable_amp' in problem_params and problem_params['disable_amp']:
                amp_enabled = False
            scaler = torch.amp.GradScaler('cuda', enabled = amp_enabled)
            for i, data_batch in enumerate(data_loader):  # Loop through batches of data
                data_batch = self.move_batch_to_device(data_batch)
                if train:
                    # Zero-out the gradient
                    optimizer.zero_grad(set_to_none=True)

                # Forward pass
                with torch.amp.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=amp_enabled):
                    total_reward, reward_to_report = self.simulate_batch(
                        loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods, discrete_allocation
                        )
                    epoch_loss += total_reward.item()  # Rewards from period 0
                    epoch_loss_to_report += reward_to_report.item()  # Rewards from period ignore_periods onwards
                    
                    mean_loss = total_reward/(len(data_batch['demands'])*periods*problem_params['n_stores'])
                
                if train and model.trainable:
                    scaler.scale(mean_loss).backward()
                    scaler.unscale_(optimizer)
                    # mean_loss.backward()
                    if model.gradient_clipping_norm_value is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), model.gradient_clipping_norm_value)
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                # if model.is_debugging:
                #     exit()
            
            return epoch_loss/(total_samples*periods*problem_params['n_stores']), epoch_loss_to_report/(total_samples*periods_tracking_loss*problem_params['n_stores'])
        
        if train:
            model.train()
            return process_batch()

        with torch.no_grad():
            model.eval()
            return process_batch()
    
    def simulate_batch(self, loss_function, simulator, model, periods, problem_params, data_batch, observation_params, ignore_periods=0, discrete_allocation=False):
        """
        Simulate for an entire batch of data, across the specified number of periods
        """

        # Initialize reward across batch
        batch_reward = 0
        reward_to_report = 0

        observation, _ = simulator.reset(periods, problem_params, data_batch, observation_params)
        for t in range(periods):

            # We add internal data to the observation to create non-admissible benchmark policies.
            # No admissible policy should use data stored in _internal_data!
            observation_and_internal_data = {k: v for k, v in observation.items()}
            observation_and_internal_data['internal_data'] = simulator._internal_data

            # Sample action
            action = model(observation_and_internal_data)
            
            if discrete_allocation:  # Round actions to the nearest integer if specified
                action = {key: val.round() for key, val in action.items()}

            observation, reward, terminated, _, _  = simulator.step(action)

            total_reward = loss_function(None, action, reward)
            if t >= ignore_periods:
                reward_to_report += total_reward

            # don't include bottleneck loss in reporting rewards
            if 'bottleneck_loss' in action:
                total_reward += action['bottleneck_loss'].sum()
            batch_reward += total_reward
            
            if terminated:
                break

        # Return reward
        return batch_reward, reward_to_report

    def save_model(self, epoch, model, optimizer, trainer_params, if_save_model_for_all_epochs=False):

        path = self.create_many_folders_if_not_exist_and_return_path(base_dir=trainer_params['base_dir'], 
                                                                     intermediate_folder_strings=trainer_params['save_model_folders']
                                                                     )
        file_name = f"{trainer_params['save_model_filename']}"
        if if_save_model_for_all_epochs:
            file_name += f"_{epoch}"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.best_performance_data['model_params_to_save'],
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_train_loss': self.best_performance_data['train_loss'],
                    'best_dev_loss': self.best_performance_data['dev_loss'],
                    }, 
                    f"{path}/{file_name}.pt"
                    )
    
    def create_folder_if_not_exists(self, folder):
        """
        Create a directory in the corresponding file, if it does not already exist
        """

        if not os.path.isdir(folder):
            os.mkdir(folder)
    
    def create_many_folders_if_not_exist_and_return_path(self, base_dir, intermediate_folder_strings):
        """
        Create a directory in the corresponding file for each file in intermediate_folder_strings, if it does not already exist
        """

        path = base_dir
        for string in intermediate_folder_strings:
            path += f"/{string}"
            self.create_folder_if_not_exists(path)
        return path
    
    def update_best_params_and_save(self, epoch, train_loss, dev_loss, trainer_params, model, optimizer, if_save_model_for_all_epochs = False):
        """
        Update best model parameters if it achieves best performance so far, and save the model
        """
        is_updated = False
        data_for_compare = {'train_loss': train_loss, 'dev_loss': dev_loss}
        
        # Check if either loss is nan
        train_loss_is_nan = torch.isnan(train_loss) if isinstance(train_loss, torch.Tensor) else math.isnan(train_loss)
        dev_loss_is_nan = torch.isnan(dev_loss) if isinstance(dev_loss, torch.Tensor) else math.isnan(dev_loss)
        if train_loss_is_nan or dev_loss_is_nan:
            return is_updated
            
        if data_for_compare[trainer_params['choose_best_model_on']] < self.best_performance_data[trainer_params['choose_best_model_on']]:  
            self.best_performance_data['train_loss'] = train_loss
            self.best_performance_data['dev_loss'] = dev_loss
            if model.trainable:
                self.best_performance_data['model_params_to_save'] = copy.deepcopy(model.state_dict())
            self.best_performance_data['update'] = True
            is_updated = True

        if trainer_params['save_model'] and model.trainable:
            if self.best_performance_data['last_epoch_saved'] + trainer_params['epochs_between_save'] <= epoch and self.best_performance_data['update']:
                self.best_performance_data['last_epoch_saved'] = epoch
                self.best_performance_data['update'] = False
                self.save_model(epoch, model, optimizer, trainer_params, if_save_model_for_all_epochs)
            elif if_save_model_for_all_epochs:
                self.save_model(epoch, model, optimizer, trainer_params, if_save_model_for_all_epochs)
        return is_updated
    
    def update_best_train_or_dev_loss(self, train_loss, dev_loss, trainer_params):
        is_updated = False

        if trainer_params['choose_best_model_on'] == 'train_loss':
            if self.best_train_loss > train_loss:
                self.best_train_loss = train_loss
                is_updated = True
        elif trainer_params['choose_best_model_on'] == 'dev_loss':
            if self.best_dev_loss > dev_loss:
                self.best_dev_loss = dev_loss
                is_updated = True
        return is_updated
    
    def plot_losses(self, ymin=None, ymax=None):
        """
        Plot train and test losses for each epoch
        """

        plt.legend()

        if ymin is not None and ymax is not None:
            plt.ylim(ymin, ymax)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
    
    def move_batch_to_device(self, data_batch):
        """
        Move a batch of data to the device (CPU or GPU)
        """

        return {k: v.to(self.device, non_blocking=True) for k, v in data_batch.items()}
    
    def load_model(self, model, optimizer, model_path):
        """
        Load a saved model
        """

        checkpoint = torch.load(model_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return model, optimizer
    
    def get_time_stamp(self):

        return int(datetime.datetime.now().timestamp())
    
    def get_year_month_day(self):
        """"
        Get current date in year_month_day format
        """

        ct = datetime.datetime.now()
        return f"{ct.year}_{ct.month:02d}_{ct.day:02d}"