from shared_imports import *
from data_handling import *
from neural_networks import *
import gymnasium as gym
from gymnasium import spaces

class Simulator(gym.Env):
    """
    Simulator class, defining a differentable simulator
    """

    metadata = {"render_modes": None}

    def __init__(self, recorder, device='cpu'):
 
        self.device = device
        self.recorder = recorder
        self.problem_params, self.observation_params, self.maximize_profit = None, None, None
        self.batch_size, self.n_stores, self.periods, self.observation, self._internal_data = None, None, None, None, None

        # Place holders. Will be overrided in reset method (as observation and action spaces depend on batch size, which might change during execution)
        self.action_space = spaces.Dict({'stores': spaces.Box(low=0.0, high=np.inf, shape=(1, 1), dtype=np.float32)})
        self.observation_space = spaces.Dict({'stores': spaces.Box(low=0.0, high=np.inf, shape=(1, 1), dtype=np.float32)})
    
    def reset(self, periods, problem_params,  data, observation_params):
        """
        Reset the environment, including initializing the observation, and return first observation

        Parameters
        ----------
        periods : int
            Number of periods in the simulation
        problem_params: dict
            Dictionary containing the problem parameters, specifying the number of warehouses, stores, whether demand is lost
            and whether to maximize profit or minimize underage+overage costs
        data : dict
            Dictionary containing the data for the simulation, including initial inventories, demands, holding costs and underage costs
        observation_params: dict
            Dictionary containing the observation parameters, specifying which features to include in the observations
        """

        self.problem_params = problem_params
        self.observation_params = observation_params

        self.batch_size, self.n_stores, self.periods = len(data['initial_inventories']), problem_params['n_stores'], periods
        

        # Data that can only be used by the simulator. E.g., all demands (including future)...
        self._internal_data = {
            'demands': data['demands'],
            'period_shift': observation_params['demand']['period_shift'],
            'store_random_yields': data['store_random_yields'] if 'store_random_yields' in data else None
            }

        if 'demand_signals' in data:
            self._internal_data['demand_signals'] = data['demand_signals']
        
        if observation_params['time_features'] is not None:
            self._internal_data.update({k: data[k] for k in observation_params['time_features']})
        if observation_params['sample_features'] is not None:
            self._internal_data.update({k: data[k] for k in observation_params['sample_features']})

        # We create "shift" tensors to calculate in which position of the long vector of the entire batch we have to add the corresponding allocation
        # This is necessary whenever the lead time is different for each sample or/and store
        self._internal_data['allocation_shift'] = self.initialize_shifts_for_allocation_put(data['initial_inventories'].shape).long()
        if 'warehouse_store_edge_lead_times' in data:
            self._internal_data['allocation_shift'] = self._internal_data['allocation_shift'].unsqueeze(-1).expand(-1, -1, data['warehouse_store_edge_lead_times'].shape[1])

        if self.problem_params['n_warehouses'] > 0:
            self._internal_data['warehouse_allocation_shift'] = self.initialize_shifts_for_allocation_put(data['initial_warehouse_inventories'].shape).long()
        
        if self.problem_params['n_extra_echelons'] > 0:
            self._internal_data['echelon_allocation_shift'] = self.initialize_shifts_for_allocation_put(data['initial_echelon_inventories'].shape).long()

        self.observation = self.initialize_observation(data, observation_params)
        # self.action_space = self.initialize_action_space(self.batch_size, problem_params, observation_params)
        # self.observation_space = self.initialize_observation_space(self.observation, periods, problem_params,)
        self.maximize_profit = problem_params['maximize_profit']
        
        return self.observation, None
    
    def initialize_shifts_for_allocation_put(self, shape):
        """
        We will add store's allocations into corresponding position by flatenning out the state vector of the
        entire batch. We create allocation_shifts to calculate in which position of that long vector we have
        to add the corresponding allocation
        """

        batch_size, n_stores, lead_time_max = shape

        # Results in a vector of lenght batch_size, where each entry corresponds to the first position of an element of a given sample
        # in the long vector of the entire batch
        n_instance_store_shift = (
            torch.arange(batch_size, device=self.device) * (lead_time_max * n_stores)
            )

        # Results in a tensor of shape batch_size x stores, where each entry corresponds to the number of positions to move 'to the right'
        # for each store, beginning from the first position within a sample
        store_n_shift = (
            torch.arange(n_stores, device=self.device) * (lead_time_max)
            ).expand(batch_size, n_stores)
        
        # Results in a vector of shape batch_size x stores, where each entry corresponds to the first position of an element of a given (sample, store)
        # in the long vector of the entire batch.
        # We then add the corresponding lead time to obtain the actual position in which to insert the action
        return n_instance_store_shift[:, None] + store_n_shift
    
    def step(self, action):
        """
        Simulate one step in the environment, returning the new observation and the reward (per sample)

        Parameters
        ----------
        action : dict
            Dictionary containing the actions to be taken in the environment for each type of location (stores and warehouses). 
            Each value is a tensor of size batch_size x n_locations, where n_locations is the number of stores or warehouses
        """

        data = {}
        current_demands = self.get_current_demands(self._internal_data)

        if self.recorder.is_recording:
            for i in range(current_demands.size(1)):
                data[f's_{i}_demand'] = current_demands.detach().cpu()[:, i]
                if 'mean' in self.observation:
                    data[f's_{i}_demand_mean'] = self.observation['mean'].detach().cpu()[:, i]
                if 'std' in self.observation:
                    data[f's_{i}_demand_std'] = self.observation['std'].detach().cpu()[:, i]
                if 'demand_signals' in self._internal_data:
                    data[f's_{i}_demand_signal'] = self._internal_data['demand_signals'].detach().cpu()[:, i, self.get_current_period()]
                data[f's_{i}_underage_costs'] = self.observation['underage_costs'].detach().cpu()[:, i]
                data[f's_{i}_holding_costs'] = self.observation['holding_costs'].detach().cpu()[:, i]
                if 'store_random_yields' in self._internal_data and self._internal_data['store_random_yields'] is not None:
                    data[f's_{i}_random_yields'] = self._internal_data['store_random_yields'].detach().cpu()[:, i, self.get_current_period()]
                for inv_loc in range(self.observation['store_inventories'].size(-1)):
                    data[f's_{i}_inventory_{inv_loc}'] = self.observation['store_inventories'].detach().cpu()[:, i, inv_loc]
                if self.problem_params['n_warehouses'] > 0:
                    for j in range(self.problem_params['n_warehouses']):
                        if action['stores'].dim() == 3:
                            data[f's_{i}_w_{j}_order'] = action['stores'].detach().cpu()[:, i, j]
                        else:
                            data[f's_{i}_w_{j}_order'] = action['stores'].detach().cpu()[:, i]

                # considers one warehouse
                if 'stores_intermediate_outputs' in action:
                    data[f's_{i}_stores_intermediate_outputs'] = action['stores_intermediate_outputs'].detach().cpu()[:, i]
            if self.problem_params['n_warehouses'] > 0:
                for i in range(self.problem_params['n_warehouses']):
                    for inv_loc in range(self.observation['warehouse_inventories'].size(-1)):
                        data[f'w_{i}_inventory_{inv_loc}'] = self.observation['warehouse_inventories'].detach().cpu()[:, i, inv_loc]
                    if 'warehouses' in action and action['warehouses'] is not None:
                        data[f'w_{i}_order'] = action['warehouses'].detach().cpu()[:, i]
                    if 'warehouse_loop_output' in action:
                        data[f's_{i}_warehouse_loop_output'] = action['warehouse_loop_output'].detach().cpu()[:, i]


        
        # Update observation corresponding to past occurrences (e.g., arrivals, orders, demands).
        # Do this before updating current period (as we consider current period + 1)
        self.update_past_data(action)

        # Update time related features (e.g., days to christmas)
        self.update_time_features(
            self._internal_data, 
            self.observation, 
            self.observation_params, 
            current_period=self.observation['current_period'].item() + 1
            )

        reward, s_underage_costs, s_holding_costs = self.calculate_store_reward_and_update_store_inventories(
            current_demands,
            action,
            self.observation,
            self.maximize_profit
            )

        # Calculate reward and update warehouse inventories
        if self.problem_params['n_warehouses'] > 0:

            w_reward, w_holding_costs, w_edge_costs = self.calculate_warehouse_reward_and_update_warehouse_inventories(
                action,
                self.observation,
                )
            reward += w_reward
        # Calculate reward and update other echelon inventories
        if self.problem_params['n_extra_echelons'] > 0:

            e_reward, e_reward_per_store = self.calculate_echelon_reward_and_update_echelon_inventories(
                action,
                self.observation,
                )
            reward += e_reward
        
        if self.recorder.is_recording:
            data['s_underage_costs'] = s_underage_costs.cpu().sum(dim=1)
            data['s_holding_costs'] = s_holding_costs.cpu().sum(dim=1)
            if self.problem_params['n_warehouses'] > 0:
                for i in range(w_holding_costs.size(1)):
                    data[f'w_{i}_holding_costs'] = w_holding_costs.detach().cpu()[:, i]
                    data[f'w_{i}_edge_costs'] = w_edge_costs.detach().cpu()[:, i].sum(dim=-1)
                
            if  self.problem_params['n_extra_echelons'] > 0:
                data['e1_holding_costs'] = e_reward_per_store[:, 0].cpu()
                data['e2_holding_costs'] = e_reward_per_store[:, 1].cpu()
            self.recorder.on_step(data)
        # Update current period
        self.observation['current_period'] += 1

        terminated = self.observation['current_period'] >= self.periods

        return self.observation, reward, terminated, None, None
    
    def get_current_period(self):
        return self.observation['current_period'].item() + self._internal_data['period_shift']

    def get_current_demands(self, data):
        """
        Get the current demands for the current period.
        period_shift specifies what we should consider as the first period in the data
        """
        
        return data['demands'][:, :, self.get_current_period()]
    
    def calculate_store_reward_and_update_store_inventories(self, current_demands, action, observation, maximize_profit=False):
        """
        Calculate reward and observation after demand and action is executed for stores
        """

        store_inventory = self.observation['store_inventories']
        inventory_on_hand = store_inventory[:, :, 0]
        
        post_inventory_on_hand = inventory_on_hand - current_demands

        # Reward given by -sales*price + holding_costs
        if maximize_profit:
            underage_costs = -observation['underage_costs'] * torch.minimum(inventory_on_hand, current_demands)
        # Reward given by underage_costs + holding_costs
        else:
            underage_costs = observation['underage_costs'] * torch.clip(-post_inventory_on_hand, min=0)

        base_holding_costs = observation['holding_costs'] * torch.clip(post_inventory_on_hand, min=0)
        holding_costs = base_holding_costs
        
        reward = underage_costs.sum(dim=1) + holding_costs.sum(dim=1)
        
        # If we are in a lost demand setting, we cannot have negative inventory
        if self.problem_params['lost_demand']:
            post_inventory_on_hand = torch.clip(post_inventory_on_hand, min=0)

        # Update store inventories based on the lead time corresponding to each sample and store.
        # If all lead times are the same, we can simplify this step by just adding the allocation to the last
        # column of the inventory tensor and moving all columns "to the left".
        # We leave this method as it is more general!
        lead_times = observation['lead_times']
        if 'warehouse_store_edge_lead_times' in self.observation:
            lead_times = self.observation['warehouse_store_edge_lead_times'].transpose(1, 2)
        observation['store_inventories'] = self.update_inventory_for_lead_times(
            store_inventory, 
            post_inventory_on_hand, 
            action['stores'], 
            lead_times, 
            self._internal_data['allocation_shift'],
            self._internal_data['store_random_yields']
            )
        
        # # Uncomment the following line to simplify the update of store inventories.
        # # Only works when all lead times are the same, and the lead time is larger than 1.
        # # Only tested for one-store setting.
        # observation['store_inventories'] = self.move_left_add_first_col_and_append(
        #     post_inventory_on_hand, 
        #     observation["store_inventories"], 
        #     int(observation["lead_times"][0, 0]), 
        #     action["stores"]
        #     )
        
        return reward, underage_costs, holding_costs
    
    def calculate_warehouse_reward_and_update_warehouse_inventories(self, action, observation):
        """
        Calculate reward and observation after action is executed for warehouses
        """

        warehouse_inventory = self.observation['warehouse_inventories']
        warehouse_inventory_on_hand = warehouse_inventory[:, :, 0]
        if 'warehouse_store_edge_lead_times' in self.observation:
            post_warehouse_inventory_on_hand = warehouse_inventory_on_hand - action['stores'].sum(dim=1)
        else:
            post_warehouse_inventory_on_hand = warehouse_inventory_on_hand - action['stores'].sum(dim=1).unsqueeze(1)

        holding_costs = observation['warehouse_holding_costs'] * torch.clip(post_warehouse_inventory_on_hand, min=0)
        reward = holding_costs.sum(dim=1) 

        edge_costs = torch.zeros_like(holding_costs)
        if 'warehouse_edge_initial_cost' in observation:
            initial_edge_cost = observation['warehouse_edge_initial_cost'].unsqueeze(-1) * action['stores'].transpose(1,2)
            edge_costs = initial_edge_cost
            if 'warehouse_edge_distance_cost' in observation:
                if 'warehouse_store_edge_lead_times' not in self.observation:
                    raise ValueError('Warehouse edge lead times not found but required for distance cost')
                distance_edge_cost = observation['warehouse_edge_distance_cost'].unsqueeze(-1) * observation['warehouse_store_edge_lead_times'] * action['stores'].transpose(1,2)
                edge_costs = edge_costs + distance_edge_cost
            reward += edge_costs.sum(dim=2).sum(dim=1)
        
        order_amount = action['warehouses']
        if 'warehouse_cluster_edges' in observation and 'mean' in observation and 'warehouse_demands_cap_factor' in observation:
            cluster_demands = torch.matmul(observation['warehouse_cluster_edges'], observation['mean'].unsqueeze(-1)).squeeze(-1)
            if observation['warehouse_demands_cap_factor'].ndim == 2:
                order_cap = cluster_demands * observation['warehouse_demands_cap_factor']
            elif observation['warehouse_demands_cap_factor'].ndim == 3:
                order_cap = cluster_demands * observation['warehouse_demands_cap_factor'][:, :, self.get_current_period()]
            else:
                raise ValueError('Warehouse demands cap factor has wrong number of dimensions')
            order_amount = torch.clip(order_amount, max=order_cap)
        
        if 'warehouse_demands_cap' in observation:
            if observation['warehouse_demands_cap'].ndim == 2:
                order_amount = torch.clip(order_amount, max=observation['warehouse_demands_cap'])
            elif observation['warehouse_demands_cap'].ndim == 3:
                order_amount = torch.clip(order_amount, max=observation['warehouse_demands_cap'][:, :, self.get_current_period()])
            else:
                raise ValueError('Warehouse demands cap has wrong number of dimensions')

        observation['warehouse_inventories'] = self.update_inventory_for_lead_times(
            warehouse_inventory, 
            post_warehouse_inventory_on_hand, 
            order_amount, 
            observation['warehouse_lead_times'], 
            self._internal_data['warehouse_allocation_shift'],
            None,
            )

        return reward, holding_costs, edge_costs
    def calculate_echelon_reward_and_update_echelon_inventories(self, action, observation):
        """
        Calculate reward and observation after action is executed for warehouses
        """

        echelon_inventories = self.observation['echelon_inventories']
        echelon_inventory_on_hand = echelon_inventories[:, :, 0]
        actions_to_subtract = torch.concat([action['echelons'][:, 1:], action['warehouses'].sum(dim=1).unsqueeze(1)], dim=1)
        post_echelon_inventory_on_hand = echelon_inventory_on_hand - actions_to_subtract

        reward = observation['echelon_holding_costs'] * torch.clip(post_echelon_inventory_on_hand, min=0)

        observation['echelon_inventories'] = self.update_inventory_for_lead_times(
            echelon_inventories, 
            post_echelon_inventory_on_hand, 
            action['echelons'], 
            observation['echelon_lead_times'], 
            self._internal_data['echelon_allocation_shift']
            )

        return reward.sum(dim=1), reward

    def initialize_observation(self, data, observation_params):
        """
        Initialize the observation of the environment
        """

        observation = {
            'store_inventories': data['initial_inventories'],
            'current_period': torch.tensor([0])
            }
        
        if observation_params['include_warehouse_inventory']:
            observation['warehouse_lead_times'] = data['warehouse_lead_times']
            observation['warehouse_holding_costs'] = data['warehouse_holding_costs']
            observation['warehouse_inventories'] = data['initial_warehouse_inventories']
            if 'warehouse_store_edges' in data:
                observation['warehouse_store_edges'] = data['warehouse_store_edges']
            if 'warehouse_store_edge_lead_times' in data:
                observation['warehouse_store_edge_lead_times'] = data['warehouse_store_edge_lead_times']
            if 'warehouse_edge_initial_cost' in data:
                observation['warehouse_edge_initial_cost'] = data['warehouse_edge_initial_cost']
            if 'warehouse_edge_distance_cost' in data:
                observation['warehouse_edge_distance_cost'] = data['warehouse_edge_distance_cost']
            if 'warehouse_cluster_edges' in data:
                observation['warehouse_cluster_edges'] = data['warehouse_cluster_edges']
            if 'warehouse_demands_cap_factor' in data:
                observation['warehouse_demands_cap_factor'] = data['warehouse_demands_cap_factor']
            if 'warehouse_demands_cap' in data:
                observation['warehouse_demands_cap'] = data['warehouse_demands_cap']
        
        if self.problem_params['n_extra_echelons'] > 0:
            observation['echelon_lead_times'] = data['echelon_lead_times']
            observation['echelon_holding_costs'] = data['echelon_holding_costs']
            observation['echelon_inventories'] = data['initial_echelon_inventories']

        # Include static features in observation (e.g., holding costs, underage costs, lead time and upper bounds)
        for k, v in observation_params['include_static_features'].items():
            if v:
                observation[k] = data[k]

        # Initialize data for past observations of certain data (e.g., arrivals, orders)
        for k, v in observation_params['include_past_observations'].items():
            if v > 0:
                if 'store_orders' == k:
                    if self.problem_params['n_warehouses'] <= 1:
                        observation[k] = torch.zeros(self.batch_size, self.n_stores, v, device=self.device)
                    else:
                        observation[k] = torch.zeros(self.batch_size, self.n_stores, self.problem_params['n_warehouses'], v, device=self.device)
                elif 'warehouse_orders' == k or 'warehouse_arrivals' == k or 'warehouse_self_loop_orders' == k:
                    observation[k] = torch.zeros(self.batch_size, self.problem_params['n_warehouses'], v, device=self.device)
                else:
                    observation[k] = torch.zeros(self.batch_size, self.n_stores, v, device=self.device)

        # Initialize past demands in the observation
        if observation_params['demand']['past_periods'] > 0:
            observation['past_demands'] = self.update_past_demands(data, observation_params, self.batch_size, self.n_stores, current_period=0)

        # Initialize time-related features, such as days to christmas
        if observation_params['time_features']:
            self.update_time_features(data, observation, observation_params, current_period=0)
        
        # Initialize sample-related features, such as the store number to which each sample belongs
        # This is just left as an example, and is not currently used by any policy
        self.create_sample_features(
            data, 
            observation, 
            observation_params
            )

        return observation
    
    def initialize_action_space(self, batch_size, problem_params, observation_params):
        """
        Initialize the action space by creating a dict with spaces.Box with shape batch_size x locations
        """

        d = {'stores': spaces.Box(low=0.0, high=np.inf, shape=(batch_size, problem_params['n_stores']), dtype=np.float32)}

        for k1, k2 in zip(['warehouses', 'extra_echelons'], ['n_warehouses', 'n_extra_echelons']):
            if problem_params[k2] > 0:
                d[k1] = spaces.Box(low=0.0, high=np.inf, shape=(batch_size, problem_params[k2]), dtype=np.float32)

        return spaces.Dict(d)
    
    def initialize_observation_space(self, initial_observation, periods, problem_params):
        
        # Initialize defaultdict to a dictionary with specific keys
        box_values = DefaultDict(lambda: {'low': -np.inf, 'high': np.inf, 'dtype': np.float32})
        box_values.update({
            'arrivals': {'low': 0 if problem_params['lost_demand'] else -np.inf, 'high': np.inf, 'dtype': np.float32},
            'holding_costs': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'lead_times': {'low': 0, 'high': 2*10, 'dtype': np.int8},
            'days_to_christmas': {'low': -365, 'high': 365, 'dtype': np.int8},
            'orders': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'past_demands': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'store_inventories': {'low': 0 if problem_params['lost_demand'] else -np.inf, 'high': np.inf, 'dtype': np.float32},
            'warehouse_inventories': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'warehouse_lead_times': {'low': 0, 'high': 2*10, 'dtype': np.int8},
            'extra_echelons_inventories': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'underage_costs': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'past_demands': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'past_demands': {'low': -np.inf, 'high': np.inf, 'dtype': np.float32},
            'warehouse_upper_bound': {'low': 0, 'high': np.inf, 'dtype': np.float32},
            'current_period': {'low': 0, 'high': periods, 'dtype': np.int32},
        })

        return spaces.Dict(
            {
            k: spaces.Box(
                low=box_values[k]['low'], 
                high=box_values[k]['high'], 
                shape=v.shape,
                dtype=box_values[k]['dtype']
                ) 
                for k, v in initial_observation.items()
                })
    
    def update_inventory_for_lead_times(self, inventory, inventory_on_hand, allocation, lead_times, allocation_shifter, store_random_yields = None):
        """
        Update the inventory for heterogeneous lead times (something simpler can be done for homogeneous lead times).
        We add the inventory into corresponding position by flatenning out the state vector of the
        entire batch. We created allocation_shifts earlier, which dictates the position shift of that long vector
        for each store and each sample. We then add the corresponding lead time to obtain the actual position in 
        which to insert the action
        """
        if store_random_yields is not None:
            random_yields = store_random_yields[torch.arange(self.batch_size), :, self.get_current_period()]
        else:
            random_yields = torch.ones_like(inventory_on_hand)

        if inventory.size()[2] == 1:
            return inventory_on_hand.unsqueeze(-1).to(dtype=allocation.dtype).put(
                        (allocation_shifter + lead_times.long() - 1).flatten(),  # Indexes where to 'put' allocation in long vector
                        torch.where(lead_times == 1, allocation * (random_yields.unsqueeze(-1) if allocation.ndim > random_yields.ndim else random_yields), allocation).flatten(),  # Apply random yields only when lead time is 1
                        accumulate=True  # True means adding to existing values, instead of replacing
                        )
        else:
            return torch.cat(
                [
                    (inventory_on_hand + inventory[:, :, 1] * random_yields).unsqueeze(-1),
                    inventory[:, :, 2:], 
                    torch.zeros_like(inventory[:, :, 1]).unsqueeze(-1)
                ],
                    dim=2
                    ).to(dtype=allocation.dtype).put(
                        (allocation_shifter + lead_times.long() - 1).flatten(),  # Indexes where to 'put' allocation in long vector
                        torch.where(lead_times == 1, allocation * (random_yields.unsqueeze(-1) if allocation.ndim > random_yields.ndim else random_yields), allocation).flatten(),  # Apply random yields only when lead time is 1
                        accumulate=True  # True means adding to existing values, instead of replacing
                        )

    def update_past_demands(self, data, observation_params, batch_size, stores, current_period):
        """
        Update the past demands in the observation
        """
        
        past_periods = observation_params['demand']['past_periods']
        current_period_shifted = current_period + self._internal_data['period_shift']
        
        if current_period_shifted == 0:
            past_demands = torch.zeros(batch_size, stores, past_periods, device=self.device)
        # If current_period_shifted < past_periods, we fill with zeros at the left
        else:
            past_demands = data['demands'][:, :, max(0, current_period_shifted - past_periods): current_period_shifted]

            fill_with_zeros = past_periods - (current_period_shifted - max(0, current_period_shifted - past_periods))
            if fill_with_zeros > 0:
                past_demands = torch.cat([
                    torch.zeros(batch_size, stores, fill_with_zeros, device=self.device), 
                    past_demands
                    ], 
                    dim=2)
        
        return past_demands
    
    def update_time_features(self, data, observation, observation_params, current_period):
        """
        Update all data that depends on time in the observation (e.g., days to christmas)
        """
        if observation_params['time_features'] is not None:
            for k in observation_params['time_features']:
                if data[k].shape[2] + 2 < current_period:
                    raise ValueError('Current period is greater than the number of periods in the data')
                observation[k] = data[k][:, :, min(current_period + observation_params['demand']['period_shift'], data[k].shape[2] - 1)]
    
    def create_sample_features(self, data, observation, observation_params):
        """
        Create features that only depend on the sample index (and not on the period) in the observation
        """

        if observation_params['sample_features'] is not None:
            for k in observation_params['sample_features']:
                observation[k] = data[k]
    
    def update_days_to_christmas(self, data, observation_params, current_period):
        """
        Update the days to christmas in the observation
        """
        days_to_christmas = data['days_to_christmas'][current_period + observation_params['demand']['period_shift']]
        
        return days_to_christmas

    def update_past_data(self, action):
        """
        Update the past data observations (e.g. last demands, arrivals and orders) in the observation
        """

        if self._internal_data['demands'].shape[2] + 2 < self.observation['current_period'].item():
            raise ValueError('Current period is greater than the number of periods in the data')
        if self.observation_params['demand']['past_periods'] > 0:
            self.observation['past_demands'] = self.update_past_demands(
                self._internal_data,
                self.observation_params,
                self.batch_size,
                self.n_stores,
                current_period=min(self.observation['current_period'].item() + 1, self._internal_data['demands'].shape[2])  # do this before updating current period!
                )
        
        if 'arrivals' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['arrivals'] > 0:
            self.observation['arrivals'] = self.move_left_and_append(self.observation['arrivals'], self.observation['store_inventories'][:, :, 1])
        
        if 'orders' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['orders'] > 0:
            self.observation['orders'] = self.move_left_and_append(self.observation['orders'], action['stores'])


        # below added for n warehouse setting
        if 'store_arrivals' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['store_arrivals'] > 0:
            self.observation['store_arrivals'] = self.move_left_and_append(self.observation['store_arrivals'], self.observation['store_inventories'][:, :, 1])

        if 'store_orders' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['store_orders'] > 0:
            if len(self.observation['store_orders'].shape) == 4:
                self.observation['store_orders'] = torch.cat([self.observation['store_orders'][:,:,:,1:], action['stores'].unsqueeze(-1)], dim=-1)
            else:  # 3 dimensions
                if len(action['stores'].shape) == 3:
                    self.observation['store_orders'] = torch.cat([self.observation['store_orders'][:,:,1:], action['stores']], dim=-1)
                else:
                    self.observation['store_orders'] = torch.cat([self.observation['store_orders'][:,:,1:], action['stores'].unsqueeze(-1)], dim=-1)
        
        if 'warehouse_arrivals' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['warehouse_arrivals'] > 0:
            self.observation['warehouse_arrivals'] = torch.cat([self.observation['warehouse_arrivals'][:,:,1:], self.observation['warehouse_inventories'][:,:,1].unsqueeze(-1)], dim=-1)

        if 'warehouse_orders' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['warehouse_orders'] > 0:
            self.observation['warehouse_orders'] = torch.cat([self.observation['warehouse_orders'][:,:,1:], action['warehouses'].unsqueeze(-1)], dim=-1)

        if 'warehouse_self_loop_orders' in self.observation_params['include_past_observations'] and self.observation_params['include_past_observations']['warehouse_self_loop_orders'] > 0:
            if 'warehouse_self_loop_orders' in action:
                self.observation['warehouse_self_loop_orders'] = torch.cat([self.observation['warehouse_self_loop_orders'][:,:,1:], action['warehouse_self_loop_orders'].unsqueeze(-1)], dim=-1)

    def move_columns_left(self, tensor_to_displace, start_index, end_index):
        """
        Move all columns in given array to the left, and return as list
        """

        return [tensor_to_displace[:, :, i + 1] for i in range(start_index, end_index)]
    
    def move_left_and_append(self, tensor_to_displace, tensor_to_append, start_index=0, end_index=None, dim=2):
        """
        Move all columns in given array to the left, and append a new tensor at the end
        """
        
        if end_index is None:
            end_index = tensor_to_displace.shape[dim] - 1
        
        return torch.stack([*self.move_columns_left(tensor_to_displace, start_index, end_index), tensor_to_append],
                           dim=dim)

    def move_left_add_first_col_and_append(self, post_inventory_on_hand, inventory, lead_time, action):
        """
        Move columns of inventory (deleting first column, as post_inventory_on_hand accounts for inventory_on_hand after demand arrives)
          to the left, add inventory_on_hand to first column, and append action at the end
        """

        return torch.stack([
            post_inventory_on_hand + inventory[:, :, 1], 
            *self.move_columns_left(inventory, 1, lead_time - 1), 
            action
            ], dim=2)