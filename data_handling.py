from shared_imports import *
import scipy.stats as stats
import scipy.optimize as optimize
import warnings

class WeibullDemandGenerator:
   def __init__(self, num_samples, periods):
       self.num_samples = num_samples
       self.periods = periods

   def estimate_weibull_scale(self, samples, M, n_tail_buckets=3, fixed_lambda=None):
       if fixed_lambda is not None:
           return fixed_lambda
           
       flat_samples = samples.flatten()
       tail_samples = flat_samples[(flat_samples >= M-n_tail_buckets) & (flat_samples <= M)]
       n_at_threshold = np.sum(flat_samples == M)
       
       def neg_log_likelihood(scale):
           eps = 1e-10
           scale = max(scale, eps)
           
           uncensored = tail_samples[tail_samples < M]
           pdf_term = np.sum(np.log(eps + stats.weibull_min.pdf(
               uncensored - (M-n_tail_buckets), 
               c=self.k,
               scale=scale
           )))
           
           if n_at_threshold > 0:
               sf = stats.weibull_min.sf(n_tail_buckets - 1, c=self.k, scale=scale)
               threshold_term = n_at_threshold * np.log(eps + sf)
           else:
               threshold_term = 0
               
           return -(pdf_term + threshold_term)
       
       result = optimize.minimize_scalar(
           neg_log_likelihood, 
           bounds=(1e-6, 20), 
           method='bounded',
           options={'xatol': 1e-8}
       )

       if not result.success:
           warnings.warn(f"Scale estimation did not converge: {result.message}")
           
       return result.x

   def fit_and_sample(self, problem_params, demand_params, seed=None):
       if seed is not None:
           np.random.seed(seed)
       
       M = problem_params.get('censoring_threshold', 7)
       fixed_lambda = problem_params.get('weibull_fixed_lambda', None)
       self.k = problem_params.get('weibull_k', 2.0)
       n_tail_buckets = problem_params.get('n_tail_buckets', 3)
       
       orig_samples = np.random.poisson(
           demand_params['mean'], 
           size=(self.num_samples, problem_params['n_stores'], self.periods)
       )
       
       imputed_samples = np.minimum(orig_samples, M)
       censored_mask = orig_samples >= M
       
       weibull_scale = self.estimate_weibull_scale(orig_samples, M, n_tail_buckets, fixed_lambda)
       
       n_censored = np.sum(censored_mask)
       lower_bound = stats.weibull_min.cdf(n_tail_buckets, c=self.k, scale=weibull_scale)
       uniform_samples = np.random.uniform(lower_bound, 1-1e-10, size=n_censored)
       
       censored_tails = np.floor(
           M + stats.weibull_min.ppf(uniform_samples, c=self.k, scale=weibull_scale) - n_tail_buckets
       ).astype(int)
       
       imputed_samples[censored_mask] = censored_tails
       
       return imputed_samples

class Scenario():
    '''
    Class to generate an instance. 
    First samples parameters (e.g, mean demand and std for each store, costs, lead times, etc...) if there are parameters to be sampled.
    Then, creates demand traces, and the initial values (e.g., of inventory) to be used.
    '''
    def __init__(self, periods, problem_params, store_params, warehouse_params, echelon_params, num_samples, observation_params, seeds=None, is_test=False):

        self.problem_params = problem_params
        self.store_params = store_params
        self.warehouse_params = warehouse_params
        self.echelon_params = echelon_params
        self.num_samples = num_samples
        self.periods = periods
        self.observation_params = observation_params
        self.seeds = seeds
        self.demands = self.generate_demand_samples(problem_params, store_params, store_params['demand'], seeds, is_test)
        self.store_random_yields = None
        if 'random_yield' in store_params:
            if 'lost_order_average_interval' in store_params['random_yield']:
                self.store_random_yields = self.generate_lost_yield_mask(store_params['random_yield'], self.demands, seeds['demand'])
            else:
                self.store_random_yields = self.generate_demand_samples(problem_params, store_params, store_params['random_yield'], seeds)
        
        if problem_params.get('exp_underage_cost', False):
            store_params['underage_cost']['range'][0] = max(np.log10(warehouse_params['holding_cost']), store_params['underage_cost']['range'][0])
            self.underage_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['underage_cost'], seeds['underage_cost'], discrete=False)
            self.underage_costs = 10**self.underage_costs
        else:
            self.underage_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['underage_cost'], seeds['underage_cost'], discrete=False)
            
        if problem_params.get('holding_cost_is_ratio_of_underage_cost', False):
            self.holding_costs = self.underage_costs * 0.1
        else:
            self.holding_costs = self.generate_data_for_samples_and_stores(problem_params, store_params['holding_cost'], seeds['holding_cost'], discrete=False)
        
        self.lead_times = self.generate_data_for_samples_and_stores(problem_params, store_params['lead_time'], seeds['lead_time'], discrete=True).to(torch.int64)
        self.means, self.stds, self.store_random_yield_mean, self.store_random_yield_std = self.generate_means_and_stds(observation_params, store_params)
        self.initial_inventories = self.generate_initial_inventories(problem_params, store_params, self.demands, self.lead_times, seeds['initial_inventory'])
        
        self.initial_warehouse_inventories = self.generate_initial_warehouse_inventory(warehouse_params)
        self.warehouse_lead_times = self.generate_warehouse_data(warehouse_params, 'lead_time')
        self.warehouse_holding_costs = self.generate_warehouse_data(warehouse_params, 'holding_cost')
        self.lost_order_mask = self.generate_lost_order_mask(warehouse_params, self.demands, seeds['demand'])

        self.initial_echelon_inventories = self.generate_initial_echelon_inventory(echelon_params)
        self.echelon_lead_times = self.generate_echelon_data(echelon_params, 'lead_time')
        self.echelon_holding_costs = self.generate_echelon_data(echelon_params, 'holding_cost')

        time_and_sample_features = {'time_features': {}, 'sample_features': {}}

        for feature_type, feature_file in zip(['time_features', 'sample_features'], ['time_features_file', 'sample_features_file']):
            if observation_params[feature_type] and observation_params[feature_file]:
                features = pd.read_csv(observation_params[feature_file])
                for k in observation_params[feature_type]:
                    tensor_to_append = torch.tensor(features[k].values)
                    if feature_type == 'time_features':
                        time_and_sample_features[feature_type][k] = tensor_to_append.unsqueeze(0).unsqueeze(0).expand(self.num_samples, self.problem_params['n_stores'], -1)
                    elif feature_type == 'sample_features':  # Currently only supports the one store case
                        time_and_sample_features[feature_type][k] = tensor_to_append.unsqueeze(1).expand(-1, self.problem_params['n_stores'])
            
        self.time_features = time_and_sample_features['time_features']
        self.sample_features = time_and_sample_features['sample_features']

        # Creates a dictionary specifying which data has to be split by sample index and which by period (when dividing into train, dev, test sets)
        self.split_by = self.define_how_to_split_data()

    def generate_lost_yield_mask(self, random_yield_params, demands, seed):
        """
        Generate lost yield mask based on lost_order_average_interval
        Returns a tensor of 1s and 0s where 0s appear on average every lost_order_average_interval periods
        """
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        interval = random_yield_params['lost_order_average_interval']
        mask = torch.ones(self.num_samples, self.problem_params['n_stores'], demands.size(2))
        
        # For each period, set 0 with probability 1/interval
        random_values = torch.rand(self.num_samples, self.problem_params['n_stores'], demands.size(2))
        mask[random_values < (1.0/interval)] = 0
        
        return mask

    def get_data(self):
        """
        Return the generated data. Will be part of a Dataset
        """

        data =  {'demands': self.demands,
                'underage_costs': self.underage_costs,
                'holding_costs': self.holding_costs,
                'lead_times': self.lead_times,
                'mean': self.means,
                'std': self.stds,
                'initial_inventories': self.initial_inventories,
                'initial_warehouse_inventories': self.initial_warehouse_inventories,
                'warehouse_lead_times': self.warehouse_lead_times,
                'warehouse_holding_costs': self.warehouse_holding_costs,
                'initial_echelon_inventories': self.initial_echelon_inventories,
                'echelon_holding_costs': self.echelon_holding_costs,
                'echelon_lead_times': self.echelon_lead_times,
                'store_random_yields': self.store_random_yields,
                'store_random_yield_mean': self.store_random_yield_mean,
                'store_random_yield_std': self.store_random_yield_std,
                }

        if self.lost_order_mask is not None:
            data['lost_order_mask'] = self.lost_order_mask
        
        for k, v in self.time_features.items():
            data[k] = v
        
        for k, v in self.sample_features.items():
            data[k] = v
        
        return {k: v.float() for k, v in data.items() if v is not None}
    
    def define_how_to_split_data(self):
        """
        Define how to split the data into different samples
        If demand comes from real data, the training and dev sets correspond to different periods.
        However, if it is generated, the split is according to sample indexes
        """

        split_by = {'sample_index': ['underage_costs', 'holding_costs', 'lead_times', 'initial_inventories', 'initial_warehouse_inventories'\
                                    , 'warehouse_lead_times', 'warehouse_holding_costs', 'store_random_yields'], 
                    'period': []}

        if self.store_params['demand']['distribution'] == 'real':
            split_by['period'].append('demands')
            if self.lost_order_mask is not None:
                split_by['period'].append('lost_order_mask')
        else:
            split_by['sample_index'].append('demands')
            if self.lost_order_mask is not None:
                split_by['sample_index'].append('lost_order_mask')
        
        for k in self.time_features.keys():
            split_by['period'].append(k)

        for k in self.sample_features.keys():
            split_by['sample_index'].append(k)
        
        return split_by
    
    def generate_demand_samples(self, problem_params, store_params, demand_params, seeds, is_test=False):
        """
        Generate demand data
        """
                
        # Sample parameters to generate demand if necessary (otherwise, does nothing)
        self.generate_demand_parameters(problem_params, demand_params, seeds)

        demand_generator_functions = {
            "normal": self.generate_normal_demand, 
            'poisson': self.generate_poisson_demand,
            'real': self.read_real_demand_data,
            }

        # Changing demand seed for consistency with results prensented in the manuscript
        self.adjust_seeds_for_consistency(problem_params, store_params, demand_params, seeds)

        # Sample demand traces
        demand = demand_generator_functions[demand_params['distribution']](problem_params, demand_params, seeds['demand'], is_test)

        if demand_params['clip']:  # Truncate at 0 from below if specified
            demand = np.clip(demand, 0, demand_params.get('clip_max', None))
        
        return torch.tensor(demand)

    def generate_costs_for_exponential_underage_costs(self, problem_params, store_params, seed):
        np.random.seed(seed)

        if store_params['underage_cost']['sample_across_stores'] and store_params['underage_cost']['vary_across_samples']:
            store_exponents = np.random.uniform(*store_params['underage_cost']['range'], size=(self.num_samples * problem_params['n_stores'])).reshape(self.num_samples, problem_params['n_stores'])
        elif store_params['underage_cost']['sample_across_stores']:
            store_exponents = np.random.uniform(*store_params['underage_cost']['range'], size=problem_params['n_stores']).reshape(1, -1).repeat(self.num_samples, axis=0)
        elif store_params['underage_cost']['vary_across_samples']:
            store_exponents = np.random.uniform(*store_params['underage_cost']['range'], size=self.num_samples).reshape(-1, 1).repeat(problem_params['n_stores'], axis=1)
        elif store_params['underage_cost']['expand']:
            store_exponents = np.array(store_params['underage_cost']['value']).reshape(1, 1).repeat(self.num_samples, axis=0).repeat(problem_params['n_stores'], axis=1)
        else:
            store_exponents = np.random.uniform(*store_params['underage_cost']['range'], size=1).repeat(self.num_samples * problem_params['n_stores']).reshape(self.num_samples, problem_params['n_stores'])
        store_underage_costs = np.power(10, store_exponents)
        store_holding_costs = 0.1 * store_underage_costs
        warehouse_holding_cost = 0.3 * store_holding_costs.sum(axis=1, keepdims=True)
        return (
            torch.tensor(store_underage_costs),
            torch.tensor(store_holding_costs), 
            torch.tensor(warehouse_holding_cost)
        )

    def adjust_seeds_for_consistency(self, problem_params, store_params, demand_params, seeds):
        """
        Adjust seeds for consistency with results prensented in the manuscript
        """

        if problem_params['n_warehouses'] == 0 and problem_params['n_stores'] == 1 and demand_params['distribution'] != 'real':
            try:
                # Changing demand seed for consistency with results prensented in the manuscript
                seeds['demand'] = seeds['demand'] + int(store_params['lead_time']['value'] + 10*store_params['underage_cost']['value'])
            except Exception as e:
                print(f'Error: {e}')
    
    def read_real_demand_data(self, problem_params, demand_params, seed, is_test=False):
        """
        Read real demand data
        """

        demand = torch.load(demand_params['file_location'])[: self.num_samples, :problem_params['n_stores']]
        return demand

    def generate_demand_parameters(self, problem_params, demand_params, seeds):
        """
        Sample parameters of demand distribution, if necessary
        """
        
        if demand_params['sample_across_stores']:  # only supported for normal demand
            demand_params.update(self.sample_normal_mean_and_std(problem_params, demand_params, seeds))
    
    def generate_normal_demand(self, problem_params, demand_params, seed, is_test=False):
        """
        Generate normal demand data
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        if problem_params['n_stores'] == 1:
            mean = demand_params['mean'][:, 0].reshape(-1, 1, 1)
            std = demand_params['std'][:, 0].reshape(-1, 1, 1)
            demand = np.random.normal(mean, std, size=(self.num_samples, 1, self.periods))
        else:
            # Calculate covariance matrix and sample from multivariate normal for all samples at once
            correlation = demand_params['correlation']
            n_stores = problem_params['n_stores']
            
            # Create block diagonal covariance matrix for all samples
            # Shape: (num_samples, n_stores, n_stores)
            cov_matrices = np.zeros((self.num_samples, n_stores, n_stores))
            for i in range(n_stores):
                for j in range(n_stores):
                    if i == j:
                        cov_matrices[:, i, j] = demand_params['std'][:, i] * demand_params['std'][:, i]
                    else:
                        cov_matrices[:, i, j] = correlation * demand_params['std'][:, i] * demand_params['std'][:, j]
            
            # Generate correlated normal samples for all samples at once
            demand = np.array([np.random.multivariate_normal(m, cov, size=self.periods) 
                             for m, cov in zip(demand_params['mean'], cov_matrices)])
            
            # Transpose to get shape (num_samples, n_stores, periods)
            demand = np.transpose(demand, (0, 2, 1))

        return demand

    def generate_poisson_demand(self, problem_params, demand_params, seed, is_test=False):
        if seed is not None:
            np.random.seed(seed)
        
        if is_test == False and 'censor_demands_for_train_and_dev' in problem_params and problem_params['censor_demands_for_train_and_dev'] != None:
            if problem_params['censor_demands_for_train_and_dev'] == 'weibull':
                demand_generator = WeibullDemandGenerator(self.num_samples, self.periods)
                return demand_generator.fit_and_sample(problem_params, demand_params, seed)
            else:
                raise Exception('Censoring method not supported')
        
        return np.random.poisson(demand_params['mean'], size=(self.num_samples, problem_params['n_stores'], self.periods))

    def generate_data(self, demand_params, **kwargs):
        """
        Generate demand data
        """
        demand_generator_functions = {"normal": self.generate_normal_demand_for_one_store}
        demand = demand_generator_functions[demand_params['distribution']](demand_params, **kwargs)
        
        if demand_params['clip']:
            demand = np.clip(demand, 0, None)

        return torch.tensor(demand)
        
    def sample_normal_mean_and_std(self, problem_params, demand_params, seeds):
        """
        Sample mean and std for normal demand
        """

        # Set seed
        np.random.seed(seeds['mean'])

        if demand_params.get('vary_across_samples', False):
            means = np.random.uniform(demand_params['mean_range'][0], 
                                    demand_params['mean_range'][1], 
                                    (self.num_samples, problem_params['n_stores'])).round(3)
            sample_shape = (self.num_samples, problem_params['n_stores'])
        else:
            means = np.random.uniform(demand_params['mean_range'][0],
                                    demand_params['mean_range'][1],
                                    problem_params['n_stores']).round(3)
            means = np.tile(means, (self.num_samples, 1))
            sample_shape = problem_params['n_stores']

        np.random.seed(seeds['coef_of_var'])
        if 'coef_of_var_range' in demand_params:
            coef = np.random.uniform(demand_params['coef_of_var_range'][0],
                                   demand_params['coef_of_var_range'][1],
                                   sample_shape)
            stds = (means * coef).round(3)
        else:
            stds = np.random.uniform(demand_params['coef_of_std_range'][0],
                                   demand_params['coef_of_std_range'][1],
                                   sample_shape).round(3)
            if demand_params.get('vary_across_samples', False) == False:
                stds = np.tile(stds, (self.num_samples, 1))
        return {'mean': means, 'std': stds}
    
    def generate_data_for_samples_and_stores(self, problem_params, cost_params, seed, discrete=False):
        """
        Generate cost or lead time data, for each sample and store
        """
        
        np.random.seed(seed)

        # We first create a default dict from the params dictionary, so that we return False by default
        # whenever we query a key that was not set by the user
        params_copy = DefaultDict(lambda: False, copy.deepcopy(cost_params))

        sample_functions = {False: np.random.uniform, True: np.random.randint}
        this_sample_function = sample_functions[discrete]
        

        if params_copy['file_location']:
            params_copy['value'] = torch.load(params_copy['file_location'])[: self.num_samples]
        if params_copy['sample_across_stores'] and params_copy['vary_across_samples']:
            return torch.tensor(this_sample_function(*params_copy['range'], self.num_samples * problem_params['n_stores'])).reshape(self.num_samples, problem_params['n_stores'])
        elif params_copy['sample_across_stores']:
            return torch.tensor(this_sample_function(*params_copy['range'], problem_params['n_stores'])).expand(self.num_samples, -1)
        elif params_copy['vary_across_samples']:
            return torch.tensor(this_sample_function(*params_copy['range'], self.num_samples)).unsqueeze(1).expand(-1, problem_params['n_stores'])
        elif params_copy['expand']:
            return torch.tensor(params_copy['value']).expand(self.num_samples, problem_params['n_stores'])
        else:
            return torch.tensor(params_copy['value'])
    
    def generate_initial_inventories(self, problem_params, store_params, demands, lead_times, seed):
        """
        Generate initial inventory data
        """
        # Set seed
        np.random.seed(seed)

        if store_params['initial_inventory']['sample']:
            demand_mean = demands.float().mean(dim=2).mean(dim=0)
            demand_mults = np.random.uniform(*store_params['initial_inventory']['range_mult'], 
                                             size=(self.num_samples, 
                                                   problem_params['n_stores'], 
                                                   max(store_params['initial_inventory']['inventory_periods'], lead_times.max()) 
                                                   )
                                            )
            return demand_mean[None, :, None] * demand_mults

        else:
            return torch.zeros(self.num_samples, 
                               problem_params['n_stores'], 
                               store_params['initial_inventory']['inventory_periods'])
    
    def generate_initial_warehouse_inventory(self, warehouse_params):
        """
        Generate initial warehouse inventory data
        """
        if warehouse_params is None:
            return None
        
        return torch.zeros(self.num_samples, 
                           1, 
                           warehouse_params['lead_time']
                           )
    
    def generate_initial_echelon_inventory(self, echelon_params):
        """
        Generate initial echelon inventory data
        """
        if echelon_params is None:
            return None
        
        return torch.zeros(self.num_samples, 
                           len(echelon_params['lead_time']), 
                           max(echelon_params['lead_time'])
                           )
    
    def generate_warehouse_data(self, warehouse_params, key):
        """
        Generate warehouse data
        For now, it is simply fixed across all samples
        """
        if warehouse_params is None:
            return None
        
        return torch.tensor([warehouse_params[key]]).expand(self.num_samples, self.problem_params['n_warehouses'])
    
    def generate_lost_order_mask(self, warehouse_params, demands, seed):
        """
        Generate lost order mask based on lost_order_average_interval
        """
        if warehouse_params is None or 'lost_order_average_interval' not in warehouse_params or warehouse_params['lost_order_average_interval'] is None:
            return None
        
        # Set seed for reproducibility
        torch.manual_seed(seed)
        
        interval = warehouse_params['lost_order_average_interval']
        mask = torch.zeros(self.num_samples, self.problem_params['n_warehouses'], demands.size(2), dtype=torch.bool)
        
        # For each period, set True with probability 1/interval
        random_values = torch.rand(self.num_samples, self.problem_params['n_warehouses'], demands.size(2))
        mask[random_values < (1.0/interval)] = True
        
        return mask

    def generate_echelon_data(self, echelon_params, key):
        """
        Generate echelon_params data
        For now, it is simply fixed across all samples
        """
        if echelon_params is None:
            return None
        
        return torch.tensor(echelon_params[key]).unsqueeze(0).expand(self.num_samples, -1)
    
    def generate_means_and_stds(self, observation_params, store_params):
        """
        Create tensors with store demand's means and stds.
        Will be used as inputs for the symmetry-aware NN.
        """

        to_return = {'mean': None, 'std': None, 'store_random_yield_mean': None, 'store_random_yield_std': None}
        for k in ['mean', 'std']:
            if k in observation_params['include_static_features'] and observation_params['include_static_features'][k]:
                to_return[k] = torch.tensor(store_params['demand'][k])# .unsqueeze(0).expand(self.num_samples, -1)
        
        if 'store_random_yield_mean' in observation_params['include_static_features'] and observation_params['include_static_features']['store_random_yield_mean']:
            to_return['store_random_yield_mean'] = torch.tensor(store_params['random_yield']['mean'])
        if 'store_random_yield_std' in observation_params['include_static_features'] and observation_params['include_static_features']['store_random_yield_std']:
            to_return['store_random_yield_std'] = torch.tensor(store_params['random_yield']['std'])
        return to_return['mean'], to_return['std'], to_return['store_random_yield_mean'], to_return['store_random_yield_std']

class MyDataset(Dataset):

    def __init__(self, num_samples, data):
        self.data = data
        self.num_samples = num_samples
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.data.items()}
    

class DatasetCreator():

    def __init__(self):

        pass

    def create_datasets(self, scenario, split=True, by_period=False, by_sample_indexes=False, periods_for_split=None, sample_index_for_split=None):

        if split:
            if by_period:
                return [self.create_single_dataset(data) for data in self.split_by_period(scenario, periods_for_split)]
            elif by_sample_indexes:
                train_data, dev_data = self.split_by_sample_index(scenario, sample_index_for_split)
            else:
                raise NotImplementedError
            return self.create_single_dataset(train_data), self.create_single_dataset(dev_data)
        else:
            return self.create_single_dataset(scenario.get_data())
    
    def split_by_sample_index(self, scenario, sample_index_for_split):
        """
        Split dataset into dev and train sets by sample index
        We consider the first entries to correspomd to the dev set (so that size of train set does not impact it)
        This should be used when demand is synthetic (otherwise, if demand is real, there would be data leakage)
        """

        data = scenario.get_data()

        dev_data = {k: v[:sample_index_for_split] for k, v in data.items()}
        train_data = {k: v[sample_index_for_split:] for k, v in data.items()}

        return train_data, dev_data
    
    def split_by_period(self, scenario, periods_for_split):

        data = scenario.get_data()
        common_data = {k: data[k] for k in scenario.split_by['sample_index']}
        out_datasets = []

        for period_range in periods_for_split:
            this_data = copy.deepcopy(common_data)
            # Change period_range to slice object (it is currently of type string)
            period_range = slice(*map(int, period_range.strip('() ').split(',')))

            for k in scenario.split_by['period']:
                this_data[k] = data[k][:, :, period_range]
            out_datasets.append(this_data)
        
        return out_datasets

    
    def create_single_dataset(self, data):
        """
        Create a single dataset
        """

        num_samples = len(data['initial_inventories'])

        return MyDataset(num_samples, data)