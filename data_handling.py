from shared_imports import *
import scipy.stats as stats
import scipy.optimize as optimize
from lifelines import KaplanMeierFitter
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

class KMSampler:
    def __init__(self, num_samples, periods):
        self.kmf = None
        self.tail_rate = None
        self.max_uncensored = None
        self.num_samples = num_samples
        self.periods = periods
    
    def fit(self, observed_data, censoring_indicators, M, n_buckets=3):
        """Fit KM estimator and tail rate to data"""
        self.kmf = KaplanMeierFitter()
        self.kmf.fit(observed_data, censoring_indicators)
        
        # Get CDF values
        times = self.kmf.survival_function_.index
        cdf = 1 - self.kmf.survival_function_.values.flatten()
        
        # Get the last n_buckets points and their densities
        last_cdfs = cdf[-(n_buckets+1):]  # Get last 3 CDF values
        densities = np.diff(last_cdfs)     # Get P(X = k) for k in [M-2, M-1, M]
        p_greater_M = 1 - last_cdfs[-1]    # P(X > M)
        densities = np.append(densities, p_greater_M)
        
        # print(f"Debug - Last CDF values: {last_cdfs}")
        # print(f"Debug - Densities at last points: {densities}")
        # print(f"Debug - P(X > M): {p_greater_M}")
        
        def neg_log_likelihood(rate):
            if rate <= 0:
                return np.inf
            
            # Calculate model probabilities
            model_probs = np.array([
                np.exp(-rate * i) - np.exp(-rate * (i + 1))
                for i in range(n_buckets)
            ])
            # for last bucket, model prob is np.exp(-rate * n_buckets). need to add it to the last bucket
            model_probs = np.append(model_probs, np.exp(-rate * n_buckets))
            
            # Avoid log(0)
            valid_idx = (densities > 0) & (model_probs > 0)
            if not np.any(valid_idx):
                return np.inf
                
            return -np.sum(densities[valid_idx] * np.log(model_probs[valid_idx]))
        
        result = optimize.minimize_scalar(
            neg_log_likelihood, 
            bounds=(0.001, 2.0),
            method='bounded'
        )
        
        self.tail_rate = result.x
        self.threshold = times[-1]  # This is M
        # print(f'Estimated tail rate: {self.tail_rate}')
        
    def sample(self, n_samples):
        """Generate new samples from fitted distribution"""
        if self.kmf is None:
            raise ValueError("Must fit model before sampling")
            
        # Get survival function values
        times = np.arange(np.min(self.kmf.survival_function_.index), 
                        np.max(self.kmf.survival_function_.index) + 1)
        cdf = 1 - self.kmf.survival_function_at_times(times).values
        
        samples = np.zeros(n_samples)
        for i in range(n_samples):
            u = np.random.uniform(0, 1)
            
            # Find the smallest time where CDF is greater than u
            mask = cdf >= u
            if np.any(mask):  # Use KM distribution
                samples[i] = times[np.where(mask)[0][0]]
            else:
                # Sample from tail using threshold (M) instead of max_uncensored
                excess = np.random.exponential(1/self.tail_rate)
                samples[i] = self.threshold + np.floor(excess)

        
        return samples.astype(int)

    def fit_and_sample(self, n_fit, problem_params, demand_params, censoring_process, seed=None):
        """
        Run experiment fitting KM to n_fit samples and generating n_generate new ones
        """
        if seed is not None:
            np.random.seed(seed)
            
        # Generate training data
        true_demand = np.random.poisson(demand_params['mean'], size=n_fit)
        thresholds = np.random.poisson(censoring_process['mean'], size=n_fit)
        
        # Apply censoring
        observed = np.minimum(true_demand, thresholds)
        # Changed to use <= for correct censoring indicators
        censoring_indicators = (true_demand <= thresholds).astype(int)

        # # print the largest uncensored value (this is, only consider the values that are not censored)    
        # print(f"Largest uncensored value: {np.max(observed[censoring_indicators == 1])}")
        # print(f"Largest censored value: {np.max(observed[censoring_indicators == 0])}")

        # Fit sampler
        self.fit(observed, censoring_indicators, M=np.max(thresholds))
        
        n_generate = problem_params['n_stores'] * self.periods * self.num_samples

        # Generate new samples
        generated_samples = self.sample(n_generate)

        # reshape to self.num_samples, self.n_stores, self.periods
        generated_samples = generated_samples.reshape(self.num_samples, problem_params['n_stores'], self.periods)


        return generated_samples

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

        augment_multiplier = store_params['data_augment_multiplier_with_fixed_demands'] if 'data_augment_multiplier_with_fixed_demands' in store_params else 1
        self.demands = self.demands.repeat(augment_multiplier, 1, 1)
        self.num_samples = self.demands.size(0)

        self.store_random_yields = None
        if 'random_yield' in store_params:    
            self.store_random_yields = self.generate_demand_samples(problem_params, store_params, store_params['random_yield'], seeds)
        
        if problem_params.get('exp_underage_cost', False):
            self.underage_costs = self.generate_data_for_samples(store_params['underage_cost'], problem_params['n_stores'], seeds['underage_cost'], discrete=False)
            self.underage_costs = 10**self.underage_costs
        else:
            self.underage_costs = self.generate_data_for_samples(store_params['underage_cost'], problem_params['n_stores'], seeds['underage_cost'], discrete=False)

        if problem_params.get('holding_cost_is_ratio_of_underage_cost', False):
            self.holding_costs = self.underage_costs * 0.1
        else:
            self.holding_costs = self.generate_data_for_samples(store_params['holding_cost'], problem_params['n_stores'], seeds['holding_cost'], discrete=False)
        
        self.lead_times = self.generate_data_for_samples(store_params['lead_time'], problem_params['n_stores'], seeds['lead_time'], discrete=True).to(torch.int64)

        self.means = None
        if 'mean' in observation_params['include_static_features'] and observation_params['include_static_features']['mean']:
            self.means = torch.tensor(store_params['demand']['mean'])
            self.means = self.means.repeat(augment_multiplier, 1)
        self.stds = None
        if 'std' in observation_params['include_static_features'] and observation_params['include_static_features']['std']:
            self.stds = torch.tensor(store_params['demand']['std'])
            self.stds = self.stds.repeat(augment_multiplier, 1)
        self.store_random_yield_mean = None
        if 'store_random_yield_mean' in observation_params['include_static_features'] and observation_params['include_static_features']['store_random_yield_mean']:
            self.store_random_yield_mean = torch.tensor(store_params['random_yield']['mean'])
        self.store_random_yield_std = None
        if 'store_random_yield_std' in observation_params['include_static_features'] and observation_params['include_static_features']['store_random_yield_std']:
            self.store_random_yield_std = torch.tensor(store_params['random_yield']['std'])

        self.initial_inventories = self.generate_initial_inventories(store_params, self.demands, self.lead_times, problem_params['n_stores'], seeds['initial_inventory'])
        self.warehouse_lead_times = None
        self.initial_warehouse_inventories = None
        self.warehouse_holding_costs = None
        self.warehouse_store_edges = None
        self.warehouse_store_edge_lead_times = None
        self.warehouse_edge_initial_cost = None
        self.warehouse_edge_distance_cost = None
        if warehouse_params is not None:
            self.warehouse_lead_times = self.generate_data_for_samples(warehouse_params['lead_time'], problem_params['n_warehouses'], seeds['lead_time'], discrete=True)
            self.initial_warehouse_inventories = self.generate_initial_inventories(warehouse_params, self.demands, self.warehouse_lead_times, problem_params['n_warehouses'], seeds['initial_inventory'])
            self.warehouse_holding_costs = self.generate_data_for_samples(warehouse_params['holding_cost'], problem_params['n_warehouses'], seeds['holding_cost'])
            if 'edge_initial_cost' in warehouse_params:
                self.warehouse_edge_initial_cost = self.generate_data_for_samples(warehouse_params['edge_initial_cost'], problem_params['n_warehouses'], seeds['warehouse'])
            if 'edge_distance_cost' in warehouse_params:
                self.warehouse_edge_distance_cost = self.generate_data_for_samples(warehouse_params['edge_distance_cost'], problem_params['n_warehouses'], seeds['warehouse'])
            if 'edges' in warehouse_params:
                self.warehouse_store_edges = self.generate_warehouse_store_edges(warehouse_params['edges'], problem_params['n_warehouses'], problem_params['n_stores'], seeds['warehouse'])
            if 'edge_lead_times' in warehouse_params:
                self.warehouse_store_edge_lead_times = self.generate_warehouse_store_edge_lead_times(warehouse_params['edge_lead_times'], self.warehouse_store_edges, seeds['warehouse'])

        self.echelon_lead_times = None
        self.initial_echelon_inventories = None
        self.echelon_holding_costs = None
        if echelon_params is not None:
            self.echelon_lead_times = self.generate_data_for_samples(echelon_params['lead_time'], self.problem_params['n_extra_echelons'], seeds['lead_time'], discrete=True)
            self.initial_echelon_inventories = self.generate_initial_inventories(echelon_params, self.demands, self.echelon_lead_times, self.echelon_lead_times.size(1), seeds['initial_inventory'])
            self.echelon_holding_costs = self.generate_data_for_samples(echelon_params['holding_cost'], self.problem_params['n_extra_echelons'], seeds['holding_cost'])

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
        self.split_by = self.define_how_to_split_data()

    def generate_warehouse_store_edge_lead_times(self, edge_lead_times_params, warehouse_store_edges, seed):
        if 'value' in edge_lead_times_params:
            return torch.tensor(edge_lead_times_params['value']).expand(self.num_samples, -1, -1)
        np.random.seed(seed)
        if len(edge_lead_times_params['range']) != warehouse_store_edges.size(1):
            raise ValueError(f"Edge lead times range size {len(edge_lead_times_params['range'])} does not match warehouse store edges size {warehouse_store_edges.size(0)}")
        
        # First create tensor of sampled lead times for all possible edges
        sampled_lead_times = torch.zeros((self.num_samples, len(edge_lead_times_params['range']), warehouse_store_edges.size(-1)))
        for warehouse_idx, lead_time_range in enumerate(edge_lead_times_params['range']):
            if edge_lead_times_params['vary_across_samples']:
                # Sample different values for each sample and store
                sampled_lead_times[:, warehouse_idx] = torch.tensor(
                    np.random.randint(lead_time_range[0], lead_time_range[1], size=(self.num_samples, warehouse_store_edges.size(-1)))
                )
            else:
                # Sample once and repeat for all samples
                lead_times = torch.tensor(
                    np.random.randint(lead_time_range[0], lead_time_range[1], size=warehouse_store_edges.size(-1))
                )
                sampled_lead_times[:, warehouse_idx] = lead_times

        # Apply the edge mask to get final lead times
        edge_lead_times = sampled_lead_times * warehouse_store_edges
        return edge_lead_times

    def generate_warehouse_store_edges(self, edges_params, n_warehouses, n_stores, seed):
        np.random.seed(seed)
        params_copy = DefaultDict(lambda: False, copy.deepcopy(edges_params))

        edges = torch.tensor(params_copy['value'])
        if edges.size(0) != n_warehouses or edges.size(1) != n_stores:
            raise ValueError(f'Edges size {edges.size()} does not match n_warehouses {n_warehouses} and n_stores {n_stores}')
        if not torch.all((edges == 0) | (edges == 1)):
            raise ValueError('Edges must contain only 0 or 1 values')
        return edges.expand(self.num_samples, -1, -1)

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
                'warehouse_edge_initial_cost': self.warehouse_edge_initial_cost,
                'warehouse_edge_distance_cost': self.warehouse_edge_distance_cost,
                'initial_echelon_inventories': self.initial_echelon_inventories,
                'echelon_holding_costs': self.echelon_holding_costs,
                'echelon_lead_times': self.echelon_lead_times,
                'store_random_yield_mean': self.store_random_yield_mean,
                'store_random_yield_std': self.store_random_yield_std,
                'warehouse_store_edges': self.warehouse_store_edges,
                'warehouse_store_edge_lead_times': self.warehouse_store_edge_lead_times,
                }

        if self.store_random_yields is not None:
            data['store_random_yields'] = self.store_random_yields

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

        split_by = {'sample_index': ['underage_costs', 'holding_costs', 'lead_times', 'initial_inventories'], 
                    'period': []}

        if self.warehouse_lead_times is not None:
            split_by['sample_index'].append('initial_warehouse_inventories')
            split_by['sample_index'].append('warehouse_lead_times')
            split_by['sample_index'].append('warehouse_holding_costs')

        if self.echelon_params is not None:
            split_by['sample_index'].append('initial_echelon_inventories')
            split_by['sample_index'].append('echelon_lead_times')
            split_by['sample_index'].append('echelon_holding_costs')

        if self.warehouse_store_edges is not None:
            split_by['sample_index'].append('warehouse_store_edges')
            split_by['sample_index'].append('warehouse_store_edge_lead_times')

        if self.warehouse_edge_initial_cost is not None:
            split_by['sample_index'].append('warehouse_edge_initial_cost')
        if self.warehouse_edge_distance_cost is not None:
            split_by['sample_index'].append('warehouse_edge_distance_cost')

        if self.store_params['demand']['distribution'] == 'real':
            split_by['period'].append('demands')
            if self.store_random_yields is not None:
                split_by['period'].append('store_random_yields')
        else:
            split_by['sample_index'].append('demands')
            if self.store_random_yields is not None:
                split_by['sample_index'].append('store_random_yields')
        
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

        if store_params['underage_cost']['sample_across_instances'] and store_params['underage_cost']['vary_across_samples']:
            store_exponents = np.random.uniform(*store_params['underage_cost']['range'], size=(self.num_samples * problem_params['n_stores'])).reshape(self.num_samples, problem_params['n_stores'])
        elif store_params['underage_cost']['sample_across_instances']:
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
        
        if demand_params['sample_across_instances']:  # only supported for normal demand
            demand_params.update(self.sample_normal_mean_and_std(problem_params, demand_params, seeds))
    
    def generate_normal_demand(self, problem_params, demand_params, seed, is_test=False):
        """
        Generate normal demand data
        """

        # Set seed
        if seed is not None:
            np.random.seed(seed)
        
        if problem_params['n_stores'] == 1:
            if demand_params['sample_across_instances']:
                mean = demand_params['mean'][:, 0].reshape(-1, 1, 1)
                std = demand_params['std'][:, 0].reshape(-1, 1, 1)
                demand = np.random.normal(mean, std, size=(self.num_samples, 1, self.periods))
            else:
                demand = np.random.normal(demand_params['mean'], demand_params['std'], size=(self.num_samples, 1, self.periods))
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
            elif problem_params['censor_demands_for_train_and_dev'] == 'kaplanmeier':
                sampler = KMSampler(num_samples=self.num_samples, periods=self.periods)
                censoring_process = {'mean': 6}
                return sampler.fit_and_sample(problem_params['kaplanmeier_n_fit'], problem_params, demand_params, censoring_process, seed)
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
    
    def generate_data_for_samples(self, cost_params, n_instances, seed, discrete=False):
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
        if params_copy['sample_across_instances'] and params_copy['vary_across_samples']:
            return torch.tensor(this_sample_function(*params_copy['range'], self.num_samples * n_instances)).reshape(self.num_samples, n_instances)
        elif params_copy['sample_across_instances']:
            return torch.tensor(this_sample_function(*params_copy['range'], n_instances)).expand(self.num_samples, -1)
        elif params_copy['vary_across_samples']:
            return torch.tensor(this_sample_function(*params_copy['range'], self.num_samples)).unsqueeze(1).expand(-1, n_instances)
        elif params_copy['expand']:
            is_list = isinstance(params_copy['value'], list) and isinstance(params_copy['value'][0], list)
            if is_list:
                return torch.tensor([params_copy['value']]).expand(self.num_samples, n_instances, -1)
            else:
                return torch.tensor([params_copy['value']]).expand(self.num_samples, n_instances)
        else:
            return torch.tensor(params_copy['value'])
    
    def generate_initial_inventories(self, store_params, demands, lead_times, n_instances, seed):
        """
        Generate initial inventory data
        """
        # Set seed
        np.random.seed(seed)

        if store_params['initial_inventory']['sample']:
            demand_mean = demands.float().mean(dim=2).mean(dim=0)
            demand_mults = np.random.uniform(*store_params['initial_inventory']['range_mult'], 
                                             size=(self.num_samples, 
                                                   n_instances, 
                                                   max(store_params['initial_inventory']['inventory_periods'], lead_times.max()) 
                                                   )
                                            )
            return demand_mean[None, :, None] * demand_mults

        else:
            return torch.zeros(self.num_samples, 
                               n_instances, 
                               max(store_params['initial_inventory']['inventory_periods'], lead_times.max()))
    
    
    def generate_warehouse_data(self, warehouse_params, key):
        """
        Generate warehouse data
        For now, it is simply fixed across all samples
        """
        if warehouse_params is None:
            return None
        
        return torch.tensor([warehouse_params[key]]).expand(self.num_samples, self.problem_params['n_warehouses'])

    def generate_echelon_data(self, echelon_params, key):
        """
        Generate echelon_params data
        For now, it is simply fixed across all samples
        """
        if echelon_params is None:
            return None
        
        return torch.tensor(echelon_params[key]).unsqueeze(0).expand(self.num_samples, -1)

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

    def create_datasets(self, scenario, split=True, periods_for_split=None):
        if split:
            return [self.create_single_dataset(data) for data in self.split_by_period(scenario, periods_for_split)]
        else:
            return self.create_single_dataset(scenario.get_data())
    
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