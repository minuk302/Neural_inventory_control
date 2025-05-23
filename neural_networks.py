from copy import deepcopy
from shared_imports import *
from quantile_forecaster import FullyConnectedForecaster
import gc
import wandb
import torch.nn.functional as F
from torch.nn import functional as init

class custom_lazy_linear(nn.LazyLinear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def reset_parameters(self):
        if not self.has_uninitialized_params() and self.in_features != 0:
            nn.init.orthogonal_(self.weight)
            if self.bias is not None:
                fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                nn.init.uniform_(self.bias, -bound, bound)


class MyNeuralNetwork(nn.Module):

    def __init__(self, args, problem_params, device='cpu'):
        """"
        Initialize neural network with given parameters

        Parameters
        ----------
        args: dictionary
            Dictionary with the following
            - inner_layer_activations: dictionary with the activation function for each neural net module (master, stores, warehouse, context net)
            - output_layer_activation: dictionary with the activation function for the output layer of each neural net module
            - neurons_per_hidden_layer: dictionary with the number of neurons for each hidden layer of each neural net module
            - output_sizes: dictionary with the output size for each neural net module
            - initial_bias: dictionary with the initial bias for each neural net module
        device: str
            Device where the neural network will be stored
        """

        super().__init__() # initialize super class
        self.device = device

        # Some models are not trainable (e.g. news-vendor policies), so we need to flag it to the trainer
        # so it does not perform greadient steps (otherwise, it will raise an error)
        self.trainable = True
        
        # Define activation functions, which will be called in forward method
        self.activation_functions = {
            'relu': nn.ReLU(), 
            'elu': nn.ELU(), 
            'tanh': nn.Tanh(),
            'softmax': nn.Softmax(dim=1),
            'softplus': nn.Softplus(),
            'sigmoid': nn.Sigmoid(),
            'leaky_relu': nn.LeakyReLU(),
            }
        
        # If warehouse_upper_bound is not None, then we will use it to multiply the output of the warehouse neural network
        self.warehouse_upper_bound_mult = None

        self.layer_normalization = 'layer_normalization' in args and args['layer_normalization']
        self.gradient_clipping_norm_value = None
        if 'gradient_clipping_norm_value' in args:
            self.gradient_clipping_norm_value = args['gradient_clipping_norm_value']

        self.orthogonal_initialization = 'orthogonal_initialization' in args and args['orthogonal_initialization']

        self.layers = {}
        # Create nn.ModuleDict to store multiple neural networks
        self.net = self.create_module_dict(args)

        # Initialize bias if given
        if args['initial_bias'] is not None:
            for key, val in args['initial_bias'].items():
                if val is not None:
                    # Position of last linear layer depends on whether there is an output layer activation function
                    pos = -2 if args['output_layer_activation'][key] else -1
                    self.initialize_bias(key, pos, val)
        
        self.is_debugging = 'is_debugging' in args and args['is_debugging']
        self.debug_identifier = args['debug_identifier'] if 'debug_identifier' in args else None
    
    def forward(self, observation):
        raise NotImplementedError
    
    def create_module_dict(self, args):
        """
        Create a dictionary of neural networks, where each key is a neural network module (e.g. master, store, warehouse, context)
        """
        
        return nn.ModuleDict({key: 
                              self.create_sequential_net(
                                  key,
                                  args['inner_layer_activations'][key], 
                                  args['output_layer_activation'][key], 
                                  args['neurons_per_hidden_layer'][key], 
                                  args['output_sizes'][key]
                                  ) 
                                  for key in args['output_sizes']
                                  }
                                  )
    
    def create_sequential_net(self, name, inner_layer_activations, output_layer_activation, neurons_per_hidden_layer, output_size):
        """
        Create a neural network with the given parameters
        """

        # Define layers
        layers = []
        linear_class = nn.LazyLinear
        if self.orthogonal_initialization:
            linear_class = custom_lazy_linear
        
        # Add hidden layers
        for i, output_neurons in enumerate(neurons_per_hidden_layer):
            layers.append(linear_class(output_neurons))
            if self.layer_normalization:
                layers.append(nn.LayerNorm(output_neurons))
            layers.append(self.activation_functions[inner_layer_activations])

        if len(neurons_per_hidden_layer) == 0:
            layers.append(linear_class(output_size))
        else:
            layers.append(nn.Linear(neurons_per_hidden_layer[-1], output_size))

        if self.layer_normalization:
            layers.append(nn.LayerNorm(output_size))
        
        # If output_layer_activation is not None, then we add the activation function to the last layer
        if output_layer_activation is not None:
            layers.append(self.activation_functions[output_layer_activation])
        
        self.layers[name] = layers

        # Define network as a sequence of layers
        return nn.Sequential(*layers)

    def initialize_bias(self, key, pos, value):
        self.layers[key][pos].bias.data.fill_(value)
    
    def apply_proportional_allocation(self, store_intermediate_outputs, warehouse_inventories, transshipment = False, soft_min = False):
        """
        Apply proportional allocation feasibility enforcement function to store intermediate outputs.
        It assigns inventory proportionally to the store order quantities, whenever inventory at the
        warehouse is not sufficient.
        """

        total_limiting_inventory = warehouse_inventories[:, 0, 0]  # Total inventory at the warehouse
        sum_allocation = store_intermediate_outputs.sum(dim=1)  # Sum of all store order quantities
        
        ratio = total_limiting_inventory / (sum_allocation + torch.finfo(sum_allocation.dtype).eps)
        if transshipment == False:
            if soft_min:
                # Use log-sum-exp approximation to min(ratio, 1)
                epsilon = 1e-1  # Smoothing parameter
                one = torch.ones_like(ratio)

                min_approx = -epsilon * torch.logsumexp(
                    torch.stack([-ratio, -one], dim=0) / epsilon,
                    dim=0
                )
                result = torch.multiply(store_intermediate_outputs, min_approx[:, None])
            else:
                result = torch.multiply(store_intermediate_outputs, torch.clip(ratio, max=1)[:, None])
        else:
            result = torch.multiply(store_intermediate_outputs, ratio[:, None])
        return result
    
    def apply_softmax_feasibility_function(self, store_intermediate_outputs, warehouse_inventory, transshipment=False):
        """
        Apply softmax across store intermediate outputs, and multiply by warehouse inventory on-hand
        If transshipment is False, then we add a column of ones to the softmax inputs, to allow for inventory to be held at the warehouse
        """

        total_warehouse_inv = warehouse_inventory[:, :, 0].sum(dim=1)  # warehouse's inventory on-hand
        softmax_inputs = store_intermediate_outputs

        # If warehouse can hold inventory, then concatenate a tensor of ones to the softmax inputs
        if not transshipment:
            softmax_inputs = torch.cat((
                softmax_inputs, 
                torch.ones_like(softmax_inputs[:, 0], device=self.device)[:, None]
                ), 
                dim=1
                )
        softmax_outputs = self.activation_functions['softmax'](softmax_inputs)

        # If warehouse can hold inventory, then remove last column of softmax outputs
        if not transshipment:
            softmax_outputs = softmax_outputs[:, :-1]

        result = torch.multiply(
            softmax_outputs, 
            total_warehouse_inv[:, None]
            )
        return result

    def flatten_then_concatenate_tensors(self, tensor_list, dim=1):
        """
        Flatten tensors in tensor_list, and concatenate them along dimension dim
        """

        return torch.cat([
            tensor.flatten(start_dim=dim) for tensor in tensor_list
            ], 
            dim=dim)
    
    def concatenate_signal_to_object_state_tensor(self, object_state, signal):
        """
        Concatenate signal (e.g. context vector) to every location's local state (e.g. store inventories or warehouse inventories).
        Signal is tipically of shape (num_samples, signal_dim) and object_state is of shape (num_samples, n_objects, object_state_dim),
        and results in a tensor of shape (num_samples, n_objects, object_state_dim + signal_dim)
        """

        n_objects = object_state.size(1)
        signal = signal.unsqueeze(1).expand(-1, n_objects, -1)
        try:
            return torch.cat((object_state, signal), dim=2)
        except RuntimeError as e:
            if 'out of memory' in str(e):
                print("CUDA out of memory. Clearing cache and trying again...")
                gc.collect()
                torch.cuda.empty_cache()
                return torch.cat((object_state, signal), dim=2)
            else:
                raise e
    
    def unpack_args(self, args, keys):
        """
        Unpacks arguments from a dictionary
        """
        return [args[key] for key in keys] if len(keys) > 1 else args[keys[0]]

class VanillaOneStore(MyNeuralNetwork):
    """
    Fully connected neural network for settings with one store and no warehouse
    """
    
    def forward(self, observation):
        """
        Uses store inventories as input and directly outputs store orders
        """
        x = observation['store_inventories']

        # Flatten input, except for the batch dimension
        x = x.flatten(start_dim=1)

        # Pass through network
        # NN architecture has to be such that output is non-negative
        x = self.net['master'](x) + 1
        x = self.activation_functions['softplus'](x)

        return {'stores': x}

class VanillaOneStoreForWarehouse(MyNeuralNetwork):
    def forward(self, observation):
        store_params = torch.stack([observation[k] for k in ['mean', 'std', 'underage_costs', 'lead_times']], dim=2)
        x = torch.cat([observation['store_inventories'], store_params], dim=2)
        x = x.flatten(start_dim=1)
        x = self.net['master'](x)
        return {'stores': x}

class BaseStock(MyNeuralNetwork):
    """
    Base stock policy
    """

    def forward(self, observation):
        """
        Get a base-level, which is the same across all stores and sample
        Calculate allocation as max(base_level - inventory_position, 0)
        """
        x = observation['store_inventories']
        inv_pos = x.sum(dim=2)
        x = self.net['master'](torch.tensor([0.0]).to(self.device))  # Constant base stock level
        return {'stores': torch.clip(x - inv_pos, min=0)} # Clip output to be non-negative

class BaseStockDistribution(MyNeuralNetwork):
    def forward(self, observation):
        self.trainable = False
        # Get required parameters
        x = observation['store_inventories']  # shape: (batch, 1, 1)
        inv_pos = x.sum(dim=2)  # (batch, 1)

        underage_costs = observation['underage_costs']  # (batch, 1)
        holding_costs = observation['holding_costs']    # (batch, 1)
        lead_times = observation['lead_times']          # (batch, 1)

        mean = 5.0  # (batch, 1)
        std = 1.6    # (batch, 1)
        L = lead_times              # (batch, 1)
        L_plus_1 = L + 1

        demand_mean = mean * L_plus_1
        demand_std = std * torch.sqrt(L_plus_1.float())

        p = underage_costs
        h = holding_costs
        critical_ratio = p / (p + h)

        # Use the inverse CDF (ppf) of the normal distribution
        from scipy.stats import norm
        demand_mean_np = demand_mean.detach().cpu().numpy()
        demand_std_np = demand_std.detach().cpu().numpy()
        critical_ratio_np = critical_ratio.detach().cpu().numpy()
        S_star_np = norm.ppf(critical_ratio_np, loc=demand_mean_np, scale=demand_std_np)
        S_star = torch.tensor(S_star_np, dtype=inv_pos.dtype, device=inv_pos.device)

        order = torch.clamp(S_star - inv_pos, min=0.0)

        return {'stores': order}
        
class EchelonStock(MyNeuralNetwork):
    """
    Echelon stock policy
    """

    def forward(self, observation):
        """
        Get a base-level for each location, which is the same across all samples.
        We obtain base-levels via partial sums, which allowed us to avoid getting "stuck" in bad local minima.
        Calculate allocation as max(base_level - inventory_position, 0) truncated above by previous location's inventory/
        Contrary to how we define other policies, we will follow and indexing where locations are ordered from upstream to downstream (last is store).
        """
        store_inventories, warehouse_inventories, echelon_inventories = self.unpack_args(
            observation, ['store_inventories', 'warehouse_inventories', 'echelon_inventories'])
        n_extra_echelons = echelon_inventories.size(1)
        
        x = self.activation_functions['softplus'](self.net['master_echelon'](torch.tensor([0.0]).to(self.device)) + 10.0)  # Constant base stock levels
        # The base level for each location wil be calculated as the outputs corresponding to all downstream locations and its own
        base_levels = torch.cumsum(x, dim=0).flip(dims=[0])

        # Inventory position (NOT echelon inventory position) for each location
        stacked_inv_pos = torch.concat((echelon_inventories.sum(dim=2), warehouse_inventories.sum(dim=2), store_inventories.sum(dim=2)), dim=1)
        
        # Tensor with the inventory on hand for the preceding location for each location k
        # For the left-most location, we set it to a large number, so that it does not truncate the allocation
        shifted_inv_on_hand = torch.concat((
            1000000*torch.ones_like(warehouse_inventories[:, :, 0]), 
            echelon_inventories[:, :, 0], 
            warehouse_inventories[:, :, 0]), 
            dim=1
            )

        # print(f'base_levels: {base_levels}')
        # print(f'stacked_inv_pos: {stacked_inv_pos[0]}')
        # print(f'echelon_pos: {torch.stack([(stacked_inv_pos[:, k:].sum(dim=1)) for k in range(2 + n_extra_echelons)], dim=1)[0]}')

        # Allocations before truncating by previous locations inventory on hand.
        # We get them by subtracting the echelon inventory position (i.e., sum of inventory positions from k onwards) from the base levels, 
        # and truncating below by 0.
        tentative_allocations = torch.clip(
            torch.stack([base_levels[k] - (stacked_inv_pos[:, k:].sum(dim=1)) 
                         for k in range(2 + n_extra_echelons)], 
                         dim=1), 
                         min=0)
        
        # print(f'tentative_allocations: {tentative_allocations[0]}')
        # Truncate below by previous locations inventory on hand
        allocations = torch.minimum(tentative_allocations, shifted_inv_on_hand)

        # print(f'shifted_inv_on_hand: {shifted_inv_on_hand[0]}')
        # print(f'allocations: {allocations[0]}')

        # print(f'stacked_inv_on_hand.shape: {shifted_inv_on_hand.shape}')
        # print()

        return {
            'stores': allocations[:, -1:],
            'warehouses': allocations[:, -2: -1],
            'echelons': allocations[:, : n_extra_echelons],
                } 

class CappedBaseStock(MyNeuralNetwork):
    """"
    Simlar to BaseStock, but with a cap on the order
    """

    def forward(self, observation):
        """
        Get a base-level and cap, which is the same across all stores and sample
        Calculate allocation as min(base_level - inventory_position, cap) and truncated below from 0
        """
        x = observation['store_inventories']
        inv_pos = x.sum(dim=2)
        x = self.net['master'](torch.tensor([0.0]).to(self.device))  # Constant base stock level
        base_level, cap = x[0], x[1]  # We interpret first input as base level, and second output as cap on the order
        
        return {'stores': torch.clip(base_level - inv_pos, min=torch.tensor([0.0]).to(self.device), max=cap)} # Clip output to be non-negative


class VanillaSerial(MyNeuralNetwork):
    """
    Vanilla NN for serial system
    """

    def forward(self, observation):
        """
        We apply a sigmoid to the output of the master neural network, and multiply by the inventory on hand for the preceding location,
        except for the left-most location, where we multiply by an upper bound on orders.
        """
        store_inventories, warehouse_inventories, echelon_inventories = self.unpack_args(
            observation, ['store_inventories', 'warehouse_inventories', 'echelon_inventories'])
        n_extra_echelons = echelon_inventories.size(1)
        
        input_tensor = self.flatten_then_concatenate_tensors([store_inventories, warehouse_inventories, echelon_inventories])
        x = self.net['master_echelon'](torch.tensor(input_tensor).to(self.device))  # Constant base stock levels
        # print(f'self.warehouse_upper_bound: {self.warehouse_upper_bound}')
        # assert False

        # Tensor with the inventory on hand for the preceding location for each location k
        # For the left-most location, we set it to an upper bound (same as warehouse upper bound). Currently 4 times mean demand.
        shifted_inv_on_hand = torch.concat((
            #self.warehouse_upper_bound.unsqueeze(1).expand(echelon_inventories.shape[0], -1),
            5.0 * self.warehouse_upper_bound_mult * torch.ones(echelon_inventories.size(0), 1, device=self.device),
            echelon_inventories[:, :, 0], 
            warehouse_inventories[:, :, 0]), 
            dim=1
            )
        # print(f'x: {x.shape}')
        # print(f'shifted_inv_on_hand: {shifted_inv_on_hand.shape}')
        # print(f"self.activation_functions['sigmoid'](x): {self.activation_functions['sigmoid'](x)[0]}")
        allocations = self.activation_functions['sigmoid'](x)*shifted_inv_on_hand
        # print(f'allocations[0]: {allocations[0]}')


        return {
            'stores': allocations[:, -1:],
            'warehouses': allocations[:, -2: -1],
            'echelons': allocations[:, : n_extra_echelons],
                }


class VanillaSerialSelfloop(MyNeuralNetwork):
    def forward(self, observation):
        store_inventories, warehouse_inventories, echelon_inventories = self.unpack_args(
            observation, ['store_inventories', 'warehouse_inventories', 'echelon_inventories'])
        n_extra_echelons = echelon_inventories.size(1)
        
        input_tensor = self.flatten_then_concatenate_tensors([store_inventories, warehouse_inventories, echelon_inventories])
        outputs = self.net['master_echelon_selfloop'](torch.tensor(input_tensor).to(self.device))
        loop_outputs = outputs[:, 2 + n_extra_echelons:]
        non_loop_outputs = outputs[:, :2 + n_extra_echelons]

        echelon_allocations = []
        for j in range(n_extra_echelons):
            if j == 0:  # First echelon
                echelon_allocations.append(non_loop_outputs[:, j:j+1])
            else:
                echelon_allocations.append(self.apply_proportional_allocation(torch.cat([non_loop_outputs[:, j:j+1], loop_outputs[:, j-1:j]], dim=1), echelon_inventories[:,j-1:j])[:, :-1])
        warehouse_allocation = self.apply_proportional_allocation(torch.cat([non_loop_outputs[:, -2: -1], loop_outputs[:, -2:-1]], dim=1), echelon_inventories[:,-1:, :])[:, :-1]
        store_allocation = self.apply_proportional_allocation(torch.cat([non_loop_outputs[:, -1:], loop_outputs[:, -1:]], dim=1), warehouse_inventories)[:, :-1]
        return {
            'stores': store_allocation,
            'warehouses': warehouse_allocation,
            'echelons': torch.cat(echelon_allocations, dim=1),
                }

class CBS_One_Warehouse(MyNeuralNetwork):
    def forward(self, observation):
        store_inventories, warehouse_inventories = self.unpack_args(
        observation, ['store_inventories', 'warehouse_inventories'])
        x = self.net['master_cbs'](torch.tensor([0.0]).to(self.device))
        

        base_levels = x[:1 + store_inventories.size(1)]  # Warehouse + store base levels
        base_levels = self.activation_functions['softplus'](base_levels + 10.0)
        store_caps = self.activation_functions['softplus'](x[1 + store_inventories.size(1):] + 10.0)  # Store caps
        
        warehouse_base_level = base_levels.sum()

        store_inv = store_inventories.sum(dim=2)
        store_base_levels = base_levels[1:].expand(store_inv.shape[0], -1)
        store_caps = store_caps.expand(store_inv.shape[0], -1)
        
        # Apply both base level and cap constraints
        store_intermediate_outputs = torch.clip(
            torch.min(
                torch.clip(store_base_levels - store_inv, min=0),
                store_caps
            ),
            min=0
        )

        warehouse_pos = warehouse_inventories.sum(dim=2).sum(dim=1) + store_inv.sum(dim=1)
        warehouse_allocation = torch.clip(warehouse_base_level.repeat(store_inv.shape[0]) - warehouse_pos, min=0)
        store_allocation = self.apply_proportional_allocation(store_intermediate_outputs, warehouse_inventories)
        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation.unsqueeze(-1)
            }

class VanillaOneWarehouse(MyNeuralNetwork):
    """
    Fully connected neural network for settings with one warehouse (or transshipment center) and many stores
    """

    def forward(self, observation):
        """
        Use store and warehouse inventories and output intermediate outputs for stores and warehouses.
        For stores, apply softmax to intermediate outputs (concatenated with a 1 when inventory can be held at the warehouse)
          and multiply by warehouse inventory on-hand
        For warehouses, apply sigmoid to intermediate outputs and multiply by warehouse upper bound.
        """
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        n_stores = store_inventories.size(1)
        input_tensor = torch.cat((store_inventories.flatten(start_dim=1), warehouse_inventories.flatten(start_dim=1)), dim=1)
        intermediate_outputs = self.net['master'](input_tensor)
        store_intermediate_outputs, warehouse_intermediate_outputs = intermediate_outputs[:, :n_stores], intermediate_outputs[:, n_stores:]

        # Apply softmax to store intermediate outputs
        # If class name is not VanillaTransshipment, then we will add a column of ones to the softmax inputs (so that inventory can be held at warehouse)
        store_allocation = \
            self.apply_softmax_feasibility_function(
                store_intermediate_outputs, 
                warehouse_inventories,
                transshipment=(self.__class__.__name__ == 'VanillaTransshipment')
                )
        
        # Apply sigmoid to warehouse intermediate outputs and multiply by warehouse upper bound
        if self.warehouse_upper_bound_mult is not None:
            upper_bound = observation['mean'].sum(dim=1, keepdim=True) * self.warehouse_upper_bound_mult
            warehouse_allocation = self.activation_functions['sigmoid'](warehouse_intermediate_outputs)*upper_bound

        # if self.is_debugging:
        #     debug_dir = "/user/ml4723/Prj/NIC/debug"
        #     if self.debug_identifier is not None:
        #         debug_dir = debug_dir + self.debug_identifier
        #     os.makedirs(debug_dir, exist_ok=True)
        #     for sample_idx in range(min(store_allocation.size(0), 32)):
        #         with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
        #             f.write("\n\n")
        #             warehouse_outputs = warehouse_allocation[sample_idx].detach().cpu().to(torch.float32).numpy()
        #             f.write(np.array2string(warehouse_outputs, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
        #             f.write('\n')
                    
        #             max_lead_time = int(observation['warehouse_lead_times'][sample_idx].max().item())
        #             for lead_time in range(max_lead_time, 0, -1):
        #                 outstanding_orders = observation['warehouse_inventories'][sample_idx, :, lead_time - 1].detach().cpu().to(torch.float32).numpy()
        #                 f.write(np.array2string(outstanding_orders, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
        #                 f.write('\n')

        #             for store_idx in range(n_stores):
        #                 store_alloc = store_allocation[sample_idx, store_idx].detach().cpu().to(torch.float32).numpy()
        #                 store_out = store_intermediate_outputs[sample_idx, store_idx].detach().cpu().to(torch.float32).numpy()
        #                 formatted_allocation_orders = '[' + ', \t'.join([f'{store_alloc:4.1f}/{store_out:4.1f}']) + ']'
        #                 f.write(formatted_allocation_orders)
                        
        #                 store_inv = observation['store_inventories'][sample_idx, store_idx].detach().cpu().to(torch.float32).numpy()
        #                 formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv[:-1]])
        #                 f.write(f"[{formatted_store_inv}]\n")

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }


class VanillaOneWarehouseSelfloop(MyNeuralNetwork):
    def forward(self, observation):
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        n_stores = store_inventories.size(1)
        input_tensor = torch.cat((store_inventories.flatten(start_dim=1), warehouse_inventories.flatten(start_dim=1)), dim=1)
        
        if self.__class__.__name__ == 'VanillaTransshipmentSelfloop':
            intermediate_outputs = self.net['master'](input_tensor)
            store_intermediate_outputs, warehouse_allocation = intermediate_outputs[:, :n_stores], intermediate_outputs[:, n_stores:]
            store_allocation = self.apply_proportional_allocation(
                store_intermediate_outputs, 
                warehouse_inventories,
                transshipment=True
                )
        else:
            intermediate_outputs = self.net['master_selfloop'](input_tensor)
            store_intermediate_outputs, warehouse_allocation, self_loop_outputs = intermediate_outputs[:, :n_stores], intermediate_outputs[:, n_stores:-1], intermediate_outputs[:, -1:]
            allocations = self.apply_proportional_allocation(
                torch.cat([store_intermediate_outputs, self_loop_outputs], dim=1), 
                warehouse_inventories)
            store_allocation = allocations[:, :-1]
        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class VanillaTransshipmentSelfloop(VanillaOneWarehouseSelfloop):
    pass

class Vanilla_N_Stores(MyNeuralNetwork):
    def forward(self, observation):
        # Flatten input, except for the batch dimension
        x = observation['store_inventories'].flatten(start_dim=1)

        # Pass through network
        # NN architecture has to be such that output is non-negative
        x = self.net['master'](x) + 1
        x = self.activation_functions['softplus'](x)
        return {
            'stores': x, 
            }

class N_Stores_Shared_Net(MyNeuralNetwork):
    def forward(self, observation):
        x = observation['store_inventories']
        x = self.net['master'](x).squeeze(-1) + 1
        x = self.activation_functions['softplus'](x)
        return {
            'stores': x, 
            }
    
class N_Stores_Per_Store_Net(MyNeuralNetwork):
    def forward(self, observation):
        # Flatten input, except for the batch dimension
        x = observation['store_inventories']

        # Pass through network
        # NN architecture has to be such that output is non-negative
        outputs = []
        for store_idx in range(x.size(1)):
            store_output = self.net[f'master_{store_idx}'](x[:,store_idx:store_idx+1]) + 1
            outputs.append(store_output)
        x = torch.cat(outputs, dim=1).squeeze(-1)
        x = self.activation_functions['softplus'](x)

        if self.is_debugging:
            debug_dir = "/user/ml4723/Prj/NIC/debug"
            if self.debug_identifier is not None:
                debug_dir = debug_dir + self.debug_identifier
            os.makedirs(debug_dir, exist_ok=True)
            for sample_idx in range(min(x.size(0), 32)):
                with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
                    f.write("\n\n")
                    # Write store orders and inventories side by side
                    orders = x[sample_idx].detach().cpu().numpy()
                    for store_idx in range(x.size(1)):
                        store_inv = observation['store_inventories'][sample_idx, store_idx].detach().cpu().numpy()
                        formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv])
                        f.write(f"{store_idx:2d}: [{orders[store_idx]:9.1f} | {formatted_store_inv}]\n")

        return {
            'stores': x, 
            }

class Vanilla_N_Warehouses_Selfloop(MyNeuralNetwork):
    def forward(self, observation):
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        n_stores = store_inventories.size(1)
        n_warehouses = warehouse_inventories.size(1)
        input_tensor = torch.cat((store_inventories.flatten(start_dim=1), warehouse_inventories.flatten(start_dim=1)), dim=1)
        outputs = self.net['master_n_warehouses_selfloop'](input_tensor)
        intermediate_outputs = outputs[:, :-n_warehouses]
        self_loop_outputs = outputs[:, -n_warehouses:]
        warehouse_allocation = intermediate_outputs[:, :n_warehouses]
        edge_mask = observation['warehouse_store_edges'].transpose(1,2)
        store_intermediate_outputs = intermediate_outputs[:, n_warehouses:].reshape(intermediate_outputs.size(0), n_stores, n_warehouses)
        
        store_allocation = []
        for warehouse_idx in range(warehouse_inventories.size(1)):
            connected_stores_mask = edge_mask[:, :, warehouse_idx]
            connected_store_outputs = store_intermediate_outputs[:, :, warehouse_idx] 
            connected_store_outputs = connected_store_outputs * connected_stores_mask
            connected_store_outputs = connected_store_outputs.masked_fill(~connected_stores_mask.bool(), float('-inf'))
            allocations = self.apply_proportional_allocation(
                torch.cat([connected_store_outputs, self_loop_outputs[:, warehouse_idx:warehouse_idx+1]], dim=1), 
                warehouse_inventories[:,warehouse_idx:warehouse_idx+1])
            store_allocation.append(allocations[:, :-1])
        store_allocation = torch.stack(store_allocation, dim=2)

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class Vanilla_N_Warehouses(MyNeuralNetwork):
    def forward(self, observation):
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        n_stores = store_inventories.size(1)
        input_tensor = torch.cat((store_inventories.flatten(start_dim=1), warehouse_inventories.flatten(start_dim=1)), dim=1)
        intermediate_outputs = self.net['master_n_warehouses'](input_tensor)
        n_warehouses = warehouse_inventories.size(1)
        warehouse_intermediate_outputs = intermediate_outputs[:, :n_warehouses]
        edge_mask = observation['warehouse_store_edges'].transpose(1,2)
        store_intermediate_outputs = intermediate_outputs[:, n_warehouses:].reshape(intermediate_outputs.size(0), n_stores, n_warehouses)
        
        store_allocation = []
        for warehouse_idx in range(warehouse_inventories.size(1)):
            connected_stores_mask = edge_mask[:, :, warehouse_idx]
            connected_store_outputs = store_intermediate_outputs[:, :, warehouse_idx] 
            connected_store_outputs = connected_store_outputs * connected_stores_mask
            connected_store_outputs = connected_store_outputs.masked_fill(~connected_stores_mask.bool(), float('-inf'))
            warehouse_allocation = self.apply_softmax_feasibility_function(
                connected_store_outputs,
                warehouse_inventories[:,warehouse_idx:warehouse_idx+1],
                transshipment=False
                )
            store_allocation.append(warehouse_allocation)
        store_allocation = torch.stack(store_allocation, dim=2)
        
        if self.warehouse_upper_bound_mult is not None:
            upper_bound = observation['mean'].sum(dim=1, keepdim=True) * self.warehouse_upper_bound_mult
            warehouse_allocation = self.activation_functions['sigmoid'](warehouse_intermediate_outputs)*upper_bound

        if self.is_debugging:
            debug_dir = "/user/ml4723/Prj/NIC/debug"
            if self.debug_identifier is not None:
                debug_dir = debug_dir + self.debug_identifier
            os.makedirs(debug_dir, exist_ok=True)
            for sample_idx in range(min(intermediate_outputs.size(0), 32)):
                with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
                    f.write("\n\n")
                    outputs_array = warehouse_allocation[sample_idx].detach().cpu().numpy()
                    f.write(np.array2string(outputs_array, formatter={'float_kind':lambda x: f"{x:11.1f}\t"}))
                    f.write('\n')
                    
                    max_lead_time = int(observation['warehouse_lead_times'][sample_idx].max().item())
                    for lead_time in range(max_lead_time, 0, -1):
                        outstanding_orders = observation['warehouse_inventories'][sample_idx, :, lead_time - 1].detach().cpu().numpy()
                        f.write(np.array2string(outstanding_orders, formatter={'float_kind':lambda x: f"{x:11.1f}\t"}))
                        f.write('\n')

                    for store_idx in range(n_stores):
                        allocation_matrix = store_allocation[sample_idx, store_idx].detach().cpu().numpy()
                        orders = (store_intermediate_outputs * edge_mask)[sample_idx, store_idx].detach().cpu().numpy()
                        formatted_allocation_orders = '[' + ', \t'.join([f'{allocation_matrix[i]:5.1f}/{orders[i]:5.1f}' for i in range(len(allocation_matrix))]) + '\t]'
                        f.write(formatted_allocation_orders)
                        
                        store_inv = observation['store_inventories'][sample_idx, store_idx].detach().cpu().numpy()
                        formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv[:-1]])
                        f.write(f"[{formatted_store_inv}]\n")

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class GNN_decentralized(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.n_stores = problem_params['n_stores']

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'holding_costs', 'underage_costs']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        list_to_cat = [observation[k].unsqueeze(-1) for k in params_to_stack]
        if 'demand_signals' in observation['internal_data']:
            if observation['current_period'] + 1 >= observation['internal_data']['demand_signals'].size(2):
                demand_signal = torch.zeros_like(observation['internal_data']['demand_signals'][:, :, observation['current_period']])
            else:
                demand_signal = observation['internal_data']['demand_signals'][:, :, observation['current_period'] + 1]
            list_to_cat.append(demand_signal)
        return torch.cat([observation['store_inventories'], *list_to_cat], dim=2)
    
    def get_warehouse_inventory_and_params(self, observation):
        params_to_cat = ['warehouse_holding_costs']
        if 'warehouse_edge_initial_cost' in observation:
            params_to_cat.append('warehouse_edge_initial_cost')
        if 'warehouse_edge_distance_cost' in observation:
            params_to_cat.append('warehouse_edge_distance_cost')
        list_to_cat_to_unsqueeze = [observation[k].unsqueeze(-1) for k in params_to_cat]
        return torch.cat([observation['warehouse_inventories'], *list_to_cat_to_unsqueeze], dim=-1)

    def get_network(self, layer_name, layer_idx):
        return self.net[layer_name]

    def forward(self, observation):
        def pad_features(tensor, inv_len, max_inv_len, max_states_len):
            inv = tensor[:,:,:inv_len]
            states = tensor[:,:,inv_len:]
            return torch.cat([
                F.pad(inv, (0, max_inv_len - inv_len)),
                F.pad(states, (0, max_states_len - (tensor.size(2) - inv_len)))
            ], dim=2)

        store_inv_len = observation['store_inventories'].size(2)
        warehouse_inv_len = observation['warehouse_inventories'].size(2)
        store_state = self.get_store_inventory_and_params(observation)
        warehouse_state = self.get_warehouse_inventory_and_params(observation)
        max_inv_len = max(store_inv_len, warehouse_inv_len)
        store_primitives_len = store_state.size(2) - store_inv_len
        warehouse_primitives_len = warehouse_state.size(2) - warehouse_inv_len
        max_primitives_len = max(store_primitives_len, warehouse_primitives_len)
        store_padded = pad_features(store_state, store_inv_len, max_inv_len, max_primitives_len)
        warehouse_padded = pad_features(warehouse_state, warehouse_inv_len, max_inv_len, max_primitives_len)
        states = torch.cat([warehouse_padded, store_padded], dim=1)
        
        nodes = self.net['initial_node'](states)
        n_warehouses = observation['warehouse_inventories'].size(1)
        adjacency = observation['warehouse_store_edges']  # 1 if connected, 0 otherwise
        adjacency_batch_idx, adjacency_warehouse_idx, adjacency_store_idx = adjacency.nonzero(as_tuple=True)
        supplier_warehouse_edges_input = torch.cat([torch.zeros_like(nodes[:, :n_warehouses]), nodes[:, :n_warehouses], observation['warehouse_lead_times'].unsqueeze(-1)], dim=-1)
        
        warehouses = nodes[:, :n_warehouses]
        stores = nodes[:, n_warehouses:]
        warehouse_emb = warehouses[adjacency_batch_idx, adjacency_warehouse_idx]
        store_emb = stores[adjacency_batch_idx, adjacency_store_idx]
        lead_times = observation['warehouse_store_edge_lead_times'][adjacency.bool()].unsqueeze(-1)
        warehouse_store_edges_input_flattened = torch.cat([warehouse_emb, store_emb, lead_times], dim=-1)
        warehouse_store_edges_input = warehouse_store_edges_input_flattened.view(nodes.size(0), -1, 2 * nodes.size(-1) + lead_times.size(-1))
        
        store_clients_edges_input = torch.cat([nodes[:, n_warehouses:], torch.zeros_like(nodes[:, n_warehouses:]), torch.zeros_like(observation['lead_times'].unsqueeze(-1))], dim=-1)
        edges_input = torch.cat([supplier_warehouse_edges_input, warehouse_store_edges_input, store_clients_edges_input], dim=1)
        edges = self.net['initial_edge'](edges_input)

        outputs = self.net['output'](edges[:, :-self.n_stores])
        warehouse_allocation = outputs[:, :n_warehouses, 0]

        wh_idx, st_idx = observation['warehouse_store_edges'][0].nonzero(as_tuple=True)
        n_edges = wh_idx.shape[0]
        one_hot_warehouse = torch.zeros(n_edges, n_warehouses, device=self.device)
        one_hot_warehouse[torch.arange(n_edges), wh_idx] = 1
        store_orders = outputs[:, n_warehouses:, 0]
        aggregated_orders = torch.matmul(store_orders, one_hot_warehouse)
        total_inventory = observation['warehouse_inventories'][:, :, 0]
        if self.__class__.__name__ == 'GNN_decentralized_transshipment':
            scaling_factor = total_inventory / (aggregated_orders + 1e-15)
        else:
            scaling_factor = torch.clamp(total_inventory / (aggregated_orders + 1e-15), max=1)
        edge_scaling = (scaling_factor.unsqueeze(1) * one_hot_warehouse.unsqueeze(0)).sum(dim=-1)
        store_allocation = store_orders * edge_scaling

        store_allocation_matrix = torch.zeros(outputs.size(0), self.n_stores, n_warehouses, device=self.device)
        batch_idx = torch.arange(outputs.size(0), device=self.device).unsqueeze(1).expand(outputs.size(0), n_edges)
        store_allocation_matrix[batch_idx, st_idx.unsqueeze(0).expand(outputs.size(0), n_edges), wh_idx.unsqueeze(0).expand(outputs.size(0), n_edges)] = store_allocation
        
        # if self.is_debugging:
        #     debug_dir = "/user/ml4723/Prj/NIC/debug"
        #     store_orders_matrix = torch.zeros(outputs.size(0), self.n_stores, n_warehouses, device=self.device)
        #     store_orders_matrix[batch_idx, st_idx.unsqueeze(0).expand(outputs.size(0), n_edges), wh_idx.unsqueeze(0).expand(outputs.size(0), n_edges)] = store_orders
        #     if self.debug_identifier is not None:
        #         debug_dir = debug_dir + self.debug_identifier
        #     os.makedirs(debug_dir, exist_ok=True)
        #     for sample_idx in range(min(outputs.size(0),8)):
        #         with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
        #             f.write("\n\n")
        #             outputs_array = outputs[sample_idx, :n_warehouses, 0].detach().cpu().numpy()
        #             f.write(np.array2string(outputs_array, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
        #             f.write('\n')
                    
        #             max_lead_time = int(observation['warehouse_lead_times'][sample_idx].max().item())
        #             for lead_time in range(max_lead_time, 0, -1):
        #                 outstanding_orders = observation['warehouse_inventories'][sample_idx, :, lead_time - 1].detach().cpu().numpy()
        #                 f.write(np.array2string(outstanding_orders, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
        #                 f.write('\n')

        #             for store_idx in range(self.n_stores):
        #                 allocation_matrix = store_allocation_matrix[sample_idx, store_idx].detach().cpu().numpy()
        #                 orders = store_orders_matrix[sample_idx, store_idx].detach().cpu().numpy()
        #                 formatted_allocation_orders = f'{store_idx} [' + ', \t'.join([f'{allocation_matrix[i]:4.1f}/{orders[i]:4.1f}' for i in range(len(allocation_matrix))]) + ']'
        #                 f.write(formatted_allocation_orders)
                        
        #                 store_inv = observation['store_inventories'][sample_idx, store_idx].detach().cpu().numpy()
        #                 formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv[:-1]])

        #                 if 'demand_signals' in observation['internal_data']:
        #                     if observation['current_period'] + 1 >= observation['internal_data']['demand_signals'].size(2):
        #                         demand_signal = torch.zeros_like(observation['internal_data']['demand_signals'][:, :, observation['current_period']])
        #                     else:
        #                         demand_signal = observation['internal_data']['demand_signals'][:, :, observation['current_period'] + 1]
        #                     f.write(f"[{formatted_store_inv}] - {demand_signal[sample_idx, store_idx].detach().cpu().numpy()} - {observation['underage_costs'][sample_idx, store_idx].detach().cpu().numpy()}\n")
        #                 else:
        #                     f.write(f"[{formatted_store_inv}] - {observation['underage_costs'][sample_idx, store_idx].detach().cpu().numpy()}\n")
        
        return {
            'stores': store_allocation_matrix,
            'warehouses': warehouse_allocation
        }

class GNN_decentralized_transshipment(GNN_decentralized):
    pass

class GNN(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.n_stores = problem_params['n_stores']
        self.pna_delta = (torch.log(torch.tensor(self.n_stores + 1, device=self.device)) 
                          + self.n_stores * torch.log(torch.tensor(2, device=self.device))) \
                          / torch.tensor(self.n_stores + 1, device=self.device)
        self.use_pna = 'use_pna' in args and args['use_pna']
        self.NN_per_layer = 'NN_per_layer' in args and args['NN_per_layer']
        self.skip_connection = 'skip_connection' in args and args['skip_connection']
        self.apply_edge_embedding = 'apply_edge_embedding' in args and args['apply_edge_embedding']
        self.apply_bottleneck_loss = 'apply_bottleneck_loss' in args and args['apply_bottleneck_loss']
        self.edges_separation_mode = args['edges_separation_mode'] if 'edges_separation_mode' in args else None
        self.soft_min = 'soft_min' in args and args['soft_min']
        self.self_loop = 'self_loop' in args and args['self_loop']
        self.n_MP = None
        if 'n_MP' in args:
            self.n_MP = args['n_MP']

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'holding_costs', 'underage_costs']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        list_to_cat = [observation[k].unsqueeze(-1) for k in params_to_stack if k in observation]
        if 'demand_signals' in observation['internal_data']:
            if observation['current_period'] + 1 >= observation['internal_data']['demand_signals'].size(2):
                demand_signal = torch.zeros_like(observation['internal_data']['demand_signals'][:, :, observation['current_period']])
            else:
                demand_signal = observation['internal_data']['demand_signals'][:, :, observation['current_period'] + 1]
            list_to_cat.append(demand_signal)
        return torch.cat([observation['store_inventories'], *list_to_cat], dim=2)
    
    def get_warehouse_inventory_and_params(self, observation):
        params_to_cat = ['warehouse_holding_costs']
        if 'warehouse_edge_initial_cost' in observation:
            params_to_cat.append('warehouse_edge_initial_cost')
        if 'warehouse_edge_distance_cost' in observation:
            params_to_cat.append('warehouse_edge_distance_cost')
        list_to_cat_to_unsqueeze = [observation[k].unsqueeze(-1) for k in params_to_cat]
        return torch.cat([observation['warehouse_inventories'], *list_to_cat_to_unsqueeze], dim=-1)

    def get_network(self, layer_name, layer_idx):
        if self.NN_per_layer:
            return self.net[f'{layer_name}_{layer_idx+1}']
        else:
            return self.net[layer_name]

    def forward(self, observation):
        def pad_features(tensor, inv_len, max_inv_len, max_states_len):
            inv = tensor[:,:,:inv_len]
            states = tensor[:,:,inv_len:]
            return torch.cat([
                F.pad(inv, (0, max_inv_len - inv_len)),
                F.pad(states, (0, max_states_len - (tensor.size(2) - inv_len)))
            ], dim=2)

        store_inv_len = observation['store_inventories'].size(2)
        warehouse_inv_len = observation['warehouse_inventories'].size(2)
        n_warehouses = observation['warehouse_inventories'].size(1)
        if 'echelon_inventories' in observation:
            echelon_inventories = observation['echelon_inventories']
            warehouse_inventories = observation['warehouse_inventories']
            echelon_inv_len = echelon_inventories.size(2)

            store_state = torch.cat([observation['store_inventories'], observation['holding_costs'].unsqueeze(2), observation['underage_costs'].unsqueeze(2)], dim=-1)
            warehouse_state = torch.cat([warehouse_inventories, observation['warehouse_holding_costs'].unsqueeze(2)], dim=-1)
            echelon_state = torch.cat([echelon_inventories, observation['echelon_holding_costs'].unsqueeze(2)], dim=-1)
    
            store_primitives_len = store_state.size(2) - store_inv_len
            warehouse_primitives_len = warehouse_state.size(2) - warehouse_inv_len
            echelon_primitives_len = echelon_state.size(2) - echelon_inv_len
            max_primitives_len = max(store_primitives_len, warehouse_primitives_len, echelon_primitives_len)

            n_echelons = echelon_inventories.size(1)
            max_inv_len = max(store_inv_len, warehouse_inv_len, echelon_inv_len)

            store_padded = pad_features(store_state, store_inv_len, max_inv_len, max_primitives_len)
            warehouse_padded = pad_features(warehouse_state, warehouse_inv_len, max_inv_len, max_primitives_len)
            echelon_padded = pad_features(echelon_state, echelon_inv_len, max_inv_len, max_primitives_len)

            states = torch.cat([echelon_padded, warehouse_padded, store_padded], dim=1)
            n_MP = n_echelons + 1
        else:
            store_state = self.get_store_inventory_and_params(observation)
            
            warehouse_state = self.get_warehouse_inventory_and_params(observation)
            max_inv_len = max(store_inv_len, warehouse_inv_len)

            store_primitives_len = store_state.size(2) - store_inv_len
            warehouse_primitives_len = warehouse_state.size(2) - warehouse_inv_len
            max_primitives_len = max(store_primitives_len, warehouse_primitives_len)

            store_padded = pad_features(store_state, store_inv_len, max_inv_len, max_primitives_len)
            warehouse_padded = pad_features(warehouse_state, warehouse_inv_len, max_inv_len, max_primitives_len)

            states = torch.cat([warehouse_padded, store_padded], dim=1)
            n_MP = 1
        if self.n_MP is not None:
            n_MP = self.n_MP
        
        nodes = self.net['initial_node'](states)
        if 'echelon_inventories' in observation:
            zero_node = torch.zeros_like(nodes[:, :1])
            if self.self_loop:
                edges_input = torch.cat([
                    torch.cat([zero_node, nodes, nodes[:, :-1]], dim=1), 
                    torch.cat([nodes, zero_node, nodes[:, :-1]], dim=1),
                    torch.cat([observation['echelon_lead_times'], observation['warehouse_lead_times'], observation['lead_times'], torch.zeros_like(observation['lead_times']), torch.zeros(nodes.size(0), nodes.size(1) - 1, device=self.device)], dim=1).unsqueeze(-1)
                ], dim=-1)
            else:
                edges_input = torch.cat([
                    torch.cat([zero_node, nodes], dim=1), 
                    torch.cat([nodes, zero_node], dim=1),
                    torch.cat([observation['echelon_lead_times'], observation['warehouse_lead_times'], observation['lead_times'], torch.zeros_like(observation['lead_times'])], dim=1).unsqueeze(-1)
                ], dim=-1)
        elif 'warehouse_store_edge_lead_times' in observation:
            warehouse_store_edges_observation = observation['warehouse_store_edges']
            if self.edges_separation_mode != None:
                store_connections = warehouse_store_edges_observation[0].sum(dim=0)
                multi_connected_stores = (store_connections > 1).nonzero().squeeze(-1)
                for store_idx in multi_connected_stores:
                    connected_warehouses = warehouse_store_edges_observation[0, :, store_idx].bool()
                    if self.edges_separation_mode == "fastest":
                        costs = observation['warehouse_store_edge_lead_times'][0, :, store_idx]
                    elif self.edges_separation_mode == "cheapest":
                        costs = observation['warehouse_edge_initial_cost'][0]
                    elif self.edges_separation_mode == "cheapest_holding":
                        costs = observation['warehouse_holding_costs'][0]

                    costs_connected = costs[connected_warehouses]
                    min_cost_idx = costs_connected.argmin(dim=-1)
                    # Get the actual warehouse index from the connected warehouses
                    connected_warehouse_indices = connected_warehouses.nonzero().squeeze(-1)
                    connection_index = connected_warehouse_indices[min_cost_idx]
                    warehouse_store_edges_observation[:, :, store_idx] = 0
                    warehouse_store_edges_observation[:, connection_index, store_idx] = 1

            adjacency = warehouse_store_edges_observation  # 1 if connected, 0 otherwise
            adjacency_batch_idx, adjacency_warehouse_idx, adjacency_store_idx = adjacency.nonzero(as_tuple=True)
            if self.__class__.__name__ == 'GNN_real':
                supplier_warehouse_edges_input = torch.cat([torch.zeros_like(nodes[:, :n_warehouses]), nodes[:, :n_warehouses], observation['warehouse_lead_times'].unsqueeze(-1), observation['warehouse_orders']], dim=-1)
            else:
                supplier_warehouse_edges_input = torch.cat([torch.zeros_like(nodes[:, :n_warehouses]), nodes[:, :n_warehouses], observation['warehouse_lead_times'].unsqueeze(-1)], dim=-1)
            
            warehouses = nodes[:, :n_warehouses]
            stores = nodes[:, n_warehouses:]
            warehouse_emb = warehouses[adjacency_batch_idx, adjacency_warehouse_idx]
            store_emb = stores[adjacency_batch_idx, adjacency_store_idx]
            lead_times = observation['warehouse_store_edge_lead_times'][adjacency.bool()].unsqueeze(-1)
            if self.__class__.__name__ == 'GNN_real':
                orders = observation['store_orders'].transpose(1,2)[adjacency.bool()]
                warehouse_store_edges_input_flattened = torch.cat([warehouse_emb, store_emb, lead_times, orders], dim=-1)
                warehouse_store_edges_input = warehouse_store_edges_input_flattened.view(nodes.size(0), -1, 2 * nodes.size(-1) + lead_times.size(-1) + orders.size(-1))
                zeros_orders = torch.zeros(observation['store_orders'].size(0), observation['store_orders'].size(1), observation['store_orders'].size(3), device=self.device)
                store_clients_edges_input = torch.cat([nodes[:, n_warehouses:], torch.zeros_like(nodes[:, n_warehouses:]), torch.zeros_like(observation['lead_times'].unsqueeze(-1)), zeros_orders], dim=-1)
            else:
                warehouse_store_edges_input_flattened = torch.cat([warehouse_emb, store_emb, lead_times], dim=-1)
                warehouse_store_edges_input = warehouse_store_edges_input_flattened.view(nodes.size(0), -1, 2 * nodes.size(-1) + lead_times.size(-1))
                store_clients_edges_input = torch.cat([nodes[:, n_warehouses:], torch.zeros_like(nodes[:, n_warehouses:]), torch.zeros_like(observation['lead_times'].unsqueeze(-1))], dim=-1)
            
            if self.self_loop:
                if self.__class__.__name__ == 'GNN_real':
                    self_loop_input = torch.cat([nodes[:, :n_warehouses], nodes[:, :n_warehouses], torch.zeros_like(observation['warehouse_lead_times'].unsqueeze(-1)), observation['warehouse_self_loop_orders']], dim=-1)
                else:
                    self_loop_input = torch.cat([nodes[:, :n_warehouses], nodes[:, :n_warehouses], torch.zeros_like(observation['warehouse_lead_times'].unsqueeze(-1))], dim=-1)
                edges_input = torch.cat([supplier_warehouse_edges_input, warehouse_store_edges_input, store_clients_edges_input, self_loop_input], dim=1)
            else:
                edges_input = torch.cat([supplier_warehouse_edges_input, warehouse_store_edges_input, store_clients_edges_input], dim=1)
        else:
            if self.__class__.__name__ == 'GNN_real':
                supplier_warehouse_edges_input = torch.cat([torch.zeros_like(nodes[:, :1]), nodes[:, :1], observation['warehouse_lead_times'].unsqueeze(-1), observation['warehouse_orders']], dim=-1)
                warehouse_store_edges_input = torch.cat([nodes[:, :1].repeat(1, self.n_stores, 1), nodes[:, 1:], observation['lead_times'].unsqueeze(-1), observation['store_orders']], dim=-1)
                zeros_orders = torch.zeros(observation['store_orders'].size(0), observation['store_orders'].size(1), observation['store_orders'].size(2), device=self.device)
                store_clients_edges_input = torch.cat([nodes[:, 1:], torch.zeros_like(nodes[:, 1:]), torch.zeros_like(observation['lead_times'].unsqueeze(-1)), zeros_orders], dim=-1)
            else:
                supplier_warehouse_edges_input = torch.cat([torch.zeros_like(nodes[:, :1]), nodes[:, :1], observation['warehouse_lead_times'].unsqueeze(-1)], dim=-1)
                warehouse_store_edges_input = torch.cat([nodes[:, :1].repeat(1, self.n_stores, 1), nodes[:, 1:], observation['lead_times'].unsqueeze(-1)], dim=-1)
                store_clients_edges_input = torch.cat([nodes[:, 1:], torch.zeros_like(nodes[:, 1:]), torch.zeros_like(observation['lead_times'].unsqueeze(-1))], dim=-1)
                   
            if self.self_loop:
                if self.__class__.__name__ == 'GNN_real':
                    self_loop_input = torch.cat([nodes[:, :1], nodes[:, :1], torch.zeros_like(observation['warehouse_lead_times'].unsqueeze(-1)), observation['warehouse_self_loop_orders']], dim=-1)
                else:
                    self_loop_input = torch.cat([nodes[:, :1], nodes[:, :1], torch.zeros_like(observation['warehouse_lead_times'].unsqueeze(-1))], dim=-1)
                edges_input = torch.cat([supplier_warehouse_edges_input, warehouse_store_edges_input, store_clients_edges_input, self_loop_input], dim=1)
            else:
                edges_input = torch.cat([supplier_warehouse_edges_input, warehouse_store_edges_input, store_clients_edges_input], dim=1)
        edges = self.net['initial_edge'](edges_input)

        for layer_idx in range(n_MP):
            if self.apply_edge_embedding:
                edges_for_aggregation = self.get_network('edge_embedding', layer_idx)(edges)
            else:
                edges_for_aggregation = edges
            if 'echelon_inventories' in observation:
                if self.self_loop:
                    supplier_edges_aggregation = (edges_for_aggregation[:, :nodes.size(1) - 1, :] + edges_for_aggregation[:, nodes.size(1)+1 :]) / torch.sqrt(torch.tensor(2, device=self.device))
                    recipient_edges_aggregation = (edges_for_aggregation[:, 1:nodes.size(1), :] + edges_for_aggregation[:, nodes.size(1)+1 :]) / torch.sqrt(torch.tensor(2, device=self.device))
                    node_update_input = torch.cat([
                        nodes,
                        torch.cat([supplier_edges_aggregation, edges_for_aggregation[:, nodes.size(1)-1:nodes.size(1)]], dim=1), 
                        torch.cat([recipient_edges_aggregation, edges_for_aggregation[:, nodes.size(1):nodes.size(1)+1]], dim=1)
                    ], dim=-1)
                else:
                    node_update_input = torch.cat([
                        nodes,
                        edges_for_aggregation[:, :-1], 
                        edges_for_aggregation[:, 1:]
                    ], dim=-1)
            elif 'warehouse_store_edge_lead_times' in observation:
                wh_idx, st_idx = adjacency[0].nonzero(as_tuple=True)
                n_edges_w_to_s = wh_idx.shape[0]
                if self.self_loop:
                    warehouse_supplier_aggregation = (edges_for_aggregation[:, :n_warehouses] + edges_for_aggregation[:, -n_warehouses:]) / torch.sqrt(torch.tensor(2, device=self.device))
                else:
                    warehouse_supplier_aggregation = edges_for_aggregation[:, :n_warehouses]
                warehouse_store_edges = edges_for_aggregation[:, n_warehouses:n_warehouses + n_edges_w_to_s]

                one_hot_warehouse = torch.zeros(n_edges_w_to_s, n_warehouses, device=self.device)
                one_hot_warehouse[torch.arange(n_edges_w_to_s), wh_idx] = 1
                warehouse_aggregated_sum = torch.matmul(warehouse_store_edges.transpose(1, 2), one_hot_warehouse)
                warehouse_aggregated_sum = warehouse_aggregated_sum.transpose(1, 2)
                warehouse_counts = one_hot_warehouse.sum(dim=0)
                warehouse_counts = warehouse_counts.unsqueeze(0).unsqueeze(-1)
                # Replace zero counts with a small value to avoid division by zero
                warehouse_counts = torch.where(warehouse_counts == 0, 
                                              torch.tensor(0.0000001, device=self.device), 
                                              warehouse_counts)
                if self.self_loop:
                    warehouse_aggregated_sum = warehouse_aggregated_sum + edges_for_aggregation[:, -n_warehouses:]
                    warehouse_counts = warehouse_counts + 1
                warehouse_recipient_aggregation = warehouse_aggregated_sum / torch.sqrt(warehouse_counts)
                # warehouse_recipient_aggregation = warehouse_aggregated_sum / warehouse_counts

                one_hot_store = torch.zeros(n_edges_w_to_s, self.n_stores, device=self.device)
                one_hot_store[torch.arange(n_edges_w_to_s), st_idx] = 1
                store_aggregated_sum = torch.matmul(warehouse_store_edges.transpose(1, 2), one_hot_store)
                store_aggregated_sum = store_aggregated_sum.transpose(1, 2)

                store_supplier_aggregation = store_aggregated_sum
                store_counts = one_hot_store.sum(dim=0)
                store_counts = store_counts.unsqueeze(0).unsqueeze(-1)
                store_supplier_aggregation = store_aggregated_sum / torch.sqrt(store_counts)
                # store_supplier_aggregation = store_aggregated_sum / store_counts
                store_recipient_aggregation = edges_for_aggregation[:, -self.n_stores:]

                if self.use_pna:
                    node_update_input = torch.cat(
                        [torch.cat([nodes[:, :n_warehouses], warehouse_supplier_aggregation, warehouse_supplier_aggregation, warehouse_recipient_aggregation, warehouse_aggregated_sum / warehouse_counts], dim=-1),
                            torch.cat([nodes[:, n_warehouses:], store_supplier_aggregation, store_aggregated_sum / store_counts, store_recipient_aggregation, store_recipient_aggregation], dim=-1)],
                        dim=1
                    )
                else:
                    node_update_input = torch.cat(
                        [torch.cat([nodes[:, :n_warehouses], warehouse_supplier_aggregation, warehouse_recipient_aggregation], dim=-1),
                            torch.cat([nodes[:, n_warehouses:], store_supplier_aggregation, store_recipient_aggregation], dim=-1)],
                        dim=1
                    )
            else: 
                if self.self_loop:
                    warehouse_supplier_aggregation = (edges_for_aggregation[:, :1, :] + edges_for_aggregation[:, -1:, :]) / torch.sqrt(torch.tensor(2, device=self.device))
                    warehouse_recipient_aggregation = (torch.sum(edges_for_aggregation[:, 1:1 + self.n_stores, :], dim=1, keepdim=True) + edges_for_aggregation[:, -1:, :]) / torch.sqrt(torch.tensor(self.n_stores + 1, device=self.device))
                else:
                    warehouse_supplier_aggregation = edges_for_aggregation[:, :1, :]
                    warehouse_recipient_aggregation = torch.sum(edges_for_aggregation[:, 1:1 + self.n_stores, :], dim=1, keepdim=True) / torch.sqrt(torch.tensor(self.n_stores, device=self.device))
                    
                store_supplier_aggregation = edges_for_aggregation[:, 1:1 + self.n_stores, :]
                store_recipient_aggregation = edges_for_aggregation[:, 1 + self.n_stores:1 + 2 * self.n_stores, :]

                node_update_input = torch.cat(
                    [torch.cat([nodes[:, :1], warehouse_supplier_aggregation, warehouse_recipient_aggregation], dim=-1),
                     torch.cat([nodes[:, 1:], store_supplier_aggregation, store_recipient_aggregation], dim=-1)],
                    dim=1
                )
            nodes_updates = self.get_network('node_update', layer_idx)(node_update_input)
            nodes = nodes + nodes_updates

            if 'echelon_inventories' in observation:
                zero_node = torch.zeros_like(nodes[:, :1])
                if self.self_loop:
                    edges_update_input = torch.cat([
                        edges,
                        torch.cat([zero_node, nodes, nodes[:, :-1]], dim=1), 
                        torch.cat([nodes, zero_node, nodes[:, :-1]], dim=1),
                    ], dim=-1)
                else:
                    edges_update_input = torch.cat([
                        edges,
                        torch.cat([zero_node, nodes], dim=1), 
                        torch.cat([nodes, zero_node], dim=1),
                    ], dim=-1)
            else:
                if 'warehouse_store_edge_lead_times' in observation:
                    supplier_warehouse_edges_update_input = torch.cat([edges[:, :n_warehouses], torch.zeros_like(nodes[:, :n_warehouses]), nodes[:, :n_warehouses]], dim=-1)
                    
                    warehouses = nodes[:, :n_warehouses]
                    stores = nodes[:, n_warehouses:]
                    warehouse_emb = warehouses[adjacency_batch_idx, adjacency_warehouse_idx]
                    store_emb = stores[adjacency_batch_idx, adjacency_store_idx]
                    warehouse_store_edges = edges[:, n_warehouses:n_warehouses+n_edges_w_to_s]
                    warehouse_store_edges_update_input = torch.cat([
                        warehouse_store_edges,
                        warehouse_emb.view(nodes.size(0), -1, nodes.size(-1)),
                        store_emb.view(nodes.size(0), -1, nodes.size(-1))
                    ], dim=-1)
                    
                    store_clients_edges_update_input = torch.cat([edges[:, n_warehouses+n_edges_w_to_s:n_warehouses+n_edges_w_to_s+self.n_stores], nodes[:, n_warehouses:], torch.zeros_like(nodes[:, n_warehouses:])], dim=-1)
                else:
                    supplier_warehouse_edges_update_input = torch.cat([edges[:, :1], torch.zeros_like(nodes[:, :1]), nodes[:, :1]], dim=-1)
                    warehouse_store_edges_update_input = torch.cat([edges[:, 1:1 + self.n_stores], nodes[:, :1].repeat(1, self.n_stores, 1), nodes[:, 1:]], dim=-1)
                    
                    store_clients_edges_update_input = torch.cat([edges[:, 1 + self.n_stores:1 + 2 * self.n_stores], nodes[:, 1:], torch.zeros_like(nodes[:, 1:])], dim=-1)
                
                if self.self_loop:
                    self_loop_update_input = torch.cat([edges[:, -n_warehouses:], nodes[:, :n_warehouses], nodes[:, :n_warehouses]], dim=-1)
                    edges_update_input = torch.cat([supplier_warehouse_edges_update_input, warehouse_store_edges_update_input, store_clients_edges_update_input, self_loop_update_input], dim=1)
                else:
                    edges_update_input = torch.cat([supplier_warehouse_edges_update_input, warehouse_store_edges_update_input, store_clients_edges_update_input], dim=1)

            edges_updates = self.get_network('edge_update', layer_idx)(edges_update_input)
            edges = edges + edges_updates

        if self.skip_connection:
            if 'echelon_inventories' in observation:
                zero_states = torch.zeros_like(states[:, :1])
                edge_states = torch.cat([
                    torch.cat([zero_states, states[:, :-1]], dim=1), 
                    states,
                    torch.cat([observation['echelon_lead_times'], observation['warehouse_lead_times'], observation['lead_times']], dim=1).unsqueeze(-1)
                ], dim=-1)
                outputs = self.net['output'](torch.cat([edge_states, edges[:, :-1]], dim=-1))
            elif 'warehouse_store_edge_lead_times' in observation:
                warehouses = states[:, :n_warehouses]
                stores = states[:, n_warehouses:]
                if self.__class__.__name__ == 'GNN_real':
                    supplier_warehouse_edges_input = torch.cat([torch.zeros_like(warehouses), warehouses, observation['warehouse_lead_times'].unsqueeze(-1), observation['warehouse_orders']], dim=-1)
                else:
                    supplier_warehouse_edges_input = torch.cat([torch.zeros_like(warehouses), warehouses, observation['warehouse_lead_times'].unsqueeze(-1)], dim=-1)
                
                warehouse_states = warehouses[adjacency_batch_idx, adjacency_warehouse_idx]
                store_states = stores[adjacency_batch_idx, adjacency_store_idx]
                lead_times = observation['warehouse_store_edge_lead_times'][adjacency.bool()].unsqueeze(-1)
                if self.__class__.__name__ == 'GNN_real':
                    orders = observation['store_orders'].transpose(1,2)[adjacency.bool()]
                    warehouse_store_edges_input_flattened = torch.cat([warehouse_states, store_states, lead_times, orders], dim=-1)
                    warehouse_store_edges_input = warehouse_store_edges_input_flattened.view(states.size(0), -1, 2 * states.size(-1) + lead_times.size(-1) + orders.size(-1))
                else:
                    warehouse_store_edges_input_flattened = torch.cat([warehouse_states, store_states, lead_times], dim=-1)
                    warehouse_store_edges_input = warehouse_store_edges_input_flattened.view(states.size(0), -1, 2 * states.size(-1) + lead_times.size(-1))
                edges_states = torch.cat([supplier_warehouse_edges_input, warehouse_store_edges_input], dim=1)
                outputs = self.net['output'](torch.cat([edges_states, edges[:, :-self.n_stores, :]], dim=-1))
            else:
                if self.__class__.__name__ == 'GNN_real':
                    supplier_warehouse_edges_states = torch.cat([torch.zeros_like(states[:, :1]), states[:, :1], observation['warehouse_lead_times'].unsqueeze(-1), observation['warehouse_orders']], dim=-1)
                    warehouse_store_edges_states = torch.cat([states[:, :1].repeat(1, self.n_stores, 1), states[:, 1:], observation['lead_times'].unsqueeze(-1), observation['store_orders']], dim=-1)
                else:
                    supplier_warehouse_edges_states = torch.cat([torch.zeros_like(states[:, :1]), states[:, :1], observation['warehouse_lead_times'].unsqueeze(-1)], dim=-1)
                    warehouse_store_edges_states = torch.cat([states[:, :1].repeat(1, self.n_stores, 1), states[:, 1:], observation['lead_times'].unsqueeze(-1)], dim=-1)
                edges_states = torch.cat([supplier_warehouse_edges_states, warehouse_store_edges_states], dim=1)
                outputs = self.net['output'](torch.cat([edges_states, edges[:, :1 + self.n_stores, :]], dim=-1))
        else:
            if 'echelon_inventories' in observation:
                if self.self_loop:
                    outputs = self.net['output'](edges[:, :nodes.size(1)])
                    loop_outputs = self.net['output'](edges[:, nodes.size(1)+1:])
                else:
                    outputs = self.net['output'](edges[:, :-1])
            elif 'warehouse_store_edge_lead_times' in observation:
                if self.self_loop:
                    outputs = self.net['output'](edges[:, :-self.n_stores - n_warehouses])
                    loop_outputs = self.net['output'](edges[:, -n_warehouses:])[:, :, 0]
                else:
                    outputs = self.net['output'](edges[:, :-self.n_stores])
            else:
                if self.self_loop:
                    outputs = self.net['output'](edges[:, :1 + self.n_stores, :])
                    warehouse_loop_output = self.net['output'](edges[:, -1:, :])
                else:
                    outputs = self.net['output'](edges[:, :1 + self.n_stores, :])

        if 'echelon_inventories' in observation:
            echelon_allocations = []
            if self.self_loop:
                for j in range(outputs.size(1) - 2):
                    if j == 0:  # First echelon
                        echelon_allocations.append(outputs[:, j:j+1, 0])
                    else:
                        echelon_allocations.append(self.apply_proportional_allocation(torch.cat([outputs[:, j:j+1, 0], loop_outputs[:, j-1:j, 0]], dim=1), echelon_inventories[:,j-1:j])[:, :-1])
                warehouse_allocation = self.apply_proportional_allocation(torch.cat([outputs[:, -2: -1, 0], loop_outputs[:, -2:-1, 0]], dim=1), echelon_inventories[:,-1:, :])[:, :-1]
                store_allocation = self.apply_proportional_allocation(torch.cat([outputs[:, -1:, 0], loop_outputs[:, -1:, 0]], dim=1), warehouse_inventories)[:, :-1]
            else:
                for j in range(outputs.size(1) - 2):
                    if j == 0:  # First echelon
                        echelon_allocations.append(outputs[:, j:j+1, 0])
                    else:
                        echelon_allocations.append(self.apply_proportional_allocation(outputs[:, j:j+1, 0], echelon_inventories[:,j-1:j]))
                warehouse_allocation = self.apply_proportional_allocation(outputs[:, -2: -1, 0], echelon_inventories[:,-1:, :])
                store_allocation = self.apply_proportional_allocation(outputs[:, -1:, 0], warehouse_inventories)

            if self.is_debugging:
                debug_dir = "/user/ml4723/Prj/NIC/debug"
                if self.debug_identifier is not None:
                    debug_dir = debug_dir + self.debug_identifier
                os.makedirs(debug_dir, exist_ok=True)
                for sample_idx in range(32):
                    with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
                        inventory_state = []
                        numeric_inventories = []  # List to store all inventory values
                        
                        # Process echelon inventories
                        for echelon_idx in range(echelon_inventories.size(1)):
                            lead_time = observation['echelon_lead_times'][sample_idx, echelon_idx].item()
                            echelon_inv = echelon_inventories[sample_idx, echelon_idx, :int(lead_time)].detach().cpu().numpy()
                            if echelon_idx > 0:
                                inventory_state.append('|')
                            inventory_state.extend(echelon_inv[::-1])
                            numeric_inventories.extend(echelon_inv[::-1])
                        
                        # Process warehouse inventories
                        inventory_state.append('|')
                        warehouse_inv = warehouse_inventories[sample_idx, :, :].detach().cpu().numpy().flatten()
                        inventory_state.extend(warehouse_inv[::-1])
                        numeric_inventories.extend(warehouse_inv[::-1])
                        
                        # Process store inventories
                        inventory_state.append('|')
                        lead_time = observation['lead_times'][sample_idx, 0].detach().cpu().numpy()
                        store_inv = observation['store_inventories'][sample_idx, 0, :int(lead_time)].detach().cpu().numpy()[::-1]
                        inventory_state.extend(store_inv)
                        numeric_inventories.extend(store_inv)
                        
                        # Write inventory with separators
                        f.write("inventory: [")
                        for i, x in enumerate(inventory_state):
                            if isinstance(x, str):
                                f.write(f" {x} ")
                            else:
                                f.write(f"{x:.1f}")
                                if i < len(inventory_state)-1 and isinstance(inventory_state[i+1], str) == False:
                                    f.write(", ")
                        f.write("]\n")
                        
                        # Calculate and write sum and std of all inventories
                        total_sum = np.sum(numeric_inventories)
                        total_std = np.std(numeric_inventories)
                        f.write(f"sum: {total_sum:.1f}, std: {total_std:.1f}\n")
                        
                        # Write outputs with aligned labels
                        raw_outputs = []
                        for j in range(outputs.size(1)):
                            raw_outputs.append(outputs[sample_idx, j, 0].detach().item())
                        f.write(f"outputs: [{', '.join(f'{x:.1f}' for x in raw_outputs)}]\n")
                        
                        orders = []
                        orders.extend([e[sample_idx].detach().item() for e in echelon_allocations])
                        orders.append(warehouse_allocation[sample_idx].detach().item())
                        orders.append(store_allocation[sample_idx].detach().item())
                        f.write(f"orders: [{', '.join(f'{x:.1f}' for x in orders)}]\n\n")

            echelon_allocations_in_tensor = torch.cat(echelon_allocations, dim=1)
            if self.apply_bottleneck_loss:
                bottleneck_loss = (outputs.squeeze(-1) - torch.cat([echelon_allocations_in_tensor, warehouse_allocation, store_allocation], dim=-1)).sum()
                return {
                    'stores': store_allocation,
                    'warehouses': warehouse_allocation,
                    'echelons': echelon_allocations_in_tensor,
                    'bottleneck_loss': bottleneck_loss
                }
            else:
                return {
                    'stores': store_allocation,
                    'warehouses': warehouse_allocation,
                    'echelons': echelon_allocations_in_tensor
                }
        elif 'warehouse_store_edge_lead_times' in observation:
            warehouse_allocation = outputs[:, :n_warehouses, 0]

            wh_idx, st_idx = warehouse_store_edges_observation[0].nonzero(as_tuple=True)
            one_hot_warehouse = torch.zeros(n_edges_w_to_s, n_warehouses, device=self.device)
            one_hot_warehouse[torch.arange(n_edges_w_to_s), wh_idx] = 1
            store_orders = outputs[:, n_warehouses:, 0]
            aggregated_orders = torch.matmul(store_orders, one_hot_warehouse)
            if self.self_loop:
                aggregated_orders = aggregated_orders + loop_outputs
            total_inventory = observation['warehouse_inventories'][:, :, 0]

            if self.__class__.__name__ == 'GNN_transshipment':
                scaling_factor = total_inventory / (aggregated_orders + 1e-15)
            else:
                scaling_factor = torch.clamp(total_inventory / (aggregated_orders + 1e-15), max=1)
            edge_scaling = (scaling_factor.unsqueeze(1) * one_hot_warehouse.unsqueeze(0)).sum(dim=-1)
            store_allocation = store_orders * edge_scaling

            store_allocation_matrix = torch.zeros(outputs.size(0), self.n_stores, n_warehouses, device=self.device)
            batch_idx = torch.arange(outputs.size(0), device=self.device).unsqueeze(1).expand(outputs.size(0), n_edges_w_to_s)
            store_allocation_matrix[batch_idx, st_idx.unsqueeze(0).expand(outputs.size(0), n_edges_w_to_s), wh_idx.unsqueeze(0).expand(outputs.size(0), n_edges_w_to_s)] = store_allocation
            # if self.is_debugging:
            #     debug_dir = "/user/ml4723/Prj/NIC/debug"
            #     store_orders_matrix = torch.zeros(outputs.size(0), self.n_stores, n_warehouses, device=self.device)
            #     store_orders_matrix[batch_idx, st_idx.unsqueeze(0).expand(outputs.size(0), n_edges_w_to_s), wh_idx.unsqueeze(0).expand(outputs.size(0), n_edges_w_to_s)] = store_orders
            #     if self.debug_identifier is not None:
            #         debug_dir = debug_dir + self.debug_identifier
            #     os.makedirs(debug_dir, exist_ok=True)
            #     for sample_idx in range(min(outputs.size(0), 8)):
            #         with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
            #             f.write("\n\n")
            #             outputs_array = outputs[sample_idx, :n_warehouses, 0].detach().cpu().numpy()
            #             f.write(np.array2string(outputs_array, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
            #             f.write('\n')
                        
            #             max_lead_time = int(observation['warehouse_lead_times'][sample_idx].max().item())
            #             for lead_time in range(max_lead_time, 0, -1):
            #                 outstanding_orders = observation['warehouse_inventories'][sample_idx, :, lead_time - 1].detach().cpu().numpy()
            #                 f.write(np.array2string(outstanding_orders, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
            #                 f.write('\n')
    
            #             for store_idx in range(self.n_stores):
            #                 allocation_matrix = store_allocation_matrix[sample_idx, store_idx].detach().cpu().numpy()
            #                 orders = store_orders_matrix[sample_idx, store_idx].detach().cpu().numpy()
            #                 formatted_allocation_orders = f'{store_idx} [' + ', \t'.join([f'{allocation_matrix[i]:4.1f}/{orders[i]:4.1f}' for i in range(len(allocation_matrix))]) + ']'
            #                 f.write(formatted_allocation_orders)
                            
            #                 store_inv = observation['store_inventories'][sample_idx, store_idx].detach().cpu().numpy()
            #                 formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv[:-1]])
            #                 if 'demand_signals' in observation['internal_data']:
            #                     if observation['current_period'] + 1 >= observation['internal_data']['demand_signals'].size(2):
            #                         demand_signal = torch.zeros_like(observation['internal_data']['demand_signals'][:, :, observation['current_period']])
            #                     else:
            #                         demand_signal = observation['internal_data']['demand_signals'][:, :, observation['current_period'] + 1]
            #                     f.write(f"[{formatted_store_inv}] - {demand_signal[sample_idx, store_idx].detach().cpu().numpy()} - {observation['underage_costs'][sample_idx, store_idx].detach().cpu().numpy()}\n")
            #                 else:
            #                     f.write(f"[{formatted_store_inv}] - {observation['underage_costs'][sample_idx, store_idx].detach().cpu().numpy()}\n")
            
            results = {
                'stores': store_allocation_matrix,
                'warehouses': warehouse_allocation
            }
            if self.apply_bottleneck_loss:
                results['bottleneck_loss'] = (store_orders - store_allocation).sum()
            if self.self_loop:
                results['warehouse_self_loop_orders'] = loop_outputs * scaling_factor
            return results
        else:
            store_intermediate_outputs = outputs[:, 1:]
            warehouse_allocation = outputs[:, :1, 0]
            if self.__class__.__name__ == 'GNN_transshipment':
                # store_allocation = self.apply_softmax_feasibility_function(store_intermediate_outputs[:,:,0], observation['warehouse_inventories'], transshipment=True)
                store_allocation = self.apply_proportional_allocation(
                    store_intermediate_outputs[:,:,0], 
                    observation['warehouse_inventories'],
                    transshipment=True,
                    soft_min=self.soft_min
                    )
            else:
                if self.self_loop:
                    allocations = self.apply_proportional_allocation(
                        torch.cat([store_intermediate_outputs[:,:,0], warehouse_loop_output[:,:,0]], dim=1), 
                        observation['warehouse_inventories'],
                        soft_min=self.soft_min
                        )
                    store_allocation = allocations[:, :-1]
                else:
                    store_allocation = self.apply_proportional_allocation(
                        store_intermediate_outputs[:,:,0], 
                        observation['warehouse_inventories'],
                        soft_min=self.soft_min
                        )
                
            # if self.is_debugging:
            #     debug_dir = "/user/ml4723/Prj/NIC/debug"
            #     if self.debug_identifier is not None:
            #         debug_dir = debug_dir + self.debug_identifier
            #     os.makedirs(debug_dir, exist_ok=True)
            #     for sample_idx in range(min(store_allocation.size(0), 8)):
            #         with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
            #             f.write("\n\n")
            #             warehouse_outputs = warehouse_allocation[sample_idx].detach().cpu().to(torch.float32).numpy()
            #             f.write(np.array2string(warehouse_outputs, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
            #             f.write('\n')
                        
            #             max_lead_time = int(observation['warehouse_lead_times'][sample_idx].max().item())
            #             for lead_time in range(max_lead_time, 0, -1):
            #                 outstanding_orders = observation['warehouse_inventories'][sample_idx, :, lead_time - 1].detach().cpu().to(torch.float32).numpy()
            #                 f.write(np.array2string(outstanding_orders, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
            #                 f.write('\n')

            #             for store_idx in range(self.n_stores):
            #                 store_alloc = store_allocation[sample_idx, store_idx].detach().cpu().to(torch.float32).numpy()
            #                 store_out = store_intermediate_outputs[sample_idx, store_idx, 0].detach().cpu().to(torch.float32).numpy()
            #                 formatted_allocation_orders = '[' + ', \t'.join([f'{store_alloc:4.1f}/{store_out:4.1f}']) + ']'
            #                 f.write(formatted_allocation_orders)
                            
            #                 store_inv = observation['store_inventories'][sample_idx, store_idx].detach().cpu().to(torch.float32).numpy()
            #                 formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv[::-1]])
            #                 f.write(f"[{formatted_store_inv}]\n")
            result = {
                'stores': store_allocation,
                'warehouses': warehouse_allocation
            }
            result['stores_intermediate_outputs'] = store_intermediate_outputs[:,:,0]
            if self.self_loop:
                result['warehouse_loop_output'] = warehouse_loop_output[:,:,0]
                result['warehouse_self_loop_orders'] = allocations[:, -1:]
            if self.apply_bottleneck_loss:
                result['bottleneck_loss'] = torch.clamp(store_intermediate_outputs[:,:,0] - store_allocation, min=0).sum()
            return result

class GNN_transshipment(GNN):
    pass

class GNN_real(GNN):
    def get_store_inventory_and_params(self, observation):
        return torch.cat([observation['store_inventories'][:, :, 0].unsqueeze(-1)] \
                         + [observation[k].unsqueeze(-1) for k in ['holding_costs']] \
                         + [observation[k] for k in ['store_arrivals', 'past_demands']] \
                         + [observation[k].unsqueeze(-1) for k in ['days_from_christmas', 'underage_costs']], dim=2)

    def get_warehouse_inventory_and_params(self, observation):
        params_to_cat = ['warehouse_holding_costs']
        if 'warehouse_edge_initial_cost' in observation:
            params_to_cat.append('warehouse_edge_initial_cost')
        if 'warehouse_edge_distance_cost' in observation:
            params_to_cat.append('warehouse_edge_distance_cost')
        list_to_cat_to_unsqueeze = [observation[k].unsqueeze(-1) for k in params_to_cat]
        list_to_cat = [observation["warehouse_arrivals"]]
        return torch.cat([observation['warehouse_inventories'], *list_to_cat_to_unsqueeze, *list_to_cat], dim=-1)

class SymmetryAware(MyNeuralNetwork):
    """
    Symmetry-aware neural network for settings with one warehouse and many stores
    """
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.include_primitive_features = 'include_primitive_features' in args and args['include_primitive_features']
        self.apply_normalization = 'apply_normalization' in args and args['apply_normalization']
        self.store_orders_for_warehouse = 'store_orders_for_warehouse' in args and args['store_orders_for_warehouse']
        self.n_sub_sample_for_context = args['n_sub_sample_for_context'] if 'n_sub_sample_for_context' in args else 0
        self.omit_context_from_store_input = 'omit_context_from_store_input' in args and args['omit_context_from_store_input']

    def get_store_inventory_and_context_params(self, observation):
        return observation['store_inventories']

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'underage_costs', 'lead_times']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        store_params = torch.stack([observation[k] for k in params_to_stack], dim=2)
        return torch.cat([observation['store_inventories'], store_params], dim=2)

    def random_subsample(self, tensor):
        n_stores = tensor.size(1)
        if self.n_sub_sample_for_context > 0:
            n_samples = min(self.n_sub_sample_for_context, n_stores)
            indices = torch.randperm(n_stores)[:n_samples]
            return tensor[:, indices, :]
        return tensor

    def get_context(self, observation, store_inventory_and_params):
        if self.include_primitive_features:
            sampled_store_inventory_and_params = self.random_subsample(store_inventory_and_params)
            input_tensor = self.flatten_then_concatenate_tensors([sampled_store_inventory_and_params, observation['warehouse_inventories']])
        else:
            store_inventory_and_context_param = self.get_store_inventory_and_context_params(observation)
            sampled_store_inventory_and_context_param = self.random_subsample(store_inventory_and_context_param)
            input_tensor = self.flatten_then_concatenate_tensors([sampled_store_inventory_and_context_param, observation['warehouse_inventories']])
        
        return self.net['context'](input_tensor)

    def normalize_observation(self, observation):
        if 'past_demands' not in observation:
            return observation, None

        R = observation['past_demands'].mean()
        if R <= 0:
            R = 1e-3

        normalized_observation = observation.copy()
        for key in ['past_demands', 'arrivals', 'orders', 'store_inventories', 'warehouse_inventories']:
            if key in observation:
                normalized_observation[key] = observation[key] / R

        return normalized_observation, R

    def forward(self, observation):
        """
        Use store and warehouse inventories and output a context vector.
        Then, use the context vector alongside warehouse/store local state to output intermediate outputs for warehouses/store.
        For stores, interpret intermediate outputs as ordered, and apply proportional allocation whenever inventory is scarce.
        For warehouses, apply sigmoid to intermediate outputs and multiply by warehouse upper bound.
        """
        R = None
        if self.apply_normalization:
            observation, R = self.normalize_observation(observation)

        store_inventory_and_params = self.get_store_inventory_and_params(observation)
        if 'context' in self.net:
            context = self.get_context(observation, store_inventory_and_params)
        
        if self.omit_context_from_store_input:
            stores_input = store_inventory_and_params
        else:
            stores_input = torch.cat([store_inventory_and_params, context.unsqueeze(1).expand(-1, store_inventory_and_params.size(1), -1)], dim=-1)
        
        store_net_results = self.net['store'](stores_input)
        store_intermediate_outputs = store_net_results[:, :, 0]
        
        if self.store_orders_for_warehouse:
            warehouse_intermediate_outputs = store_net_results[:, :, 1].sum(dim=1, keepdim=True)
        else:
            if 'context' in self.net:
                warehouses_and_context = self.concatenate_signal_to_object_state_tensor(observation['warehouse_inventories'], context)
                warehouse_intermediate_outputs = self.net['warehouse'](warehouses_and_context)[:, :, 0]
            else:
                warehouse_intermediate_outputs = self.net['warehouse'](observation['warehouse_inventories'])[:, :, 0]

        if self.__class__.__name__ == 'SymmetryAwareTransshipment':
            store_allocation = self.apply_softmax_feasibility_function(store_intermediate_outputs, observation['warehouse_inventories'], transshipment=True)
        else:
            store_allocation = self.apply_proportional_allocation(
                store_intermediate_outputs, 
                observation['warehouse_inventories']
                )
        warehouse_allocation = warehouse_intermediate_outputs
        if self.warehouse_upper_bound_mult is not None:
            upper_bound = observation['mean'].sum(dim=1, keepdim=True) * self.warehouse_upper_bound_mult
            warehouse_allocation = warehouse_intermediate_outputs * upper_bound

        if R is not None:
            store_allocation = store_allocation * R
            warehouse_allocation = warehouse_allocation * R

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class Pretrained_Store(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        self.fixed_nets = {}
        self.include_context_for_warehouse_input = 'include_context_for_warehouse_input' in args and args['include_context_for_warehouse_input']
        
        store_net = SymmetryAware({
            'name': 'symmetry_aware',
            'neurons_per_hidden_layer': {
                'store': [64, 64],
                'warehouse': [32, 32],
                'context': [256]
            },
            'inner_layer_activations': {
                'store': 'elu',
                'warehouse': 'elu',
                'context': 'elu'
            },
            'output_layer_activation': {
                'store': 'softplus',
                'warehouse': 'sigmoid',
                'context': 'sigmoid'
            },
            'output_sizes': {
                'store': 1,
                'warehouse': 1,
                'context': 256
            },
            'initial_bias': None,
            'store_orders_for_warehouse': False,
            'apply_normalization': False,
            'warehouse_upper_bound_mult': 6
        }, problem_params, device)

        current_uc = int(round(problem_params['underage_cost']))
        underage_costs_to_model = {
            11: '/user/ml4723/Prj/NIC/ray_results/warehouse_varying_underage_cost/pretrained_store/run_2024-12-12_21-22-59/run_2c34f_00007_7_learning_rate=0.0010,omit_context_from_store_input=True,samples=3,store_orders_for_warehouse=False_2024-12-12_21-23-00',
        }
        model_path = f"{underage_costs_to_model[current_uc]}/model.pt"
        checkpoint = torch.load(model_path, map_location=device)  # Load to specified device
        store_net.load_state_dict(checkpoint['model_state_dict'])
        store_net = store_net.to(device)  # Move model to specified device
        # Turn off gradients for fixed nets
        for param in store_net.parameters():
            param.requires_grad = False
        self.fixed_nets[f'store'] = copy.deepcopy(store_net)
        super().__init__(args, problem_params, device)

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'underage_costs', 'lead_times']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        store_params = torch.stack([observation[k] for k in params_to_stack], dim=2)
        return torch.cat([observation['store_inventories'], store_params], dim=2)

    def forward(self, observation):
        """
        Use store and warehouse inventories and output a context vector.
        Then, use the context vector alongside warehouse/store local state to output intermediate outputs for warehouses/store.
        For stores, interpret intermediate outputs as ordered, and apply proportional allocation whenever inventory is scarce.
        For warehouses, apply sigmoid to intermediate outputs and multiply by warehouse upper bound.
        """
        store_inventory_and_params = self.get_store_inventory_and_params(observation)
        stores_input = store_inventory_and_params

        store_net_results = self.fixed_nets['store'].net['store'](stores_input)
        store_intermediate_outputs = store_net_results[:, :, 0]

        if self.include_context_for_warehouse_input:
            input_tensor = self.flatten_then_concatenate_tensors([observation['store_inventories'], observation['warehouse_inventories']])
            context = self.net['context'](input_tensor)
            warehouse_input = self.concatenate_signal_to_object_state_tensor(observation['warehouse_inventories'], context)
            warehouse_intermediate_outputs = self.net['warehouse'](warehouse_input)[:, :, 0]
        else:
            warehouse_intermediate_outputs = self.net['warehouse'](observation['warehouse_inventories'])[:, :, 0]

        store_allocation = self.apply_proportional_allocation(
            store_intermediate_outputs, 
            observation['warehouse_inventories']
            )
        warehouse_allocation = warehouse_intermediate_outputs
        if self.warehouse_upper_bound_mult is not None:
            upper_bound = observation['mean'].sum(dim=1, keepdim=True) * self.warehouse_upper_bound_mult
            warehouse_allocation = warehouse_intermediate_outputs * upper_bound

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class SymmetryAwareTransshipment(SymmetryAware):
    pass

class SymmetryAwareRealData(SymmetryAware):
    def get_store_inventory_and_context_params(self, observation):
        return torch.cat([observation['store_inventories'][:, :, 0].unsqueeze(-1)] \
             + [observation[k].unsqueeze(-1) for k in ['days_from_christmas']] \
             + [observation[k] for k in ['past_demands', 'arrivals', 'orders']], dim=2)

    def get_store_inventory_and_params(self, observation):
        return torch.cat([observation['store_inventories'][:, :, 0].unsqueeze(-1)] \
             + [observation[k].unsqueeze(-1) for k in ['days_from_christmas', 'underage_costs', 'holding_costs']] \
             + [observation[k] for k in ['past_demands', 'arrivals', 'orders']], dim=2)

class VanillaTransshipment(VanillaOneWarehouse):
    """
    Fully connected neural network for setting with one transshipment center (that cannot hold inventory) and many stores
    """

    pass

class DataDrivenNet(MyNeuralNetwork):
    """
    Fully connected neural network with optional normalization
    """
    
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.apply_normalization = 'apply_normalization' in args and args['apply_normalization']
    
    def forward(self, observation):
        R = None
        if self.apply_normalization:
            R = observation['past_demands'].mean()
            
            # Normalize quantity-related inputs
            normalized_store_inventories = observation['store_inventories'][:, :, 0] / R
            normalized_past_demands = observation['past_demands'] / R
            normalized_arrivals = observation['arrivals'] / R
            normalized_orders = observation['orders'] / R
            
            input_data = [normalized_store_inventories, normalized_past_demands, normalized_arrivals, normalized_orders]
        else:
            input_data = [observation['store_inventories'][:, :, 0], observation['past_demands'], observation['store_arrivals'], observation['store_orders']]
        
        input_data += [observation[key] for key in ['underage_costs', 'days_from_christmas']]

        if 'warehouse_inventories' in observation:
            warehouse_inventories = observation['warehouse_inventories'] / R if self.apply_normalization else observation['warehouse_inventories']
            warehouse_arrivals = observation['warehouse_arrivals'] / R if self.apply_normalization else observation['warehouse_arrivals']
            warehouse_orders = observation['warehouse_orders'] / R if self.apply_normalization else observation['warehouse_orders']
            input_data.append(warehouse_arrivals)
            input_data.append(warehouse_orders)
            input_data.append(warehouse_inventories)
        
        input_tensor = self.flatten_then_concatenate_tensors(input_data)
        outputs = self.net['master'](input_tensor)
        
        if 'warehouse_inventories' not in observation:
            return {'stores': outputs * R} if self.apply_normalization else {'stores': outputs}

        n_stores = observation['store_inventories'].size(1)
        store_intermediate_outputs, warehouse_intermediate_outputs = outputs[:, :n_stores], outputs[:, n_stores:]
        
        store_allocation = self.apply_proportional_allocation(store_intermediate_outputs, warehouse_inventories)
        warehouse_allocation = warehouse_intermediate_outputs
        
        if self.apply_normalization:
            store_allocation = store_allocation * R
            warehouse_allocation = warehouse_allocation * R
        
        return {'stores': store_allocation, 'warehouses': warehouse_allocation}

class Data_Driven_N_Warehouses(MyNeuralNetwork):
    def forward(self, observation):
        store_inventories, warehouse_inventories = observation['store_inventories'], observation['warehouse_inventories']
        n_stores = store_inventories.size(1)
        # Get input data
        input_data = [store_inventories[:, :, 0], warehouse_inventories[:, :, 0]]
        input_data += [observation[key] for key in ['past_demands', 'holding_costs', 'underage_costs', 'days_from_christmas', 'store_arrivals', 'store_orders', 'warehouse_arrivals', 'warehouse_orders']]
        input_tensor = self.flatten_then_concatenate_tensors(input_data)
        intermediate_outputs = self.net['master_n_warehouses'](input_tensor)
        n_warehouses = warehouse_inventories.size(1)
        
        edge_mask = observation['warehouse_store_edges'].transpose(1,2)
        store_intermediate_outputs = intermediate_outputs[:, n_warehouses:].reshape(intermediate_outputs.size(0), n_stores, n_warehouses)
        store_allocation = []
        for warehouse_idx in range(n_warehouses):
            connected_stores_mask = edge_mask[:, :, warehouse_idx]
            connected_store_outputs = store_intermediate_outputs[:, :, warehouse_idx] 
            connected_store_outputs = connected_store_outputs * connected_stores_mask
            allocation = self.apply_proportional_allocation(
                connected_store_outputs,
                warehouse_inventories[:, warehouse_idx:warehouse_idx+1, :]
                )
            store_allocation.append(allocation)
        store_allocation = torch.stack(store_allocation, dim=2)

        if self.is_debugging:
            debug_dir = "/user/ml4723/Prj/NIC/debug"
            if self.debug_identifier is not None:
                debug_dir = debug_dir + self.debug_identifier
            os.makedirs(debug_dir, exist_ok=True)
            for sample_idx in range(min(intermediate_outputs.size(0), 8)):
                with open(f"{debug_dir}/{sample_idx}.txt", "a") as f:
                    f.write("\n\n")
                    warehouse_outputs = intermediate_outputs[sample_idx, :n_warehouses].detach().cpu().numpy()
                    f.write(np.array2string(warehouse_outputs, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
                    f.write('\n')
                    
                    max_lead_time = int(observation['warehouse_lead_times'][sample_idx].max().item())
                    for lead_time in range(max_lead_time, 0, -1):
                        outstanding_orders = warehouse_inventories[sample_idx, :, lead_time - 1].detach().cpu().numpy()
                        f.write(np.array2string(outstanding_orders, formatter={'float_kind':lambda x: f"{x:9.1f}\t"}))
                        f.write('\n')

                    for store_idx in range(n_stores):
                        store_alloc = store_allocation[sample_idx, store_idx].detach().cpu().numpy()
                        store_out = store_intermediate_outputs[sample_idx, store_idx].detach().cpu().numpy()
                        formatted_allocation_orders = f'{store_idx} [' + ', \t'.join([f'{store_alloc[i]:4.1f}/{store_out[i]:4.1f}' for i in range(len(store_alloc))]) + ']'
                        f.write(formatted_allocation_orders)
                        
                        store_inv = store_inventories[sample_idx, store_idx].detach().cpu().numpy()
                        formatted_store_inv = ', '.join([f'{inv:.1f}' for inv in store_inv[:-1]])
                        f.write(f"[{formatted_store_inv}] - {observation['underage_costs'][sample_idx, store_idx].detach().cpu().numpy()}\n")
            
        
        warehouse_allocation = intermediate_outputs[:, :n_warehouses]
        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class TransformedNV_NoQuantile(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
        self.n_stores = problem_params['n_stores']
    
    def forward(self, observation):
        demand_mean, demand_std, underage_costs, holding_costs, store_inventories, warehouse_inventories = [
            observation[key] for key in ['mean', 'std', 'underage_costs', 'holding_costs', 'store_inventories', 'warehouse_inventories']
        ]
        
        # Calculate critical ratio for each store
        critical_ratio = underage_costs / (underage_costs + holding_costs)
        # Prepare input for store network
        store_input = torch.cat([demand_mean.unsqueeze(-1), demand_std.unsqueeze(-1), critical_ratio.unsqueeze(-1)], dim=2)
        
        # Get store base stock levels and caps
        store_output = self.net['store'](store_input)
        stores_base_stock_levels, stores_caps = store_output[:, :, 0], store_output[:, :, 1]
        
        # Calculate store allocation with cap
        uncapped_store_allocation = stores_base_stock_levels - store_inventories.sum(dim=2)
        store_intermediate_allocation = torch.min(torch.clip(uncapped_store_allocation, min=0), stores_caps)
        store_allocation = self.apply_proportional_allocation(store_intermediate_allocation, warehouse_inventories)
        
        warehouse_output = self.net['warehouse'](torch.tensor([0.0]).to(self.device))
        warehouse_base_stock_level = warehouse_output[0] * self.n_stores
        warehouse_cap = warehouse_output[1] * self.n_stores

        warehouse_pos = warehouse_inventories.sum(dim=2)
        uncapped_warehouse_allocation = warehouse_base_stock_level - warehouse_pos
        warehouse_allocation = torch.min(torch.clip(uncapped_warehouse_allocation, min=0), warehouse_cap)
        
        return {
            'stores': store_allocation,
            'warehouses': warehouse_allocation
        }

class TransformedNV_NoQuantile_SeparateStores(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
        self.n_stores = problem_params['n_stores']
        self.net['stores'] = nn.ModuleList([
            nn.Sequential(*[
                nn.LazyLinear(self.net['store'][i].out_features)
                if isinstance(self.net['store'][i], nn.Linear)
                else self.net['store'][i].__class__()
                for i in range(len(self.net['store']))
            ])
            for _ in range(self.n_stores)
        ])
    
    def forward(self, observation):
        store_inventories, warehouse_inventories = [
            observation[key] for key in ['store_inventories', 'warehouse_inventories']
        ]
        n_samples = store_inventories.shape[0]
        stores_output = torch.stack([self.net['stores'][i](torch.tensor([0.0]).to(self.device)).repeat(n_samples, 1) for i in range(self.n_stores)])
        stores_output = stores_output.permute(1, 0, 2)  # Reshape to (n_samples, n_stores, 2)
        stores_base_stock_levels, stores_caps = stores_output[:, :, 0], stores_output[:, :, 1]
        
        # Calculate store allocation with cap
        uncapped_store_allocation = stores_base_stock_levels - store_inventories.sum(dim=2)
        store_intermediate_allocation = torch.min(torch.clip(uncapped_store_allocation, min=0), stores_caps)
        store_allocation = self.apply_proportional_allocation(store_intermediate_allocation, warehouse_inventories)
        
        warehouse_output = self.net['warehouse'](torch.tensor([0.0]).to(self.device))
        warehouse_base_stock_level = warehouse_output[0]
        warehouse_cap = warehouse_output[1]

        warehouse_pos = warehouse_inventories.sum(dim=2)
        uncapped_warehouse_allocation = warehouse_base_stock_level - warehouse_pos
        warehouse_allocation = torch.min(torch.clip(uncapped_warehouse_allocation, min=0), warehouse_cap)
        
        return {
            'stores': store_allocation,
            'warehouses': warehouse_allocation
        }

class TransformedNV_CalculatedQuantile(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
    
    def forward(self, observation):
        demand_mean, demand_std, underage_costs, holding_costs, store_inventories, warehouse_inventories = [
            observation[key] for key in ['mean', 'std', 'underage_costs', 'holding_costs', 'store_inventories', 'warehouse_inventories']
        ]
        critical_ratio = underage_costs / (underage_costs + holding_costs)
        
        store_quantiles = self.net['store'](critical_ratio.unsqueeze(-1)).squeeze(-1)
        # Clip store_quantiles to avoid extreme values
        store_quantiles = torch.clamp(store_quantiles, max=1-1e-7)
        stores_base_stock_levels = demand_mean + demand_std * torch.erfinv(2 * store_quantiles - 1) * math.sqrt(2)
        
        uncapped_store_allocation = stores_base_stock_levels - store_inventories.sum(dim=2)
        store_allocation = torch.clip(uncapped_store_allocation, min=0)
        store_allocation = self.apply_proportional_allocation(store_allocation, warehouse_inventories)
        
        warehouse_output = self.net['warehouse'](torch.tensor([0.0]).to(self.device))
        warehouse_base_stock_level = warehouse_output[0]
        warehouse_cap = warehouse_output[1]

        warehouse_pos = warehouse_inventories.sum(dim=2)
        uncapped_warehouse_allocation = warehouse_base_stock_level - warehouse_pos
        warehouse_allocation = torch.min(torch.clip(uncapped_warehouse_allocation, min=0), warehouse_cap)
        
        return {
            'stores': store_allocation,
            'warehouses': warehouse_allocation
        }

class QuantilePolicy(MyNeuralNetwork):
    """
    Base class for quantile policies.
    These policies rely on mappings from features to desired quantiles, and then "invert" the quantiles using a 
    quantile forecaster to get base-stock levels.
    """

    def __init__(self, args, problem_params, device='cpu'):

        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
        self.fixed_nets = {'quantile_forecaster': self.load_forecaster(args, requires_grad=False), 'long_quantile_forecaster': self.load_long_forecaster(args, requires_grad=False)}
        self.allow_back_orders = False  # We will set to True only for non-admissible ReturnsNV policy
        self.n_stores = problem_params['n_stores']
        self.warehouse_lead_time = 6  # warehouse lead time

    def load_forecaster(self, nn_params, requires_grad=True):
        """"
        Create quantile forecaster and load weights from file
        """
        quantile_forecaster = FullyConnectedForecaster([128, 128], lead_times=nn_params['forecaster_lead_times'], qs=np.arange(0.05, 1, 0.05))
        quantile_forecaster = quantile_forecaster
        quantile_forecaster.load_state_dict(torch.load(f"{nn_params['forecaster_location']}"))
        
        # Set requires_grad to False for all parameters if we are not training the forecaster
        for p in quantile_forecaster.parameters():
            p.requires_grad_(requires_grad)
        return quantile_forecaster.to(self.device)
    
    def load_long_forecaster(self, nn_params, requires_grad=True):
        """"
        Create quantile forecaster and load weights from file
        """
        quantile_forecaster = FullyConnectedForecaster([128, 128], lead_times=nn_params['long_forecaster_lead_times'], qs=np.arange(0.05, 1, 0.05))
        quantile_forecaster = quantile_forecaster
        quantile_forecaster.load_state_dict(torch.load(f"{nn_params['long_forecaster_location']}"))
        
        # Set requires_grad to False for all parameters if we are not training the forecaster
        for p in quantile_forecaster.parameters():
            p.requires_grad_(requires_grad)
        return quantile_forecaster.to(self.device)
    
    def forecast_base_stock_allocation(self, quantile_forecaster, past_demands, days_from_christmas, store_inventories, lead_times, quantiles, allow_back_orders=False):
        """"
        Get store allocation by mapping the quantiles to base stock levels with the use of the quantile forecaster
        """

        base_stock_levels = \
            quantile_forecaster.get_quantile(
                torch.cat([
                    past_demands, 
                    # for warehouse..
                    days_from_christmas.unsqueeze(-1)
                    # days_from_christmas.unsqueeze(1).expand(past_demands.shape[0], past_demands.shape[1], 1)
                    ], dim=2
                    ), 
                    quantiles, 
                    lead_times
                    )

        # If we allow back orders, then we don't clip at zero from below
        if allow_back_orders:
            store_allocation = base_stock_levels - store_inventories.sum(dim=2)
        else:
            # store_allocation = torch.clip(base_stock_levels - store_inventories.sum(dim=2), min=0)
            # for warehouse..
            store_allocation = torch.clip(base_stock_levels - store_inventories.sum(dim=2, keepdim=True), min=0)


        # return base_stock_levels, {"stores": store_allocation}
        # for warehouse..
        return base_stock_levels, {"stores": store_allocation.squeeze(-1)}
    
    def compute_desired_quantiles(self, args):

        raise NotImplementedError
    
    def forward(self, observation):
        try:
            """
            Get store allocation by mapping features to quantiles for each store.
            Then, with the quantile forecaster, we "invert" the quantiles to get base-stock levels and obtain the store allocation.
            """

            underage_costs, holding_costs, lead_times, past_demands, days_from_christmas, store_inventories = [observation[key] for key in ['underage_costs', 'holding_costs', 'lead_times', 'past_demands', 'days_from_christmas', 'store_inventories']]

            # Get the desired quantiles for each store, which will be used to forecast the base stock levels
            # This function is different for each type of QuantilePolicy
            # quantiles = self.compute_desired_quantiles({'underage_costs': underage_costs, 'holding_costs': holding_costs})
            # for warehouse..
            quantiles = self.compute_desired_quantiles({'underage_costs': underage_costs.unsqueeze(-1), 'holding_costs': holding_costs.unsqueeze(-1)})

            # Get store allocation by mapping the quantiles to base stock levels with the use of the quantile forecaster
            stores_base_stock_levels, result = self.forecast_base_stock_allocation(
                self.fixed_nets['quantile_forecaster'], past_demands, days_from_christmas, store_inventories, lead_times, quantiles, allow_back_orders=self.allow_back_orders
                )
            if 'warehouse_inventories' not in observation:
                return result

            warehouse_inventories = observation['warehouse_inventories']
            result['stores'] = self.apply_proportional_allocation(result['stores'], warehouse_inventories)
            long_desired_quantiles = self.net['long_desired_quantiles'](underage_costs.unsqueeze(-1)/(underage_costs.unsqueeze(-1) + holding_costs.unsqueeze(-1)))

            # # Get base stock levels for longer horizon using long quantile forecaster
            stores_long_base_stock_levels, _ = self.forecast_base_stock_allocation(
                self.fixed_nets['long_quantile_forecaster'], 
                past_demands,
                days_from_christmas, 
                store_inventories,
                lead_times + self.warehouse_lead_time,
                long_desired_quantiles,
                allow_back_orders=self.allow_back_orders
            )
            warehouse_base_stock_level = stores_long_base_stock_levels.sum(dim=1)
            warehouse_pos = warehouse_inventories.sum(dim=2) + store_inventories.sum(dim=2).sum(dim=1, keepdim=True)
            result['warehouses'] = torch.clip(warehouse_base_stock_level - warehouse_pos, min=0)
            return result
        except Exception as e:
            with open(f"/user/ml4723/Prj/NIC/error_log.txt", "a") as f:
                f.write(f"Process ID: {os.getpid()}\n")
                f.write(f"Error in forward method: {str(e)}\n")
                import traceback
                f.write(f"Traceback:\n{traceback.format_exc()}\n")
                f.write(f"Call stack:\n{''.join(traceback.format_stack())}\n")
                f.write("-" * 50 + "\n")
                raise e
        
class TransformedNV(QuantilePolicy):

    def compute_desired_quantiles(self, args):
        """"
        Maps the newsvendor quantile (u/[u+h]) to a new quantile
        """
        return self.net['master'](args['underage_costs']/(args['underage_costs'] + args['holding_costs']))

class QuantileNV(QuantilePolicy):

    def __init__(self, args, problem_params, device='cpu'):

        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
        self.trainable = False

    def compute_desired_quantiles(self, args):
        """"
        Returns the newsvendor quantile (u/[u+h])
        """

        return args['underage_costs']/(args['underage_costs'] + args['holding_costs'])

class ReturnsNV(QuantileNV):
    """"
    Same as QuantileNV, but allows back orders (so it is a non-admissible policy)
    """

    def __init__(self, args, problem_params, device='cpu'):

        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
        self.trainable = False
        self.allow_back_orders = True

class FixedQuantile(QuantilePolicy):

    def compute_desired_quantiles(self, args):
        """"
        Returns the same quantile for all stores and periods
        """

        return self.net['master'](torch.tensor([0.0]).to(self.device)).unsqueeze(1).expand(args['underage_costs'].shape[0], args['underage_costs'].shape[1])


class JustInTime(MyNeuralNetwork):
    """"
    Non-admissible policy, that looks into the future and orders so that units arrive just-in-time so satisfy demand
    Can be considered as an "oracle policy"
    """

    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args=args, problem_params=problem_params, device=device) # Initialize super class
        self.trainable = False

    def forward(self, observation):
        """
        Get store allocation by looking into the future and ordering so that units arrive just-in-time to satisfy demand
        """
        current_period = observation['current_period']
        lead_times = observation['lead_times']
        if 'warehouse_store_edge_lead_times' in observation:
            lead_times = torch.ones_like(observation['lead_times'])
        demands, period_shift = self.unpack_args(observation['internal_data'], ["demands", "period_shift"])
    
        num_samples, num_stores, max_lead_time = demands.shape

        stores_future_demands = torch.stack([
            demands[:, j][
                torch.arange(num_samples), 
                torch.clip(current_period.to(self.device) + period_shift + lead_times[:, j].long(), max=max_lead_time - 1)
                ] 
            for j in range(num_stores)
            ], dim=1
            )

        if 'warehouse_lead_times' in observation:
            warehouse_lead_times = self.unpack_args(observation, ["warehouse_lead_times"])
            if 'warehouse_store_edge_lead_times' in observation:
                n_warehouses = observation['warehouse_inventories'].size(1)
                warehouse_store_edges_observation = observation['warehouse_store_edges']
                
                # Handle multi-connected stores by selecting cheapest warehouse
                store_connections = warehouse_store_edges_observation[0].sum(dim=0)
                multi_connected_stores = (store_connections > 1).nonzero().squeeze(-1)
                for store_idx in multi_connected_stores:
                    connected_warehouses = warehouse_store_edges_observation[0, :, store_idx].bool()
                    costs = observation['warehouse_edge_initial_cost'][0]
                    costs_connected = costs[connected_warehouses]
                    min_cost_idx = costs_connected.argmin(dim=-1)
                    connected_warehouse_indices = connected_warehouses.nonzero().squeeze(-1)
                    connection_index = connected_warehouse_indices[min_cost_idx]
                    warehouse_store_edges_observation[:, :, store_idx] = 0
                    warehouse_store_edges_observation[:, connection_index, store_idx] = 1
                
                # Create store allocation matrix with shape (n_samples, n_stores, n_warehouses)
                store_allocation = torch.zeros(num_samples, num_stores, n_warehouses, device=self.device)
                
                # For each store, allocate its future demand to the connected warehouse
                for store_idx in range(num_stores):
                    # Skip if current period is shorter than warehouse's lead time
                    if current_period < warehouse_lead_times[0, 0].long().cpu():
                        continue
                    # Find which warehouse is connected to this store
                    connected_warehouse = warehouse_store_edges_observation[0, :, store_idx].nonzero().squeeze(-1)
                    if connected_warehouse.numel() > 0:  # If there is a connected warehouse
                        warehouse_idx = connected_warehouse.item()
                        store_allocation[:, store_idx, warehouse_idx] = stores_future_demands[:, store_idx]
                
                # Calculate future demands for each warehouse based on connected stores
                warehouse_future_demands = torch.zeros(num_samples, n_warehouses, device=self.device)
                
                for warehouse_idx in range(n_warehouses):
                    # Get stores connected to this warehouse
                    connected_stores = warehouse_store_edges_observation[0, warehouse_idx].bool()
                    connected_store_indices = connected_stores.nonzero().squeeze(-1)
                    
                    # For each connected store, calculate its future demand
                    for store_idx in connected_store_indices:
                        store_lead_time = lead_times[:, store_idx].long()
                        warehouse_lead_time = warehouse_lead_times[:, warehouse_idx].long()
                        total_lead_time = store_lead_time + warehouse_lead_time + period_shift
                        
                        future_period = torch.clip(
                            current_period.to(self.device) + total_lead_time, 
                            max=max_lead_time - 1
                        )
                        
                        store_future_demand = demands[:, store_idx][torch.arange(num_samples), future_period]
                        warehouse_future_demands[:, warehouse_idx] += store_future_demand
                
                # Return store allocation and warehouse future demands
                return {'stores': store_allocation, 'warehouses': torch.clip(warehouse_future_demands, min=0)}
            else:
                warehouse_future_demands = torch.stack([
                    demands[:, j][
                        torch.arange(num_samples), 
                        torch.clip(current_period.to(self.device) + period_shift + warehouse_lead_times[:, 0].long() + lead_times[:, j].long(), max=max_lead_time - 1)
                        ] 
                    for j in range(num_stores)
                    ], dim=1
                    )
                warehouse_future_demands = warehouse_future_demands.sum(dim=1).unsqueeze(-1)
                store_allocation = self.apply_proportional_allocation(torch.clip(stores_future_demands, min=0), observation['warehouse_inventories'])
                return {'stores': store_allocation,'warehouses': torch.clip(warehouse_future_demands, min=0),}
        return {'stores': torch.clip(stores_future_demands, min=0)}

class WeeklyForecastNN(MyNeuralNetwork):
    """
    Base class for ...
    """

    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args=args, problem_params=problem_params, device=device) # initialize super class
        self.fixed_nets = {'quantile_forecaster': self.load_forecaster(args, requires_grad=False)}
        self.allow_back_orders = False  # we will set to True only for non-admissible ReturnsNV policy
        
    def load_forecaster(self, nn_params, requires_grad=True):
        """"
        Create quantile forecaster and load weights from file
        """
        quantile_forecaster = FullyConnectedForecaster([128, 128], lead_times=nn_params['forecaster_lead_times'], qs=np.array([0.5]))
        quantile_forecaster = quantile_forecaster
        quantile_forecaster.load_state_dict(torch.load(f"{nn_params['forecaster_location']}"))
        
        # Set requires_grad to False for all parameters if we are not training the forecaster
        for p in quantile_forecaster.parameters():
            p.requires_grad_(requires_grad)
        return quantile_forecaster.to(self.device)
    
    def interpolate_tensor(self, x, n):
        # Calculate the indices
        lower_index = torch.floor(n).long()  # This gives the lower index (e.g., 4 for n = 4.7)
        upper_index = torch.ceil(n).long()   # This gives the upper index (e.g., 5 for n = 4.7)
        
        # Calculate the weights for interpolation
        upper_weight = n - lower_index   # Weight for the upper index (e.g., 0.7 for n = 4.7)
        lower_weight = 1 - upper_weight  # Weight for the lower index (e.g., 0.3 for n = 4.7)
        
        # Extract the values at the lower and upper indices
        lower_value = x[:, :, lower_index]
        upper_value = x[:, :, upper_index]
        
        # Perform the linear interpolation
        y = lower_weight * lower_value + upper_weight * upper_value
        
        return y[:, :, 0]
    
    def forward(self, observation):
        """
        Get store allocation by mapping features to quantiles for each store.
        Then, with the quantile forecaster, we "invert" the quantiles to get base-stock levels and obtain the store allocation.
        """

        underage_costs, holding_costs, lead_times, past_demands, days_from_christmas, store_inventories = [observation[key] for key in ['underage_costs', 'holding_costs', 'lead_times', 'past_demands', 'days_from_christmas', 'store_inventories']]
        forecaster_output = self.fixed_nets['quantile_forecaster'](torch.cat([past_demands, days_from_christmas.unsqueeze(1).expand(past_demands.shape[0], past_demands.shape[1], 1)], dim=2))
        forecaster_output = forecaster_output[:, :, 0]
        # print(f'forecaster_output: {forecaster_output[0]}')
        zero_to_one = self.net['master'](torch.tensor([0.0]).to(self.device))  # constant base stock level
        pos = zero_to_one*9  # indexes go from 0 to 9, so we interpret output as proportion from 0 to 9
        base_level = self.interpolate_tensor(forecaster_output, pos)
        # print(f'pos: {pos}')
        # print(f'base_level: {base_level[0]}')
        # print(f'inventory_position: {store_inventories.sum(dim=2)[0]}')

        # calculate allocation using base level
        store_allocation = torch.clip(base_level - store_inventories.sum(dim=2), min=0)
        # print(f'store_allocation: {store_allocation[0]}')
        # print()

        # Get store allocation by mapping the quantiles to base stock levels with the use of the quantile forecaster
        return {"stores": store_allocation}
    
    def forecast_base_stock_allocation(self, past_demands, days_from_christmas, store_inventories, lead_times, quantiles, allow_back_orders=False):
        """"
        Get store allocation by mapping the quantiles to base stock levels with the use of the quantile forecaster
        """
        net_output = self.net['master'](torch.cat([past_demands, days_from_christmas.unsqueeze(1).expand(past_demands.shape[0], past_demands.shape[1], 1)], dim=2))

class NeuralNetworkCreator:
    """
    Class to create neural networks
    """

    def set_default_output_size(self, module_name, problem_params):
        
        default_sizes = {
            'master': problem_params['n_stores'] + problem_params['n_warehouses'], 
            'master_cbs': problem_params['n_stores'] * 2 + problem_params['n_warehouses'],
            'master_echelon': problem_params['n_stores'] + problem_params['n_warehouses'] + problem_params['n_extra_echelons'],
            'master_echelon_selfloop': problem_params['n_stores'] + 2 * (problem_params['n_warehouses'] + problem_params['n_extra_echelons']),
            'master_selfloop': problem_params['n_stores'] + problem_params['n_warehouses'] * 2,
            'store': 1, 
            'warehouse': 1, 
            'master_n_warehouses': problem_params['n_warehouses'] + problem_params['n_warehouses'] * problem_params['n_stores'],
            'master_n_warehouses_selfloop': problem_params['n_warehouses'] * 2 + problem_params['n_warehouses'] * problem_params['n_stores'],
            'context': None
            }
        return default_sizes[module_name]

    def get_architecture(self, name):

        architectures = {
            'vanilla_one_store': VanillaOneStore, 
            'vanilla_one_store_for_warehouse': VanillaOneStoreForWarehouse,
            'base_stock': BaseStock,
            'base_stock_distribution': BaseStockDistribution,
            'capped_base_stock': CappedBaseStock,
            'echelon_stock': EchelonStock,
            'vanilla_serial': VanillaSerial,
            'VanillaSerialSelfloop': VanillaSerialSelfloop,
            'vanilla_transshipment': VanillaTransshipment,
            'VanillaTransshipmentSelfloop': VanillaTransshipmentSelfloop,
            'vanilla_one_warehouse': VanillaOneWarehouse,
            'VanillaOneWarehouseSelfloop': VanillaOneWarehouseSelfloop,
            'vanilla_n_stores': Vanilla_N_Stores,
            'n_stores_shared_net': N_Stores_Shared_Net,
            'n_stores_per_store_net': N_Stores_Per_Store_Net,
            'vanilla_n_warehouses': Vanilla_N_Warehouses,
            'Vanilla_N_Warehouses_Selfloop': Vanilla_N_Warehouses_Selfloop,
            'symmetry_aware': SymmetryAware,
            'symmetry_aware_real_data': SymmetryAwareRealData,
            'data_driven': DataDrivenNet,
            'data_driven_n_warehouses': Data_Driven_N_Warehouses,
            'transformed_nv': TransformedNV,
            'fixed_quantile': FixedQuantile,
            'quantile_nv': QuantileNV,
            'returns_nv': ReturnsNV,
            'just_in_time': JustInTime,
            'symmetry_aware_transshipment': SymmetryAwareTransshipment,
            'weekly_forecast_NN': WeeklyForecastNN,
            'transformed_nv_noquantile': TransformedNV_NoQuantile,
            'transformed_nv_calculated_quantile': TransformedNV_CalculatedQuantile,
            'transformed_nv_noquantile_sep_stores': TransformedNV_NoQuantile_SeparateStores,
            'pretrained_store': Pretrained_Store,
            'CBS_One_Warehouse': CBS_One_Warehouse,
            'GNN': GNN,
            'GNN_real': GNN_real,
            'GNN_decentralized': GNN_decentralized,
            'GNN_decentralized_transshipment': GNN_decentralized_transshipment,
            'GNN_transshipment': GNN_transshipment,
            }
        return architectures[name]
    
    def create_neural_network(self, problem_params, nn_params, device='cpu'):

        nn_params_copy = copy.deepcopy(nn_params)

        # If not specified in config file, set output size to default value
        for key, val in nn_params_copy['output_sizes'].items():
            if val is None:
                nn_params_copy['output_sizes'][key] = self.set_default_output_size(key, problem_params)

        model = self.get_architecture(nn_params_copy['name'])(
            nn_params_copy, 
            problem_params,
            device=device
            )
        
        # Calculate warehouse upper bound if specified in config file
        if 'warehouse_upper_bound_mult' in nn_params.keys():
            model.warehouse_upper_bound_mult = nn_params['warehouse_upper_bound_mult']
        
        return model.to(device)
