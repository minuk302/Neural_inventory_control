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
    
    def apply_proportional_allocation(self, store_intermediate_outputs, warehouse_inventories):
        """
        Apply proportional allocation feasibility enforcement function to store intermediate outputs.
        It assigns inventory proportionally to the store order quantities, whenever inventory at the
        warehouse is not sufficient.
        """

        total_limiting_inventory = warehouse_inventories[:, 0, 0]  # Total inventory at the warehouse
        sum_allocation = store_intermediate_outputs.sum(dim=1)  # Sum of all store order quantities

        # Multiply current allocation by minimum between inventory/orders and 1
        final_allocation = \
            torch.multiply(store_intermediate_outputs,
                           torch.clip(total_limiting_inventory / (sum_allocation + 0.000000000000001), max=1)[:, None])
        return final_allocation
    
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

        return torch.multiply(
            softmax_outputs, 
            total_warehouse_inv[:, None]
            )

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
        
        x = self.activation_functions['softplus'](self.net['master'](torch.tensor([0.0]).to(self.device)) + 10.0)  # Constant base stock levels
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
        x = self.net['master'](torch.tensor(input_tensor).to(self.device))  # Constant base stock levels
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

class VanillaSerial_proportional(MyNeuralNetwork):
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
        x = self.net['master'](torch.tensor(input_tensor).to(self.device))  # Constant base stock levels
        # print(f'self.warehouse_upper_bound: {self.warehouse_upper_bound}')
        # assert False

        # Tensor with the inventory on hand for the preceding location for each location k
        # For the left-most location, we set it to an upper bound (same as warehouse upper bound). Currently 4 times mean demand.
        shifted_inv_on_hand = torch.concat((
            #self.warehouse_upper_bound.unsqueeze(1).expand(echelon_inventories.shape[0], -1),
            torch.zeros(echelon_inventories.size(0), 1, device=self.device),
            echelon_inventories[:, :, 0], 
            warehouse_inventories[:, :, 0]), 
            dim=1
            )
        # print(f'x: {x.shape}')
        # print(f'shifted_inv_on_hand: {shifted_inv_on_hand.shape}')
        # print(f"self.activation_functions['sigmoid'](x): {self.activation_functions['sigmoid'](x)[0]}")
        allocations = torch.min(x, shifted_inv_on_hand)
        allocations[:, 0] = x[:, 0]
        # print(f'allocations[0]: {allocations[0]}')


        return {
            'stores': allocations[:, -1:],
            'warehouses': allocations[:, -2: -1],
            'echelons': allocations[:, : n_extra_echelons],
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

        return {
            'stores': store_allocation, 
            'warehouses': warehouse_allocation
            }

class GNN_MP(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.use_attention = 'use_attention' in args and args['use_attention']
        self.n_stores = problem_params['n_stores']
        self.pna_delta = (torch.log(torch.tensor(self.n_stores + 1, device=self.device)) 
                          + self.n_stores * torch.log(torch.tensor(2, device=self.device))) \
                          / torch.tensor(self.n_stores + 1, device=self.device)
        self.use_pna = 'use_pna' in args and args['use_pna']
        self.NN_per_layer = 'NN_per_layer' in args and args['NN_per_layer']
        self.skip_connection = 'skip_connection' in args and args['skip_connection']
        self.use_node_for_skip_connection = 'use_node_for_skip_connection' in args and args['use_node_for_skip_connection']

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'underage_costs', 'lead_times']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        store_params = torch.stack([observation[k] for k in params_to_stack], dim=2)
        return torch.cat([observation['store_inventories'], store_params], dim=2)

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
        if 'echelon_inventories' in observation:
            echelon_inventories = observation['echelon_inventories']
            warehouse_inventories = observation['warehouse_inventories']
            echelon_inv_len = echelon_inventories.size(2)

            store_state = torch.cat([observation['store_inventories'], observation['holding_costs'].unsqueeze(2), observation['lead_times'].unsqueeze(2)], dim=-1)
            warehouse_state = torch.cat([warehouse_inventories, observation['warehouse_holding_costs'].unsqueeze(2), observation['warehouse_lead_times'].unsqueeze(2)], dim=-1)
            echelon_state = torch.cat([echelon_inventories, observation['echelon_holding_costs'].unsqueeze(2), observation['echelon_lead_times'].unsqueeze(2)], dim=-1)
    
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

            face_encoding = torch.zeros(*states.shape[:-1], 2, device=self.device)
            face_encoding[:, :1, 0] = 1  
            face_encoding[:, -1:, 1] = 1

            n_MP = 1 + n_echelons
        else:
            store_state = self.get_store_inventory_and_params(observation)
            warehouse_state = observation['warehouse_inventories']
            max_inv_len = max(store_inv_len, warehouse_inv_len)

            store_primitives_len = store_state.size(2) - store_inv_len
            warehouse_primitives_len = warehouse_state.size(2) - warehouse_inv_len
            max_primitives_len = max(store_primitives_len, warehouse_primitives_len)

            store_padded = pad_features(store_state, store_inv_len, max_inv_len, max_primitives_len)
            warehouse_padded = pad_features(warehouse_state, warehouse_inv_len, max_inv_len, max_primitives_len)

            states = torch.cat([warehouse_padded, store_padded], dim=1)

            face_encoding = torch.zeros(*states.shape[:-1], 2, device=self.device)
            face_encoding[:, :1, 0] = 1 # warehouse, supply-facing
            face_encoding[:, 1:, 1] = 1 # stores, demand-facing

            n_MP = 2
        
        states = torch.cat([states, face_encoding], dim=-1)
        initial_nodes = self.net['initial_embedding'](states)
        nodes = initial_nodes
        for layer_idx in range(n_MP):
            embedded_nodes = self.get_network('node_embedding', layer_idx)(nodes)
            if 'echelon_inventories' in observation:
                zero_node = torch.zeros_like(embedded_nodes[:, :1])
                update_input = torch.cat([
                    nodes,
                    torch.cat([zero_node, embedded_nodes[:, :-1]], dim=1), 
                    torch.cat([embedded_nodes[:, 1:], zero_node], dim=1)
                ], dim=-1)
            else:
                store_supplier_aggregation = embedded_nodes[:, :1].expand(-1, store_state.size(1), -1)
                store_recipient_aggregation = torch.zeros_like(embedded_nodes[:, 1:])
                
                warehouse_supplier_aggregation = torch.zeros_like(embedded_nodes[:, :1])
                if self.use_attention:
                    attention_input = torch.cat([embedded_nodes[:, :1].expand(-1, store_state.size(1), -1),embedded_nodes[:, 1:]], dim = -1)
                    attention_weights = self.net['attention'](attention_input)
                    attention_weights = F.softmax(attention_weights, dim=2)
                    warehouse_recipient_aggregation = torch.sum(attention_weights * embedded_nodes[:, 1:], dim=1, keepdim=True)
                else:
                    warehouse_recipient_aggregation = embedded_nodes[:, 1:].mean(dim=1, keepdim=True)
                update_input = torch.cat(
                    [torch.cat([nodes[:, :1], warehouse_supplier_aggregation, warehouse_recipient_aggregation], dim=-1),
                     torch.cat([nodes[:, 1:], store_supplier_aggregation, store_recipient_aggregation], dim=-1)],
                    dim=1
                )

            updates = self.get_network('update_embedding', layer_idx)(update_input)
            nodes = nodes + updates

            # if self.use_attention:
            #     def construct_target_nodes_and_attention_input(self_nodes, supplier_nodes, recipient_nodes):
            #         self_node_padded = F.pad(self_nodes, (0,2)).unsqueeze(2)
            #         supplier_padded = F.pad(supplier_nodes, (0,2))
            #         supplier_padded[:, :, :, -2] = 1
            #         recipient_padded = F.pad(recipient_nodes, (0,2))
            #         recipient_padded[:, :, :, -1] = 1
            #         target_nodes = torch.cat([self_node_padded, supplier_padded, recipient_padded], dim=2)
                
            #         expanded_self_nodes = self_nodes.unsqueeze(2).expand(-1, -1, target_nodes.size(2), -1)
            #         attention_input = torch.cat([expanded_self_nodes, target_nodes], dim=-1)
            #         return target_nodes, attention_input

            #     zero_node = torch.zeros_like(embedded_nodes[:, :1])
            #     if 'echelon_inventories' in observation: # serial system
            #         target_nodes, attention_input = construct_target_nodes_and_attention_input(
            #             embedded_nodes,
            #             torch.cat([zero_node, embedded_nodes[:, :-1]], dim=1).unsqueeze(2),
            #             torch.cat([embedded_nodes[:, 1:], zero_node], dim=1).unsqueeze(2),
            #         )
            #     else:
            #         warehouse_target_nodes, warehouse_attention_input = construct_target_nodes_and_attention_input(
            #             embedded_nodes[:, :1],
            #             zero_node.unsqueeze(1),
            #             embedded_nodes[:, 1:].unsqueeze(1),
            #         )

            #         store_target_nodes, store_attention_input = construct_target_nodes_and_attention_input(
            #             embedded_nodes[:, 1:],
            #             embedded_nodes[:, :1].expand(-1, store_state.size(1), -1).unsqueeze(2),
            #             zero_node.expand(-1, store_state.size(1), -1).unsqueeze(2),
            #         )

            #     attention_weights = self.net['attention'](attention_input)  # [samples, nodes, nodes, 1]
            #     attention_weights = F.softmax(attention_weights, dim=2)  # Normalize over target nodes
                
            #     # Apply attention weights to aggregate nodes
            #     update_input = torch.sum(attention_weights * target_nodes, dim=2)  # [samples, nodes, embedding]
            #     nodes = self.net['update_embedding'](update_input)
            # elif self.use_pna:
            #     aggregators = [
            #         lambda x: x.mean(dim=1),
            #         lambda x: x.min(dim=1)[0],
            #         lambda x: x.max(dim=1)[0],
            #         lambda x: x.std(dim=1) if x.size(1) > 1 else torch.zeros_like(x[:,0])
            #     ]
            #     aggregated_stores = torch.stack([agg(store_nodes_embedded) for agg in aggregators], dim=1)
            #     aggregated_warehouse = torch.stack([agg(warehouse_node_embedded) for agg in aggregators], dim=1)
                
            #     scalers = [
            #         lambda x: x,  # identity
            #         lambda x: x * (torch.log(torch.tensor(store_inventory_and_params.size(1) + 1, device=self.device)) / self.pna_delta),  # amplification
            #         lambda x: x * (self.pna_delta / torch.log(torch.tensor(store_inventory_and_params.size(1) + 1, device=self.device)))  # attenuation
            #     ]
            #     scaled_stores = torch.cat([scale(aggregated_stores) for scale in scalers], dim=1)
            #     scaled_warehouse = torch.cat([scale(aggregated_warehouse) for scale in scalers], dim=1)
                
            #     store_aggregation = scaled_warehouse.unsqueeze(1).expand(-1, store_nodes.size(1), -1, -1).view(scaled_warehouse.size(0),self.n_stores,-1)
            #     warehouse_aggregation = scaled_stores.view(scaled_stores.size(0),1,-1)
            # else:            # Log intermediate node states


        if self.skip_connection:
            outputs = self.net['output'](torch.cat([states, nodes], dim=-1))
        else:
            outputs = self.net['output'](nodes)

        if 'echelon_inventories' in observation:
            def proportional_minimum(x, y):
                return torch.min(x, y)
            echelon_allocations = []
            for j in range(outputs.size(1) - 2):
                if j == 0:  # First echelon
                    echelon_allocations.append(outputs[:, j:j+1, 0])
                else:
                    echelon_allocations.append(proportional_minimum(outputs[:, j:j+1, 0], echelon_inventories[:,j-1:j,0]))
            warehouse_allocation = proportional_minimum(outputs[:, -2: -1, 0], echelon_inventories[:,-1,:1])
            store_allocation = proportional_minimum(outputs[:, -1:, 0], warehouse_inventories[:,:,0])
            return {
                'stores': store_allocation,
                'warehouses': warehouse_allocation,
                'echelons': torch.cat(echelon_allocations, dim=1)
            }
        else:
            store_intermediate_outputs = outputs[:, 1:]
            warehouse_allocation = outputs[:, :1, 0]
            if self.__class__.__name__ == 'GNN_MP_transshipment':
                store_allocation = self.apply_softmax_feasibility_function(store_intermediate_outputs[:,:,0], observation['warehouse_inventories'], transshipment=True)
            else:
                store_allocation = self.apply_proportional_allocation(
                    store_intermediate_outputs[:,:,0], 
                    observation['warehouse_inventories']
                    )
            return {
                'stores': store_allocation,
                'warehouses': warehouse_allocation
            }


class GNN_MP_sigmoid(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.use_attention = 'use_attention' in args and args['use_attention']
        self.n_stores = problem_params['n_stores']
        self.pna_delta = (torch.log(torch.tensor(self.n_stores + 1, device=self.device)) 
                          + self.n_stores * torch.log(torch.tensor(2, device=self.device))) \
                          / torch.tensor(self.n_stores + 1, device=self.device)
        self.use_pna = 'use_pna' in args and args['use_pna']
        self.NN_per_layer = 'NN_per_layer' in args and args['NN_per_layer']
        self.skip_connection = 'skip_connection' in args and args['skip_connection']
        self.use_node_for_skip_connection = 'use_node_for_skip_connection' in args and args['use_node_for_skip_connection']

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'underage_costs', 'lead_times']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        store_params = torch.stack([observation[k] for k in params_to_stack], dim=2)
        return torch.cat([observation['store_inventories'], store_params], dim=2)

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
        if 'echelon_inventories' in observation:
            echelon_inventories = observation['echelon_inventories']
            warehouse_inventories = observation['warehouse_inventories']
            echelon_inv_len = echelon_inventories.size(2)

            store_state = torch.cat([observation['store_inventories'], observation['holding_costs'].unsqueeze(2), observation['lead_times'].unsqueeze(2)], dim=-1)
            warehouse_state = torch.cat([warehouse_inventories, observation['warehouse_holding_costs'].unsqueeze(2), observation['warehouse_lead_times'].unsqueeze(2)], dim=-1)
            echelon_state = torch.cat([echelon_inventories, observation['echelon_holding_costs'].unsqueeze(2), observation['echelon_lead_times'].unsqueeze(2)], dim=-1)
    
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

            face_encoding = torch.zeros(*states.shape[:-1], 2, device=self.device)
            face_encoding[:, :1, 0] = 1  
            face_encoding[:, -1:, 1] = 1

            n_MP = 1 + n_echelons
        else:
            store_state = self.get_store_inventory_and_params(observation)
            warehouse_state = observation['warehouse_inventories']
            max_inv_len = max(store_inv_len, warehouse_inv_len)

            store_primitives_len = store_state.size(2) - store_inv_len
            warehouse_primitives_len = warehouse_state.size(2) - warehouse_inv_len
            max_primitives_len = max(store_primitives_len, warehouse_primitives_len)

            store_padded = pad_features(store_state, store_inv_len, max_inv_len, max_primitives_len)
            warehouse_padded = pad_features(warehouse_state, warehouse_inv_len, max_inv_len, max_primitives_len)

            states = torch.cat([warehouse_padded, store_padded], dim=1)

            face_encoding = torch.zeros(*states.shape[:-1], 2, device=self.device)
            face_encoding[:, :1, 0] = 1 # warehouse, supply-facing
            face_encoding[:, 1:, 1] = 1 # stores, demand-facing

            n_MP = 2
        
        states = torch.cat([states, face_encoding], dim=-1)
        initial_nodes = self.net['initial_embedding'](states)
        nodes = initial_nodes
        for layer_idx in range(n_MP):
            embedded_nodes = self.get_network('node_embedding', layer_idx)(nodes)
            if 'echelon_inventories' in observation:
                zero_node = torch.zeros_like(embedded_nodes[:, :1])
                update_input = torch.cat([
                    nodes,
                    torch.cat([zero_node, embedded_nodes[:, :-1]], dim=1), 
                    torch.cat([embedded_nodes[:, 1:], zero_node], dim=1)
                ], dim=-1)
            else:
                store_supplier_aggregation = embedded_nodes[:, :1].expand(-1, store_state.size(1), -1)
                store_recipient_aggregation = torch.zeros_like(embedded_nodes[:, 1:])
                
                warehouse_supplier_aggregation = torch.zeros_like(embedded_nodes[:, :1])
                if self.use_attention:
                    attention_input = torch.cat([embedded_nodes[:, :1].expand(-1, store_state.size(1), -1),embedded_nodes[:, 1:]], dim = -1)
                    attention_weights = self.net['attention'](attention_input)
                    attention_weights = F.softmax(attention_weights, dim=2)
                    warehouse_recipient_aggregation = torch.sum(attention_weights * embedded_nodes[:, 1:], dim=1, keepdim=True)
                else:
                    warehouse_recipient_aggregation = embedded_nodes[:, 1:].mean(dim=1, keepdim=True)
                update_input = torch.cat(
                    [torch.cat([nodes[:, :1], warehouse_supplier_aggregation, warehouse_recipient_aggregation], dim=-1),
                     torch.cat([nodes[:, 1:], store_supplier_aggregation, store_recipient_aggregation], dim=-1)],
                    dim=1
                )

            updates = self.get_network('update_embedding', layer_idx)(update_input)
            nodes = nodes + updates
        if self.skip_connection:
            outputs = self.net['output'](torch.cat([states, nodes], dim=-1))
        else:
            outputs = self.net['output'](nodes)

        shifted_inv_on_hand = torch.concat((
            5.0 * self.warehouse_upper_bound_mult * torch.ones(echelon_inventories.size(0), 1, device=self.device),
            echelon_inventories[:, :, 0],
            warehouse_inventories[:, :, 0]),
            dim=1
        )
        allocations = self.activation_functions['sigmoid'](outputs[:,:,0])*shifted_inv_on_hand

        return {
            'stores': allocations[:, -1:],
            'warehouses': allocations[:, -2: -1],
            'echelons': allocations[:, : outputs.size(1)-2],
        }

class GNN_MP_noembedding(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.use_attention = 'use_attention' in args and args['use_attention']
        self.n_stores = problem_params['n_stores']
        self.pna_delta = (torch.log(torch.tensor(self.n_stores + 1, device=self.device)) 
                          + self.n_stores * torch.log(torch.tensor(2, device=self.device))) \
                          / torch.tensor(self.n_stores + 1, device=self.device)
        self.use_pna = 'use_pna' in args and args['use_pna']
        self.NN_per_layer = 'NN_per_layer' in args and args['NN_per_layer']
        self.skip_connection = 'skip_connection' in args and args['skip_connection']
        self.use_node_for_skip_connection = 'use_node_for_skip_connection' in args and args['use_node_for_skip_connection']

    def get_store_inventory_and_params(self, observation):
        params_to_stack = ['mean', 'std', 'underage_costs', 'lead_times']
        if 'store_random_yield_mean' in observation:
            params_to_stack.extend(['store_random_yield_mean', 'store_random_yield_std'])
        store_params = torch.stack([observation[k] for k in params_to_stack], dim=2)
        return torch.cat([observation['store_inventories'], store_params], dim=2)

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
        if 'echelon_inventories' in observation:
            echelon_inventories = observation['echelon_inventories']
            warehouse_inventories = observation['warehouse_inventories']
            echelon_inv_len = echelon_inventories.size(2)

            store_state = torch.cat([observation['store_inventories'], observation['holding_costs'].unsqueeze(2), observation['lead_times'].unsqueeze(2)], dim=-1)
            warehouse_state = torch.cat([warehouse_inventories, observation['warehouse_holding_costs'].unsqueeze(2), observation['warehouse_lead_times'].unsqueeze(2)], dim=-1)
            echelon_state = torch.cat([echelon_inventories, observation['echelon_holding_costs'].unsqueeze(2), observation['echelon_lead_times'].unsqueeze(2)], dim=-1)
    
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

            face_encoding = torch.zeros(*states.shape[:-1], 2, device=self.device)
            face_encoding[:, :1, 0] = 1  
            face_encoding[:, -1:, 1] = 1

            n_MP = 1 + n_echelons
        else:
            store_state = self.get_store_inventory_and_params(observation)
            warehouse_state = observation['warehouse_inventories']
            max_inv_len = max(store_inv_len, warehouse_inv_len)

            store_primitives_len = store_state.size(2) - store_inv_len
            warehouse_primitives_len = warehouse_state.size(2) - warehouse_inv_len
            max_primitives_len = max(store_primitives_len, warehouse_primitives_len)

            store_padded = pad_features(store_state, store_inv_len, max_inv_len, max_primitives_len)
            warehouse_padded = pad_features(warehouse_state, warehouse_inv_len, max_inv_len, max_primitives_len)

            states = torch.cat([warehouse_padded, store_padded], dim=1)

            face_encoding = torch.zeros(*states.shape[:-1], 2, device=self.device)
            face_encoding[:, :1, 0] = 1 # warehouse, supply-facing
            face_encoding[:, 1:, 1] = 1 # stores, demand-facing

            n_MP = 2
        
        states = torch.cat([states, face_encoding], dim=-1)
        initial_nodes = self.net['initial_embedding'](states)
        nodes = initial_nodes
        for layer_idx in range(n_MP):
            embedded_nodes = nodes# self.get_network('node_embedding', layer_idx)(nodes)
            if 'echelon_inventories' in observation:
                zero_node = torch.zeros_like(embedded_nodes[:, :1])
                update_input = torch.cat([
                    nodes,
                    torch.cat([zero_node, embedded_nodes[:, :-1]], dim=1), 
                    torch.cat([embedded_nodes[:, 1:], zero_node], dim=1)
                ], dim=-1)
            else:
                store_supplier_aggregation = embedded_nodes[:, :1].expand(-1, store_state.size(1), -1)
                store_recipient_aggregation = torch.zeros_like(embedded_nodes[:, 1:])
                
                warehouse_supplier_aggregation = torch.zeros_like(embedded_nodes[:, :1])
                if self.use_attention:
                    attention_input = torch.cat([embedded_nodes[:, :1].expand(-1, store_state.size(1), -1),embedded_nodes[:, 1:]], dim = -1)
                    attention_weights = self.net['attention'](attention_input)
                    attention_weights = F.softmax(attention_weights, dim=2)
                    warehouse_recipient_aggregation = torch.sum(attention_weights * embedded_nodes[:, 1:], dim=1, keepdim=True)
                else:
                    warehouse_recipient_aggregation = embedded_nodes[:, 1:].mean(dim=1, keepdim=True)
                update_input = torch.cat(
                    [torch.cat([nodes[:, :1], warehouse_supplier_aggregation, warehouse_recipient_aggregation], dim=-1),
                     torch.cat([nodes[:, 1:], store_supplier_aggregation, store_recipient_aggregation], dim=-1)],
                    dim=1
                )

            updates = self.get_network('update_embedding', layer_idx)(update_input)
            nodes = nodes + updates
        if self.skip_connection:
            outputs = self.net['output'](torch.cat([states, nodes], dim=-1))
        else:
            outputs = self.net['output'](nodes)
        if 'echelon_inventories' in observation:
            def proportional_minimum(x, y):
                return x * torch.minimum(y/(x + 1e-8), torch.ones_like(y))
            echelon_allocations = []
            for j in range(outputs.size(1) - 2):
                if j == 0:  # First echelon
                    echelon_allocations.append(outputs[:, j:j+1, 0])
                else:
                    echelon_allocations.append(proportional_minimum(outputs[:, j:j+1, 0], echelon_inventories[:,j-1:j,0]))
            warehouse_allocation = proportional_minimum(outputs[:, -2: -1, 0], echelon_inventories[:,-1,:1])
            store_allocation = proportional_minimum(outputs[:, -1:, 0], warehouse_inventories[:,:,0])
            return {
                'stores': store_allocation,
                'warehouses': warehouse_allocation,
                'echelons': torch.cat(echelon_allocations, dim=1)
            }
        else:
            store_intermediate_outputs = outputs[:, 1:]
            warehouse_allocation = outputs[:, :1, 0]
            if self.__class__.__name__ == 'GNN_MP_transshipment':
                store_allocation = self.apply_softmax_feasibility_function(store_intermediate_outputs[:,:,0], observation['warehouse_inventories'], transshipment=True)
            else:
                store_allocation = self.apply_proportional_allocation(
                    store_intermediate_outputs[:,:,0], 
                    observation['warehouse_inventories']
                    )
            return {
                'stores': store_allocation,
                'warehouses': warehouse_allocation
            }

class GNN_MP_real(GNN_MP):
    def get_store_inventory_and_params(self, observation):
        return torch.cat([observation['store_inventories'][:, :, 0].unsqueeze(-1)] \
             + [observation[k].unsqueeze(-1) for k in ['days_from_christmas', 'underage_costs', 'holding_costs']] \
             + [observation[k] for k in ['past_demands', 'arrivals', 'orders']], dim=2)

class GNN_MP_transshipment(GNN_MP):
    pass

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
        stores_input = store_inventory_and_params
        
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

class SymmetryGNN(SymmetryAware):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.n_stores = problem_params['n_stores']
        self.pna_delta = (torch.log(torch.tensor(self.n_stores + 1, device=self.device)) 
                          + self.n_stores * torch.log(torch.tensor(2, device=self.device))) \
                          / torch.tensor(self.n_stores + 1, device=self.device)
        self.use_pna = 'use_pna' in args and args['use_pna']
        self.use_WISTEMB = 'use_WISTEMB' in args and args['use_WISTEMB']
        self.no_aggregation = 'no_aggregation' in args and args['no_aggregation']
        self.randomize = 'randomize' in args and args['randomize']
        self.stop_gradient_at_store_embedding = 'stop_gradient_at_store_embedding' in args and args['stop_gradient_at_store_embedding']
    def get_context(self, observation, store_inventory_and_params):
        if self.use_WISTEMB:
            warehouse_inventories = observation['warehouse_inventories'].expand(-1, self.n_stores, -1)
            combined_input = torch.cat([store_inventory_and_params, warehouse_inventories], dim=-1)
            store_embeddings = self.net['store_embedding'](combined_input)
        else:
            if self.stop_gradient_at_store_embedding:
                store_embeddings = self.net['store_embedding'](store_inventory_and_params.detach())
            else:
                store_embeddings = self.net['store_embedding'](store_inventory_and_params)

        if 'attention' in self.net:
            warehouse_inventories = observation['warehouse_inventories'].expand(-1, self.n_stores, -1)
            attention_input = torch.cat([store_inventory_and_params, warehouse_inventories], dim=-1)
            attention_scores = self.net['attention'](attention_input).squeeze(-1)
            attention_weights = F.softmax(attention_scores, dim=1)
            aggregated_store_embeddings = torch.sum(store_embeddings * attention_weights.unsqueeze(-1), dim=1)
        elif self.use_pna:
            aggregators = [
                lambda x: x.mean(dim=1),
                lambda x: x.min(dim=1)[0],
                lambda x: x.max(dim=1)[0],
                lambda x: x.std(dim=1)
            ]
            aggregated = torch.stack([agg(store_embeddings) for agg in aggregators], dim=1)
            scalers = [
                lambda x: x,  # identity
                lambda x: x * (torch.log(torch.tensor(store_inventory_and_params.size(1) + 1, device=self.device)) / self.pna_delta),  # amplification
                lambda x: x * (self.pna_delta / torch.log(torch.tensor(store_inventory_and_params.size(1) + 1, device=self.device)))  # attenuation
            ]
            
            scaled = torch.cat([scale(aggregated) for scale in scalers], dim=1)
            aggregated_store_embeddings = scaled.view(scaled.size(0), -1)
        elif self.no_aggregation:
            if self.randomize:
                random_indices = torch.randperm(store_inventory_and_params.size(1))
                aggregated_store_embeddings = store_inventory_and_params[:, random_indices, :]
            else:
                aggregated_store_embeddings = store_embeddings
        else:
            aggregated_store_embeddings = store_embeddings.mean(dim=1)

        input_tensor = self.flatten_then_concatenate_tensors([aggregated_store_embeddings, observation['warehouse_inventories']])
        return self.net['context'](input_tensor)

class GNN_Separation(MyNeuralNetwork):
    def __init__(self, args, problem_params, device='cpu'):
        super().__init__(args, problem_params, device)
        self.n_stores = problem_params['n_stores']
        self.pna_delta = (torch.log(torch.tensor(self.n_stores + 1, device=self.device)) 
                          + self.n_stores * torch.log(torch.tensor(2, device=self.device))) \
                          / torch.tensor(self.n_stores + 1, device=self.device)
        self.use_pna = 'use_pna' in args and args['use_pna']

    def get_store_inventory_and_context_params(self, observation):
        return observation['store_inventories']

    def get_store_inventory_and_params(self, observation):
        store_params = torch.stack([observation[k] for k in ['mean', 'std', 'underage_costs', 'lead_times']], dim=2)
        return torch.concat([observation['store_inventories'], store_params], dim=2)

    def forward(self, observation):
        store_inventory_and_params = self.get_store_inventory_and_params(observation)
        store_embeddings_store = store_inventory_and_params
        store_embeddings_warehouse = store_inventory_and_params

        store_embeddings_store = self.net['store_embedding_store'](store_embeddings_store)
        store_embeddings_warehouse = self.net['store_embedding_warehouse'](store_embeddings_warehouse)

        if self.use_pna:
            aggregators = [
                lambda x: x.mean(dim=1),
                lambda x: x.min(dim=1)[0],
                lambda x: x.max(dim=1)[0],
                lambda x: x.std(dim=1)
            ]
            aggregated_store_embeddings_store = torch.stack([agg(store_embeddings_store) for agg in aggregators], dim=1)
            aggregated_store_embeddings_warehouse = torch.stack([agg(store_embeddings_warehouse) for agg in aggregators], dim=1)
            scalers = [
                lambda x: x,  # identity
                lambda x: x * (torch.log(torch.tensor(store_inventory_and_params.size(1) + 1, device=self.device)) / self.pna_delta),  # amplification
                lambda x: x * (self.pna_delta / torch.log(torch.tensor(store_inventory_and_params.size(1) + 1, device=self.device)))  # attenuation
            ]
            
            scaled_store_embeddings_store = torch.cat([scale(aggregated_store_embeddings_store) for scale in scalers], dim=1)
            scaled_store_embeddings_warehouse = torch.cat([scale(aggregated_store_embeddings_warehouse) for scale in scalers], dim=1)
            store_aggregation_store = scaled_store_embeddings_store.view(scaled_store_embeddings_store.size(0), 1, -1)
            store_aggregation_warehouse = scaled_store_embeddings_warehouse.view(scaled_store_embeddings_warehouse.size(0), 1, -1)
        else:
            store_aggregation_store = torch.mean(store_embeddings_store, dim=1, keepdim=True)
            store_aggregation_warehouse = torch.mean(store_embeddings_warehouse, dim=1, keepdim=True)

        input_tensor_store = self.flatten_then_concatenate_tensors([store_aggregation_store, observation['warehouse_inventories']])
        input_tensor_warehouse = self.flatten_then_concatenate_tensors([store_aggregation_warehouse, observation['warehouse_inventories']])
        context_store = self.net['context_store'](input_tensor_store)
        context_warehouse = self.net['context_warehouse'](input_tensor_warehouse)

        warehouses_and_context = self.concatenate_signal_to_object_state_tensor(observation['warehouse_inventories'], context_warehouse)
        warehouse_intermediate_outputs = self.net['warehouse'](warehouses_and_context)[:, :, 0]
        stores_and_context = self.concatenate_signal_to_object_state_tensor(self.get_store_inventory_and_params(observation), context_store)
        store_intermediate_outputs = self.net['store'](stores_and_context)[:, :, 0]

        store_allocation = self.apply_proportional_allocation(
            store_intermediate_outputs, 
            observation['warehouse_inventories']
            )
        warehouse_allocation = warehouse_intermediate_outputs
        if self.use_warehouse_upper_bound:
            warehouse_allocation = warehouse_intermediate_outputs * self.warehouse_upper_bound.unsqueeze(1)
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

class SymmetryGNNRealData(SymmetryAwareRealData, SymmetryGNN):
    pass

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
            input_data = [observation['store_inventories'][:, :, 0], observation['past_demands'], observation['arrivals'], observation['orders']]
        
        input_data += [observation[key] for key in ['underage_costs', 'days_from_christmas']]

        if 'warehouse_inventories' in observation:
            warehouse_inventories = observation['warehouse_inventories'] / R if self.apply_normalization else observation['warehouse_inventories']
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
        
        current_period, lead_times \
            = self.unpack_args(observation, ["current_period", "lead_times"])
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
            'store': 1, 
            'warehouse': 1, 
            'context': None
            }
        return default_sizes[module_name]

    def get_architecture(self, name):

        architectures = {
            'vanilla_one_store': VanillaOneStore, 
            'vanilla_one_store_for_warehouse': VanillaOneStoreForWarehouse,
            'base_stock': BaseStock,
            'capped_base_stock': CappedBaseStock,
            'echelon_stock': EchelonStock,
            'vanilla_serial': VanillaSerial,
            'vanilla_serial_proportional': VanillaSerial_proportional,
            'vanilla_transshipment': VanillaTransshipment,
            'vanilla_one_warehouse': VanillaOneWarehouse,
            'symmetry_aware': SymmetryAware,
            'symmetry_aware_real_data': SymmetryAwareRealData,
            'symmetry_GNN_real_data' : SymmetryGNNRealData,
            'data_driven': DataDrivenNet,
            'transformed_nv': TransformedNV,
            'fixed_quantile': FixedQuantile,
            'quantile_nv': QuantileNV,
            'returns_nv': ReturnsNV,
            'just_in_time': JustInTime,
            'symmetry_aware_transshipment': SymmetryAwareTransshipment,
            'SymmetryGNN': SymmetryGNN,
            'weekly_forecast_NN': WeeklyForecastNN,
            'GNN_Separation': GNN_Separation,
            'transformed_nv_noquantile': TransformedNV_NoQuantile,
            'transformed_nv_calculated_quantile': TransformedNV_CalculatedQuantile,
            'transformed_nv_noquantile_sep_stores': TransformedNV_NoQuantile_SeparateStores,
            'pretrained_store': Pretrained_Store,
            'GNN_MP': GNN_MP,
            'GNN_MP_real': GNN_MP_real,
            'GNN_MP_transshipment': GNN_MP_transshipment,
            'GNN_MP_noembedding': GNN_MP_noembedding,
            'GNN_MP_sigmoid': GNN_MP_sigmoid,
            }
        return architectures[name]
    
    def create_neural_network(self, scenario, nn_params, device='cpu'):

        nn_params_copy = copy.deepcopy(nn_params)

        # If not specified in config file, set output size to default value
        for key, val in nn_params_copy['output_sizes'].items():
            if val is None:
                nn_params_copy['output_sizes'][key] = self.set_default_output_size(key, scenario.problem_params)

        model = self.get_architecture(nn_params_copy['name'])(
            nn_params_copy, 
            scenario.problem_params,
            device=device
            )
        
        # Calculate warehouse upper bound if specified in config file
        if 'warehouse_upper_bound_mult' in nn_params.keys():
            model.warehouse_upper_bound_mult = nn_params['warehouse_upper_bound_mult']
        
        return model.to(device)
