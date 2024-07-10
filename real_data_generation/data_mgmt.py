import pickle
import json
import pandas as pd
import torch
import datetime
import os, shutil
import copy
import time
from collections import defaultdict
import itertools
import numpy as np

# dump object to file
def save_obj(object, file, extension=".pkl"):

    with open(f"{file}{extension}", "wb") as f:
        pickle.dump(object, f, pickle.HIGHEST_PROTOCOL)


# dump dict-like or list-like object to file
def dump_json(object, file):

    with open(f'{file}.txt', 'w') as fp:
        json.dump(object, fp)


# load dict-like or list-like object
def load_json(file):

    with open(f"{file}.txt", "r") as fp:
        return json.load(fp)


# laod object from file
def load_obj(file):
    with open(f"{file}.pkl", "rb") as f:
        return pickle.load(f)


# laod object from file as turn into df
def load_obj_as_df(file, columns=None):

    d = load_obj(file)
    if columns:
        d = {key: d[key] for key in columns}
    df = pd.DataFrame.from_dict(d)
    return df


# dump lookup table/dict to file
def dump_lookup_dict(lookup_dict):
    aux_dict = {str(key): int(val) for key, val in zip(lookup_dict.keys(), lookup_dict.values())}
    # print(f"aux_dict: {aux_dict}")
    dump_json(aux_dict, "Input/lookup")


# load lookup table/dict to file
def load_lookup_table(file):
    lookup_dict = load_json(file)
    return {tuple(int(s) for s in key.replace(")", "").replace("(", "").split(",")): int(val)
            for key, val in zip(lookup_dict.keys(), lookup_dict.values())}


# get current date in year_month_day format
def get_year_month_day():

    ct = datetime.datetime.now()
    return f"{ct.year}_{ct.month:02d}_{ct.day:02d}"


# get time stamp, to use in model dump name
def get_time_stamp():

    return int(datetime.datetime.now().timestamp())


# def dump_torch_model_and_optimizer_state(order_model, optimizer, scenario, model_name, other_data=None, path_string="",
#                                          folder="Model_dumps", model_directory_string="", date=None):
# dump model and optimizer to file
def dump_torch_model_and_optimizer_state(order_model, optimizer, data_for_save, best_performance_data, folder="Model_dumps"):

    nn_models = order_model.nn_models
    # print(f'optimizer.dev_metrics: {optimizer.dev_metrics}')
    # set dictionary with data about neural net model, parameter optimizer and other metrics
        # 'state_dicts': [model.state_dict() for model in nn_models],
        # 'state_dicts': order_model.all_state_dicts(),
    state_dicts, optimizer_dicts = {}, {}
    if data_for_save['save_model_weights']:  # models params might be too heavy, so only save if specified
        state_dicts = best_performance_data['model_params_to_save']
        optimizer_dicts = optimizer.param_optimizer.state_dict()

    state = {
        'state_dicts': state_dicts,
        'optimizer': optimizer_dicts,
        'current_metrics': optimizer.get_current_metrics(),
        'attributes_to_set': order_model.attributes_to_set(),
        'dev_metrics': optimizer.dev_metrics,
        'best_train_loss': best_performance_data['train_loss'],
        'best_dev_loss': best_performance_data['dev_loss'],
    }
    print(f" current_metrics: {state['current_metrics'].keys()}")
    if data_for_save:
        state.update(data_for_save)
        # state = {**state, **data_for_save}

    #create corresponding directories if they don't already exist
    # date = get_year_month_day()
    intermediate_strings = [f"{data_for_save['date']}"]  # list of names of consecutive folders (appended one by one to path)
    if data_for_save['model_directory_string']:
        intermediate_strings.append(f"/{data_for_save['model_directory_string']}")
        # create_folder_if_not_exists(path)
    intermediate_strings += [f"{data_for_save['scenario_name']}", f"{data_for_save['model_name']}"]
    path = create_many_folders_if_not_exist_and_return_path(f"{folder}", intermediate_strings)

    # print(f'state_dicts: {state["state_dicts"]}')
    # dump model to file
    torch.save(state, f"{path}/{data_for_save['path_string']}_{data_for_save['timestamp']}.pt")

    print()
    print(f"Saving {path}/{data_for_save['path_string']}_{data_for_save['timestamp']}.pt")
    print(f"best train loss: {best_performance_data['train_loss']}")
    print(f"best dev loss: {best_performance_data['dev_loss']}")


#create a sequence of folders (one within the other) if they don't exist
# path: path from which to start creating folders
# list of directory names
def create_many_folders_if_not_exist_and_return_path(path, intermediate_folder_strings):

    for string in intermediate_folder_strings:
        path += f"/{string}"
        create_folder_if_not_exists(path)
    return path


# return newest file in directory given by path
def newest_file_in_dir(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    paths.sort()
    return paths[-1]


# return newest file in directory given by path
def all_files_in_dir(path):
    files = os.listdir(path)
    paths = [os.path.join(path, basename) for basename in files]
    paths.sort()
    return paths


# get corresponding directory name
# folder: initial path
def get_path_to_dir(folder, model_load_date, model_directory_string, scenario, which_model_load):

    path_to_dir = f"{folder}/{model_load_date}"
    if model_directory_string:
        path_to_dir += f"/{model_directory_string}"
    path_to_dir += f"/{scenario}/{which_model_load}"

    return path_to_dir

# mode: 0 (last), 1 (best), 2 (all)
def many_load_torch_model_and_optimizer_state(model_load_date, scenario, which_model_load, model_load_name=None,
                                              folder="Model_dumps", model_directory_string="", mode=0):
    # state = load_torch_model_and_optimizer_state(model_load_date, scenario, model_name,
    #                                              model_directory_string=exp, best=True)

    if mode in [0, 1, 3]:
        return [load_torch_model_and_optimizer_state(model_load_date, scenario, which_model_load,
                                                 model_directory_string=model_directory_string, mode=mode)]
                                                #  model_directory_string=model_directory_string, best=bool(mode))]
    else:
        return load_torch_model_and_optimizer_state(model_load_date, scenario, which_model_load,
                                                     model_directory_string=model_directory_string, all=True)


# load torch model, optimizer and other data. if model_load_name=None, load last model saved within the directory
def load_torch_model_and_optimizer_state(model_load_date, scenario, which_model_load, model_load_name=None,
                                         folder="Model_dumps", model_directory_string=None, mode=0, all=False):
                                        #  folder="Model_dumps", model_directory_string=None, best=False, all=False):

# 
    path_to_dir = get_path_to_dir(folder, model_load_date, model_directory_string, scenario, which_model_load)

    # print(f"{path_to_dir}/{model_load_name}")
    paths = [os.path.join(path_to_dir, basename) for basename in os.listdir(path_to_dir)]
    if model_load_name:
        return torch.load(f"{path_to_dir}/{model_load_name}", map_location=torch.device('cpu'))
    elif mode in [1, 3]: # 1= select by best_dev_loss, 3= select by test_loss
        # for path in paths:
            # if path[-3:] == ".pt":
            #     print(path)
            #     torch.load(f"{path}", map_location=torch.device('cpu'))
        states = [(torch.load(f"{path}", map_location=torch.device('cpu')), path) for path in paths if path[-3:] == ".pt"]
        # states = [torch.load(f"{path}", map_location=torch.device('cpu')) for path in paths if path[-3:] == ".pt"]
        # print(f"{[state['current_metrics']['dev_loss'][-1] for state in states]}")

        if mode == 1:
            print("Sorting by best_dev_loss")
            states.sort(key=lambda x: x[0]['best_dev_loss'])
        elif mode == 3:
            print("Sorting by test loss")
            # only keep entries where x[0]['current_metrics']['test_loss'] is non-negative
            states = [state for state in states if state[0]['current_metrics']['test_loss'] >= -0.000001]
            states.sort(key=lambda x: x[0]['current_metrics']['test_loss'])
        # states.sort(key=lambda x: x[0]['best_train_loss'])

        print(scenario)
        print(f'Loaded path: {states[0][1]}')
        print()

        
        # states.sort(key=lambda x: x['current_metrics']['dev_loss'][-1])
        # print(states[0]['current_metrics']['test_loss'])
        # print(states[0]['current_metrics']['dev_loss'][-1])
        # print(f"{[state['current_metrics']['dev_loss'][-1] for state in states]}")
        # print()
        return states[0][0]
    elif all:
        # states = []
        # for path in paths:
        #     if path[-3:] == ".pt":
        #         print(path)
        #         try:
        #             states.append(torch.load(f"{path}", map_location=torch.device('cpu')))
        #         except RuntimeError as err:
        #             print(err)

        states = [torch.load(f"{path}", map_location=torch.device('cpu')) for path in paths if path[-3:] == ".pt"]
        
        # print(paths)
        # print([state["config_params"]["lr"] for state in states])
        return [state for state in states if state["current_metrics"]["epochs_done"] > 0]
    else:
        latest_file_string = newest_file_in_dir(path_to_dir)
        print(latest_file_string)
        return torch.load(latest_file_string, map_location=torch.device('cpu'))

def load_many_state_dicts(order_model, state_dicts):
    
    for nn_model, state_dict in zip(order_model.nn_models, state_dicts):

        nn_model.load_state_dict(state_dict)


# load torch model, optimizer and other data, and then set/fetch them to the corresponding model.optimizer instances
def load_and_set_model_and_optimizer_state(order_model, optimizer, model_load_date, scenario, which_model_load,
                                           model_load_name="", folder="Model_dumps", lr=None, best=True):

    # load data for the model
    state = load_torch_model_and_optimizer_state(model_load_date, scenario, which_model_load, model_load_name, folder, mode=best)

    print(state["config_params"])

    # set/fetch state data for every neural net within the model
    for nn_model, state_dict in zip(order_model.nn_models, state['state_dicts']):

        nn_model.load_state_dict(state_dict)

    if state['attributes_to_set']:
        for key, val in state['attributes_to_set'].items():
            setattr(order_model, key, val)
        order_model.set_all_parameters()

    # set state for optimizer and current metrics (e.g. epochs, loss arrays...)
    optimizer.param_optimizer = torch.optim.Adam(order_model.all_parameters, lr=optimizer.hyper_parameters.lr)
    optimizer.param_optimizer.load_state_dict(state['optimizer'])
    optimizer.set_metrics_from_load(state['current_metrics'])

    # set learning rate to newly-defined learning rate
    if lr:
        optimizer.param_optimizer.param_groups[0]['lr'] = lr

def unpack_args(args, keys):
        return [args[key] for key in keys] if len(keys) > 1 else args[keys[0]]


# create a directory in the corresponding file, if it does not already exist
def create_folder_if_not_exists(path):

    if not os.path.isdir(path):
        os.mkdir(path)


# creates and returns a string to dump the output file of the instance execution
def create_file_string(strings, values, folder, extension=""):

    string = f"exps/{folder}/"
    param_string = "".join(f"_{s}{v}" for s, v in zip(strings, values))[1:]
    string = string + param_string + extension

    return string, param_string


def chain(*iterables):
    for it in iterables:
        for element in it:
            yield element


# create a scenario to file
# keys: keys for dictionary
# data: corresponding value for dictionary entry
# scenario_string: name for dumping the scenario
def create_scenario(keys, data, scenario_string, folder="Input/Scenarios"):

    date = get_year_month_day()
    create_folder_if_not_exists(f"{folder}/{date}")
    d = {key: val for key, val in zip(keys, data)}

    save_obj(d, f"{folder}/{date}/{scenario_string}") #save .pkl
    dump_json(d, f"{folder}/{date}/{scenario_string}") #dump .json

# load scenario from file
def get_scenario(scenario_string,
                 keys=("problem_params", "settings", "warehouse_params", "store_params", "seeds", "echelon_params"),
                 folder="Input/Scenarios"):

    data = load_obj(f"{folder}/{scenario_string}")
    return [data[key] for key in keys]


# add one or more layers to existing neural net architecture and return
# architectures: list of neural net architectures (each is a list of nn.layers)
# adjustments: list of layers to add to architecture
# arch_pos: string detailing which architecture within architectures to consider
def adjust_architecture(architectures, adjustments, adjustment_positions, arch_pos=0):
    new_architecture = [copy.copy(a) for a in architectures[arch_pos]]
    for adjustment, adjustment_position in zip(adjustments, adjustment_positions):
        new_architecture.insert(adjustment_position, adjustment)
    new_architecture = tuple(new_architecture)
    return [new_architecture]


# copy model from one folder to another folder
# date_from: model date from which to copy
# date_to: model date to which copy
# scenario_strings: list of scenario names to copy
# model_names: list of model names to copy
def copy_files_to_folder(date_from, date_to, scenario_strings, model_names, all=True):

    for scenario in scenario_strings:
        for model_name in model_names:
            path_to_dir = f"Model_dumps/{date_from}/{scenario}/{model_name}"
            print(f'path_to_dir: {path_to_dir}')
            if os.path.isdir(path_to_dir):
                print(path_to_dir)
                if all:
                    all_files = all_files_in_dir(path_to_dir)
                else:
                    all_files = [newest_file_in_dir(path_to_dir)]
                for f in all_files:
                    latest_file_string = f.split(sep="/")[-1]
                    # latest_file_string = newest_file_in_dir(path_to_dir).split(sep="/")[-1]
                    path_from = path_to_dir + f"/{latest_file_string}"
                    path_to = create_many_folders_if_not_exist_and_return_path("Model_dumps", [date_to, scenario, model_name])
                    path_to += f"/{latest_file_string}"
                    state = torch.load(f"{path_from}", map_location=torch.device('cpu'))
                    # print(state['config_params']['neurons'])
                    # if state['config_params']['neurons'][0] == 512:
                    #     print(f"neurons: {state['config_params']['neurons']}")
                    print(f'dumping on {path_to}')
                    shutil.copy(path_from, path_to)
                    



# generate scenario name from data values
def scenario_name_generator(T, n_warehouses, n_stores, w_holding, w_lead_time, sym, underages, lead_times, distributions,
                            mean_demands, lost_demand, perturbation, symmetric, correlation, ar_weights=None,
                            ar_noise_var=None, serial=0, random_params=0, echelon_params=None):

    consider_warehouse = int(n_warehouses > 0)  # if n_warehouses is > 0, there is a warehouse...
    scenario_name = ""
    if serial:
        scenario_name += f"ser{len(echelon_params['lead_time']) + 2}_h{echelon_params['holding_costs']}_l{echelon_params['lead_time']}"
    if consider_warehouse:  # specific format if there is a warehouse
        scenario_name += f"w{consider_warehouse}_h{w_holding[0]}_l{w_lead_time[0]}_"
    scenario_name += f"s{n_stores}_sym{symmetric}_r{random_params}_"

    dist_params_string = str(mean_demands[0][0]) + f"".join([f"-{m}" for m in mean_demands[0][1:]])
    # print(dist_params_string)

    scenario_name += f"d{distributions[0][0]}"

    if distributions[0] == "ar":
        scenario_name += f"{ar_weights}-{ar_noise_var}"


    if sym or True:  # if all stores are equal (i.e symmetrical)
        scenario_name += f"_m{dist_params_string}_" \
                         f"u{underages[0]}_l{lead_times[0]}_"
    else:  # otherwise, must include data from each store
        scenario_name += f"_d{[d[0] for d in distributions]}_m{mean_demands}_" \
                         f"u{underages}_l{lead_times}_"

    scenario_name += f"p{perturbation}_cor{correlation}_ld{lost_demand}_T{T}_ri{0}"

    return scenario_name


# get the cross product tuples of all the lists in many_lists
def cross_prod_tuples(many_lists):

    return list(itertools.product(*many_lists))

def list_intersection(l1, l2):

    print(list(set(l1) & set(l2)))
    return list(set(l1) & set(l2))

if __name__ == "__main__":

    copy_files = True
    if copy_files:  # copy model files from one folder to another
        # date_from = "2023_04_28_small_batch"
        i = 5
        date_from = f"2024_03_01/sample_efficiency"
        # date_from = "2022_11_07"
        # date_to = "2023_05_05_hp2"
        date_to = f"2024_01_23/sample_efficiency"
        # scenario_strings = [f's1_sym1_r0_dp_m5.0_u{u}_l{l}_p0.0_cor0.0_ld1_T50_ri0'
        # scenario_strings = [f'ser4_h[0.1, 0.2]_l[2, 4]w1_h0.5_l3_s1_sym1_r0_dn_m5.0-2.0_u{u}_l{l}_p0.0_cor0.0_ld0_T50_ri0'
        # scenario_strings = [f'w1_h0.3_l6_s{s}_sym0_r1_dn_m(5.0, 0.5)-(0.0, 0.5)_u({u}, 0.3)_l(2, 3)_p0.0_cor0.5_ld1_T50_ri0'
        scenario_strings = [f"w1_h0.3_l6_s{s}_sym0_r1_dn_m(5.0, 0.5)-(0.0, 0.5)_u({u}, 0.3)_l(2, 3)_p0.0_cor0.5_ld1_T50_ri0"
        # scenario_strings = [f"w1_h0.5_l3_s{s}_sym0_r1_dn_m(5.0, 0.5)-(0.0, 0.32)_u(4.0, 0.0)_l(6, 6)_p0.0_cor0.5_ld0_T50_ri0"
        # scenario_strings = [f"ser4_h[0.1, 0.2]_l[2, 4]w1_h0.5_l3_s1_sym1_r0_dn_m5.0-2.0_u{u}_l{l}_p0.0_cor0.0_ld0_T50_ri0"
        # scenario_strings = [f"s1_sym1_da_m5.0_u{u}_l{l}_p0.0_cor0.0_ld1_T50_ri0"
        # scenario_strings = [f"w1_h0.5_l3_s{s}_sym{sym}_dn_m5.0-2.23_u19.0_l4_p{per}_cor0.0_ld1_T50_ri0"
        # scenario_strings = [f"w1_h0.5_l3_s{s}_sym1_dn_m5.0-2.23_u19.0_l4_p{per}_cor0.0_ld1_T50_ri0"
                            # for s in [1]
                            for s in [50, 30, 20, 10, 5, 3]
                            # for s in [5, 10, 40, 100]
                            for u in [9.0]
                            # for u in [39.0, 19.0, 9.0, 4.0]
                            # for l in [1]
                            # for l in [1, 2, 3, 4]
                            for l in [(3, 8)]
                            for ld in [0]
                            for sym in [0]
                            for cor in [0.0]
                            for per in [0.0]]
        # model_name = ["capped_base_stock_fixed"]
        # model_name = ["context_then_alone_softplus_sigmoid"]
        model_name = ["generic_net_softmax_sigmoid"]
        # model_name = ["forecaster_random_quantiles", "back_orders_news_vendor_net", "forecaster_news_vendor_net", "forecaster_transformed_nv", "forecaster_same_quantile", "forecaster_different_quantiles", "look_forward_net"]
        # model_name = ["context_then_alone", "echelon_cbs"]
        # model_name = ["generic_net_sigmoid_trans", "transshipment_ordering"]
        # mode = 1
        copy_files_to_folder(date_from, date_to, scenario_strings, model_name, all=True)

    generate_scenarios = False
    if generate_scenarios:  # create scenarios and dump to file
        # number of stores, warehouses, and instances and time horizon
        # many_n_stores, many_n_warehouses, n_extra_echelons, T = [1], [1], 2, 50
        # many_n_stores, many_n_warehouses, n_extra_echelons, T = [1, 3, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 130, 150], [0, 1], 0, 50
        many_n_stores, many_n_warehouses, n_extra_echelons, T = [3, 5, 10], [1], 0, 70
        # many_n_stores, many_n_warehouses, n_train, n_test, T = [3], [1], 2**13, 2*12, 50
        many_perturbations = [0.0]
        # data_params = {"n_train": n_train, "n_test": n_test}
        seeds = {"underage": 28, "holding": 73, "mean": 33, "stds": 92, "demand_sequence": 57,
                 "w_lead_time": 88, "perturbation": 84, "store_lead_times": 41}
        sym = 0  # if 1, use data for homogeneous stores (symmetric)
        random_params = 1
        lost_demand = 0
        many_cor = [0.0, 0.5]
        holding_costs = [1.0]  # unitary holding cost per store
        h = holding_costs[0]
        # many_lead_times = [(2, 10)]
        # many_lead_times = [(3, 8)]
        # many_lead_times = [1, 2, 3, 4]
        # many_lead_times = [(2, 3)]
        # many_lead_times = [(4, 4)]
        many_lead_times = [(2, 2), (4, 4), (6, 6)]
        many_underage_costs = [4.0, 9.0]
        # many_underage_costs = [4.0, 9.0, 19.0, 39.0]
        # many_underage_costs = [2.0, 3.0, 4.0, 6.0, 9.0, 13.0, 19.0, 39.0]
        serial = 0
        ar_weights = (0.4, 0.2, 0.08, 0.05, 0.01)
        ar_noise_var = 1
        # many_underage_costs = [4.0, 9.0, 19.0, 39.0]
        # lead_time = [10]  # lead time for each store (only works if same for all stores)

        for uc, lt, n_stores, n_warehouses, cor, perturbation in \
                cross_prod_tuples([many_underage_costs, many_lead_times, many_n_stores, many_n_warehouses, many_cor,
                                   many_perturbations]):

            if sym == random_params == 1:  # if stores are symmetric, we should not sample random parameters
                assert False

            consider_warehouse = int(n_warehouses > 0)
            # if n_stores == 1:
            #     sym = 1
            problem_params = {"n_stores": n_stores, "n_warehouses": n_warehouses, "T": T, "serial": serial,
                              "n_extra_echelons": n_extra_echelons}
            lead_time = [lt]
            if sym:  # if stores are homogeneous
                distributions = ["normal"]  # poisson, geometric, normal...
                # demand_params = [[5.0]]  # (e.g. for normal: 2 params per store, poisson: 1)
                demand_params = [[5.0, 2.0]]  # (e.g. for normal: 2 params per store, poisson: 1)
                # demand_params = [[5.0, 1.6]]  # (e.g. for normal: 2 params per store, poisson: 1)
                # demand_params = [[72.0, 10.5356]]  # (e.g. for normal: 2 params per store, poisson: 1)
                underage_costs = [uc]  # underage/price for each store

            else:  # if stores are heterogeneous
                if random_params:  # if store parameters are sampled rather than explicitly given
                    # holding_costs = [(h, 0.3)]
                    # underage_costs = [(uc, 0.3)]
                    holding_costs = [(h, 0.0)]
                    underage_costs = [(uc, 0.0)]
                    # distributions = ["real"]  # poisson, geometric, normal...
                    # distributions = ["real"]  # poisson, geometric, normal...
                    distributions = ["normal"]  # poisson, geometric, normal...
                    # demand_params = [[(5.0, 0.0)]]  # (e.g. for normal: 2 params per store, poisson: 1)
                    # demand_params = [[(5.0, 0.5), (0.0, 0.5)]]  # (e.g. for normal: 2 params per store, poisson: 1)
                    demand_params = [[(5.0, 0.5), (0.0, 0.32)]]  # (e.g. for normal: 2 params per store, poisson: 1)
                    # demand_params = [[(5.0, 0.2)]]  # (e.g. for normal: 2 params per store, poisson: 1)

                else:
                    # distributions = ["poisson" for _ in range(n_stores)]
                    distributions = ["normal" for _ in range(n_stores)]
                    # demand_params = [[5.0] for _ in range(n_stores)]
                    # demand_params = [[5.0, 1.6] for _ in range(n_stores)]
                    # demand_params = [[5.0, 2.0], [6.3, 1.7], [4.2, 1.5], [4.9, 1.0], [7.4, 3.1],
                    #                  [5.0, 2.0], [6.3, 1.7], [4.2, 1.5], [4.9, 1.0], [7.4, 3.1]]
                    demand_params = [[5.0, 1.6], [6.5, 2.0], [4.2, 1.0], [4.9, 1.2], [7.4, 2.8], [5.4, 0.8],
                                     [2.3, 0.5], [9.3, 3.2], [6.2, 2.1], [7.3, 2.3]]
                    # demand_params = [[5.0, 2.23], [6.3, 1.7], [4.2, 2.1]]
                    demand_params = demand_params[: n_stores]
                    # underage_costs = [19.0, 17.3, 24.5]
                    # underage_costs = np.concatenate([[4.0, 9.0, 19.0, 39.0] for _ in range(6)]).tolist()
                    # underage_costs = np.concatenate([[4.0, 9.0, 19.0, 39.0] for _ in range(6)]).tolist()
                    underage_costs = [uc for _ in range(n_stores)]
                    # underage_costs = [19.0, 17.3, 24.5, 17.8, 22.1]
                    h, l = holding_costs[0], lead_time[0]
                    holding_costs = [h for _ in range(n_stores)]  # unitary holding cost per store
                    # holding_costs = [1.2, 1.4, 0.7, 0.8, 1.1]  # unitary holding cost per store
                    # holding_costs = [h for _ in range(n_stores)]  # unitary holding cost per store
                    lead_time = [l for _ in range(n_stores)]  # lead time for each store
                    # all_lead_times = [1, 4, 7, 10, 15, 20]
                    # all_lead_times = [1, 2, 3, 4]
                    # lead_time = np.concatenate([[all_lead_times[j] for _ in range(4)] for j in range(6)]).tolist()
                    # lead_time = [lt for _ in range(n_stores)]
                    print(underage_costs)
                    print(lead_time)

            if distributions[0] != "ar":
                ar_weights = tuple([0])
                ar_noise_var = 0

            store_params = {"demand_type": [(distributions[i], demand_params[i], ar_weights, ar_noise_var) for i in range(len(distributions))],
                            "holding_costs": holding_costs,
                            "underage_costs": underage_costs,
                            "lead_time": lead_time,
                            "symmetric": sym,
                            "random_params": random_params,
                            "lost_demand": lost_demand,
                            "perturbation": perturbation,
                            "correlation": cor}

            # parameters for warehouse
            warehouse_params = {"holding_costs": [0.5],  # unitary holding costs
                                "lead_time": [3],  # lead time
                                # "lead_time": [(2, 8)],  # lead time
                                "initial_inventory": [0],  # inventory at beginning of planning horizon
                                "inv_per_period": [0],
                                "random_lead_time": 0}

            # parameters for warehouse
            echelon_params = {"holding_costs": [0.1, 0.2],  # unitary holding costs
                              "lead_time": [2, 4],  # lead time
                              }
            # # parameters for warehouse
            # echelon_params = {"holding_costs": [0.0125, 0.025, 0.05, 0.1, 0.2],  # unitary holding costs
            #                   "lead_time": [5, 3, 6, 2, 4],  # lead time
            #                   }

            if warehouse_params["random_lead_time"] and isinstance(warehouse_params["lead_time"][0], int):
                assert False

            if serial and (n_stores > 1 or n_warehouses != 1):
                assert False

            if serial and (n_extra_echelons != len(echelon_params['holding_costs'])):
                assert False

            # data not currently used
            use_prev_lookup = True  # if False, generate lookup table, else get it from previous dump
            use_score_ordering = True
            use_reserve_price = True
            discrete_demand = False
            stationary = 1
            which_score = "profit"  # which score to use, when using score_ordering (bidding)

            settings = {"use_prev_lookup": use_prev_lookup, "use_score_ordering": use_score_ordering, "which_score": which_score,
                        "use_reserve_price": use_reserve_price, "discrete_demand": discrete_demand, "stationary": stationary,
                        "consider_warehouse": consider_warehouse}
            data = [problem_params, settings, warehouse_params, store_params, seeds, echelon_params]
            keys = ["problem_params", "settings", "warehouse_params", "store_params", "seeds", "echelon_params"]

            mean_demands = [d for d in demand_params]

            # generate scenario name from data values
            scenario_name = \
                scenario_name_generator(T, n_warehouses, n_stores, warehouse_params["holding_costs"],
                                        warehouse_params["lead_time"], sym, store_params["underage_costs"],
                                        store_params["lead_time"], distributions, mean_demands,
                                        store_params["lost_demand"], store_params["perturbation"], sym, cor,
                                        ar_weights=ar_weights, ar_noise_var=ar_noise_var, serial=serial,
                                        random_params=random_params, echelon_params=echelon_params)

            create_scenario(keys, data, scenario_name)  # dump scenario to file

