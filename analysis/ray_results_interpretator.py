import shared_imports
import os
import pandas as pd
import json

class RayResultsInterpretator:
    def __init__(self):
        pass

    def check_ctx_size(self, ctx_size, params):
        if ctx_size == None:
            return True
        if 'context' in params and params['context'] == ctx_size:
            return True
        if 'for_all_networks' in params and params['for_all_networks'] == ctx_size:
            return True
        return False

    def extract_data(self, top_folder, ctx_size):
        results = []
        for submain_folder in os.listdir(top_folder):
            main_folder = os.path.join(top_folder, submain_folder)
            for subfolder in os.listdir(main_folder):
                subfolder_path = os.path.join(main_folder, subfolder)
                progress_file = os.path.join(subfolder_path, 'progress.csv')
                params_file = os.path.join(subfolder_path, 'params.json')
                
                if os.path.exists(progress_file) == False or  os.path.exists(params_file) == False:
                    continue
                try:
                    data = pd.read_csv(progress_file)
                    data.fillna(0, inplace=True)
                    with open(params_file, 'r') as file:
                        params = json.load(file)
                    if self.check_ctx_size(ctx_size, params) == False:
                        continue

                    param_dict = {
                        'n_stores': 'n_stores',
                        'context': 'context',
                        'warehouse_holding_cost': 'warehouse_holding_cost',
                        'warehouse_lead_time': 'warehouse_lead_time',
                        'stores_correlation': 'stores_correlation',
                        'learning_rate': 'learning_rate',
                        'master_neurons': 'master_neurons',
                        'store_embedding': 'store_embedding',
                        'for_all_networks': 'for_all_networks',
                    }
                    result = {}
                    for key, value in param_dict.items():
                        if value in params:
                            result[key] = params.get(value)
                        else:
                            if key == 'warehouse_lead_time':
                                result[key] = 6
                            elif key == 'stores_correlation':
                                result[key] = 0.5

                    result['best_dev_loss'] = data['dev_loss'].min()
                    result['test_loss(at best_dev)'] = data[data['dev_loss'] == result['best_dev_loss']]['test_loss'].iloc[0]
                    result['train_loss(at best_dev)'] = data[data['dev_loss'] == result['best_dev_loss']]['train_loss'].iloc[0]
                    result['best_test_loss'] = data['test_loss'].min()
                    result['best_train_loss'] = data['train_loss'].min()
                    result['path'] = subfolder_path
                    results.append(result)
                except Exception as e:
                    print(f"Error processing files in {subfolder_path}: {e}")
        return results

    def make_table(self, paths, ctx_sizes):
        results = []
        for num_stores, path in paths.items():
            for ctx_size in ctx_sizes:
                data = self.extract_data(path, ctx_size)
                if len(data) == 0:
                    continue
                df = pd.DataFrame(data).sort_values(by='best_dev_loss', ascending=True)
                top_row = df.iloc[0]
                result_row = {
                    "# of stores": num_stores,
                    "context size": ctx_size,
                    "Learning Rate": top_row['learning_rate'],
                    "Train Loss": top_row['train_loss(at best_dev)'],
                    "Dev Loss": top_row['best_dev_loss'],
                    "Test Loss": top_row['test_loss(at best_dev)'],
                    # "path": top_row['path']
                }
                results.append(result_row)
        return pd.DataFrame(results)