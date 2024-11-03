import os
import pandas as pd
import json

class RayResultsinterpreter:
    def __init__(self):
        pass

    def extract_data(self, top_folder, sort_by):
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
                    if data['dev_loss'].isna().any() or ('test_loss' in data and data['test_loss'].isna().any()) or data['train_loss'].isna().any():
                        print(f"Error, NaN values found in loss columns {subfolder_path}: ")
                        continue
                    data.fillna(0, inplace=True)
                    with open(params_file, 'r') as file:
                        params = json.load(file)
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
                        'master': 'master',
                        'store_underage_cost': 'store_underage_cost',
                        'samples': 'samples',
                        'repeats': 'repeats',
                        'training_n_samples': 'training_n_samples',
                        'store': 'store',
                        'n_sub_sample_for_context': 'n_sub_sample_for_context',
                        'apply_normalization': 'apply_normalization',
                        'store_orders_for_warehouse': 'store_orders_for_warehouse',
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

                    if 'test_loss' in data:
                        result['best_test_loss'] = data['test_loss'].min()
                    result['best_dev_loss'] = data['dev_loss'].min()
                    result['best_train_loss'] = data['train_loss'].min()
                    loss_columns = ['dev_loss', 'test_loss', 'train_loss']
                    for loss_column in loss_columns:
                        if loss_column in data:
                            result[f'{loss_column}(at best_{sort_by})'] = data[data[sort_by] == result[f'best_{sort_by}']][loss_column].iloc[0]

                    result['path'] = subfolder_path
                    results.append(result)
                except Exception as e:
                    print(f"Error processing files in {subfolder_path}: {e}")
        return results

    def make_table(self, paths, conditions, custom_data_filler = None, sort_by = 'dev_loss', pick_row_from_run_by='dev_loss'):
        results = []
        for num_stores, path in paths.items():
            data = self.extract_data(path, pick_row_from_run_by)
            if len(data) == 0:
                continue
            df = pd.DataFrame(data).sort_values(by=f'{sort_by}(at best_{pick_row_from_run_by})', ascending=True)

            for condition_key, condition_values in conditions.items():
                df = df[df[condition_key].isin(condition_values)]

            if len(conditions.keys()) == 0:
                grouped = [('', df)]  # Single group with empty name and all data
            elif len(conditions.keys()) == 1:
                grouped = df.groupby(list(conditions.keys())[0])
            else:
                grouped = df.groupby(list(conditions.keys()))
            for group_name, group_df in grouped:
                top_row = group_df.iloc[0]
                result_row = {
                    "# of stores": num_stores,
                }
                if len(conditions.keys()) == 1:
                    result_row[list(conditions.keys())[0]] = group_name
                else:
                    for condition_key, condition_value in zip(conditions.keys(), group_name):
                        result_row[condition_key] = condition_value

                if 'learning_rate' in top_row:
                    result_row["Learning Rate"] = top_row['learning_rate']
                result_row["Train Loss"] = top_row[f'train_loss(at best_{pick_row_from_run_by})']
                result_row["Dev Loss"] = top_row[f'dev_loss(at best_{pick_row_from_run_by})']
                if f'test_loss(at best_{pick_row_from_run_by})' in top_row:
                    result_row["Test Loss"] = top_row[f'test_loss(at best_{pick_row_from_run_by})']
                result_row["# of runs"] = len(group_df)
                # result_row["path"] = top_row['path']
                if custom_data_filler:
                    custom_data_filler(result_row, top_row)
                results.append(result_row)
        return pd.DataFrame(results)