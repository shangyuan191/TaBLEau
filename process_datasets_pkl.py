import csv
import os
import pandas as pd
import json
import pickle
from sklearn.model_selection import train_test_split
import numpy as np
from datasets import Dataset, DatasetDict
from datasets import load_dataset
# from huggingface_hub import login
# login(token="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")
if __name__=="__main__":
    

    dataset = load_dataset("SkylerChuang/small_binclass_datasets", "kaggle_Analytics_Vidhya_Loan_Prediction")
    df = dataset["df"]
    print(df)
    # dataset_sizes=['small','large']
    # for dataset_size in dataset_sizes:
    #     with open(f'./datasets/{dataset_size}_datasets/{dataset_size}_preprocessed_train_data.pkl', 'rb') as f:
    #         train_data = pickle.load(f)
    #     with open(f'./datasets/{dataset_size}_datasets/{dataset_size}_preprocessed_val_data.pkl', 'rb') as f:
    #         val_data = pickle.load(f)   
    #     with open(f'./datasets/{dataset_size}_datasets/{dataset_size}_preprocessed_test_data.pkl', 'rb') as f:
    #         test_data = pickle.load(f)
    #     print(dataset_size)
    #     for task_type in list(train_data.keys()):
    #         print(task_type)
    #         # data_info=[]
    #         dataset_dict = DatasetDict()
    #         for dataset_name in train_data[task_type].keys():
    #             original_dataset_name=dataset_name
    #             dataset_name=original_dataset_name.replace("[", "").replace("]", "_").replace(" ", "_").replace("(", "_").replace(")", "_").replace("-", "_").replace("&","_")
    #             # dataset_names.append(dataset_name)
    #             path_name=f'./datasets/{dataset_size}_datasets/{task_type}/{dataset_name}'
    #             if not os.path.exists(path_name):
    #                 os.makedirs(path_name)
                
    #             X_num_train=train_data[task_type][original_dataset_name]['X_num'] if train_data[task_type][original_dataset_name]['X_num'] is not None else pd.DataFrame()
    #             X_cat_train=train_data[task_type][original_dataset_name]['X_cat'] if train_data[task_type][original_dataset_name]['X_cat'] is not None else pd.DataFrame()
    #             y_train = pd.DataFrame(train_data[task_type][original_dataset_name]['y'], columns=['target'])  # 設定 y_train 的列名

    #             X_num_val=val_data[task_type][original_dataset_name]['X_num'] if val_data[task_type][original_dataset_name]['X_num'] is not None else pd.DataFrame()
    #             X_cat_val=val_data[task_type][original_dataset_name]['X_cat'] if val_data[task_type][original_dataset_name]['X_cat'] is not None else pd.DataFrame()
    #             y_val = pd.DataFrame(val_data[task_type][original_dataset_name]['y'], columns=['target'])  # 設定 y_val 的列名

    #             X_num_test=test_data[task_type][original_dataset_name]['X_num'] if test_data[task_type][original_dataset_name]['X_num'] is not None else pd.DataFrame()
    #             X_cat_test=test_data[task_type][original_dataset_name]['X_cat'] if test_data[task_type][original_dataset_name]['X_cat'] is not None else pd.DataFrame()
    #             y_test = pd.DataFrame(test_data[task_type][original_dataset_name]['y'], columns=['target'])  # 設定 y_test 的列名

    #             X_num_result = pd.concat([X_num_train,X_num_val,X_num_test], axis=0)
    #             X_cat_result = pd.concat([X_cat_train,X_cat_val,X_cat_test], axis=0)
    #             y_result = pd.concat([y_train,y_val,y_test], axis=0)
                
    #             # 生成數值和類別特徵的名稱
    #             num_feature_names = [f'N_feature_{i+1}' for i in range(len(X_num_result.columns))]
    #             cat_feature_names = [f'C_feature_{i+1}' for i in range(len(X_cat_result.columns))]

    #             # 為數值和類別數據重新命名欄位並合併
    #             X_num_result.columns = num_feature_names
    #             X_cat_result.columns = cat_feature_names
    #             X_result = pd.concat([X_num_result, X_cat_result], axis=1)

    #             # 合併 X 和 y 資料
    #             data = pd.concat([X_result, y_result], axis=1)
    #             data.to_csv(f'{path_name}/{dataset_name}.csv', index=False,header=True)
    #             # data=Dataset.from_pandas(data)
    #             tmp_dict={'df':data}
    #             data=Dataset.from_dict(tmp_dict)
    #             dataset_dict[dataset_name]=data
    #         if dataset_size=='small' and task_type=="multiclass":
    #             continue
    #         dataset_dict.push_to_hub(f"SkylerChuang/{dataset_size}_{task_type}_datasets")
    #         #     data_info.append([dataset_size,task_type,original_dataset_name,dataset_name,data.shape])
    #         # for di in data_info:
    #         #     print(di)