# # """Reported (reproduced) results of Tuned XGBoost on TabularBenchmark of
# # the Trompt paper https://arxiv.org/abs/2305.18446.

# # electricity (A4): 88.52 (91.09)
# # eye_movements (A5): 66.57 (64.21)
# # MagicTelescope (B2): 86.05 (86.50)
# # bank-marketing (B4): 80.34 (80.41)
# # california (B5): 90.12 (89.71)
# # credit (B7): 77.26 (77.4)
# # pol (B14): 98.09 (97.5)
# # jannis (mathcal B4): 79.67 (77.81)

# # Reported (reproduced) results of Tuned CatBoost on TabularBenchmark of
# # the Trompt paper: https://arxiv.org/abs/2305.18446

# # electricity (A4): 87.73 (88.09)
# # eye_movements (A5): 66.84 (64.27)
# # MagicTelescope (B2): 85.92 (87.18)
# # bank-marketing (B4): 80.39 (80.50)
# # california (B5): 90.32 (87.56)
# # credit (B7): 77.59 (77.29)
# # pol (B14): 98.49 (98.21)
# # jannis (mathcal B4): 79.89 (78.96)
# # """
# # import argparse
# # import os.path as osp
# # import random

# # import numpy as np
# # import torch

# # from torch_frame.datasets import TabularBenchmark
# # from torch_frame.gbdt import CatBoost, LightGBM, XGBoost
# # from torch_frame.typing import Metric

# # parser = argparse.ArgumentParser()
# # parser.add_argument('--gbdt_type', type=str, default='xgboost',
# #                     choices=['xgboost'])
# # parser.add_argument('--dataset', type=str, default='eye_movements')
# # parser.add_argument('--saved_model_path', type=str,
# #                     default='storage/gbdts.txt')
# # # Add this flag to match the reported number.
# # parser.add_argument('--seed', type=int, default=0)
# # args = parser.parse_args()

# # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# # random.seed(args.seed)
# # np.random.seed(args.seed)
# # torch.manual_seed(args.seed)

# # # Prepare datasets
# # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
# #                 args.dataset)
# # dataset = TabularBenchmark(root=path, name=args.dataset)
# # dataset.materialize()
# # dataset = dataset.shuffle()
# # # Split ratio following https://arxiv.org/abs/2207.08815
# # # 70% is used for training. 30% of the remaining is used for validation.
# # # The final reminder is used for testing.
# # train_dataset, val_dataset, test_dataset = dataset[:0.7], dataset[
# #     0.7:0.79], dataset[0.79:]

# # num_classes = None
# # metric = None
# # task_type = dataset.task_type
# # if dataset.task_type.is_classification:
# #     metric = Metric.ACCURACY
# #     num_classes = dataset.num_classes
# # else:
# #     metric = Metric.RMSE
# #     num_classes = None

# # gbdt_cls_dict = {
# #     'xgboost': XGBoost,
# #     'catboost': CatBoost,
# #     'lightgbm': LightGBM,
# # }
# # gbdt = gbdt_cls_dict[args.gbdt_type](
# #     task_type=task_type,
# #     num_classes=num_classes,
# #     metric=metric,
# # )

# # if osp.exists(args.saved_model_path):
# #     gbdt.load(args.saved_model_path)
# # else:
# #     gbdt.tune(tf_train=train_dataset.tensor_frame,
# #               tf_val=val_dataset.tensor_frame, num_trials=20)
# #     gbdt.save(args.saved_model_path)

# # pred = gbdt.predict(tf_test=test_dataset.tensor_frame)
# # score = gbdt.compute_metric(test_dataset.tensor_frame.y, pred)
# # print(f"{gbdt.metric} : {score}")

def main(train_df, val_df, test_df, dataset_results, config):
    """
    XGBoost主函數: 運行XGBoost模型，根據任務類型自動選擇評估指標
    
    Args:
        train_df: 訓練集DataFrame
        val_df: 驗證集DataFrame
        test_df: 測試集DataFrame
        dataset_results: 資料集結果與信息
        config: 實驗配置
    
    Returns:
        dict: 實驗結果
    """
    # 這裡可以添加XGBoost模型的運行邏輯
    # 例如，使用XGBoost進行訓練和預測
    print("Running XGBoost model...")
    # print(f"dataset_results: {dataset_results}")

    try:
        import random
        import numpy as np
        import torch
        from torch_frame.datasets import Yandex
        from torch_frame.gbdt import XGBoost
        from torch_frame.typing import Metric, TaskType
        
        print("Running XGBoost model...")
        
        # 獲取配置參數
        dataset_name = dataset_results['dataset']
        task_type_str = dataset_results['info']['task_type']
        train_val_test_split_ratio = config.get('train_val_test_split_ratio', [0.7, 0.09, 0.21])
        seed = config.get('seed', 0)
        num_trials = config.get('num_trials', 20)
        
        # 設置隨機種子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # 設備設置
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 將任務類型字符串轉換為TaskType枚舉
        task_type = None
        if task_type_str.lower() == 'binclass':
            task_type = TaskType.BINARY_CLASSIFICATION
        elif task_type_str.lower() == 'multiclass':
            task_type = TaskType.MULTICLASS_CLASSIFICATION
        elif task_type_str.lower() == 'regression':
            task_type = TaskType.REGRESSION
        else:
            raise ValueError(f"Unknown task type: {task_type_str}")
        
        # 數據集加載和物化
        dataset = Yandex(train_df, val_df, test_df, name=dataset_name, task_type=task_type_str)
        dataset.materialize()
        
        # 直接根據split_col取得train/val/test
        tf_train = dataset.tensor_frame[dataset.df['split_col'] == 0]
        tf_val = dataset.tensor_frame[dataset.df['split_col'] == 1]
        tf_test = dataset.tensor_frame[dataset.df['split_col'] == 2]
        
        # 確定任務類型
        is_classification = dataset.task_type.is_classification
        
        # 設置適當的評估指標和類別數
        num_classes = None
        metric = None
        metric_name = None
        
        if is_classification:
            num_classes = dataset.num_classes
            if num_classes == 2:
                metric = Metric.ROCAUC
                metric_name = 'AUC'
            else:
                metric = Metric.ACCURACY
                metric_name = 'Acc'
        else:
            metric = Metric.RMSE
            metric_name = 'RMSE'
            num_classes = None
        
        print(f"Task type: {task_type_str}, Metric: {metric_name}")
        if is_classification:
            print(f"Number of classes: {num_classes}")
        
        # 創建XGBoost模型（移除 tree_method）
        gbdt = XGBoost(
            task_type=task_type,
            num_classes=num_classes,
            metric=metric,
        )
        # 設定超參數搜尋空間
        search_space = {
            'max_depth': (3, 6),
            'learning_rate': (0.05, 0.2),
            'min_child_weight': (1, 5),
            'subsample': (0.7, 1.0),
            'colsample_bytree': (0.7, 1.0),
        }
        # 調優XGBoost模型（只傳支援的參數）
        print(f"Tuning XGBoost with {num_trials} trials...")
        gbdt.tune(
            tf_train=tf_train,
            tf_val=tf_val, 
            num_trials=num_trials
        )
        
        # 進行預測和評估
        print("Evaluating model...")
        
        # 對訓練集評估
        train_pred = gbdt.predict(tf_test=tf_train)
        train_score = gbdt.compute_metric(tf_train.y, train_pred)
        
        # 對驗證集評估
        val_pred = gbdt.predict(tf_test=tf_val)
        val_score = gbdt.compute_metric(tf_val.y, val_pred)
        
        # 對測試集評估
        test_pred = gbdt.predict(tf_test=tf_test)
        test_score = gbdt.compute_metric(tf_test.y, test_pred)
        
        print(f"Train {metric_name}: {train_score:.4f}")
        print(f"Val {metric_name}: {val_score:.4f}")
        print(f"Test {metric_name}: {test_score:.4f}")
        print("\n")
        # 返回結果
        return {
            'train_metrics': [train_score],
            'val_metrics': [val_score],
            'test_metrics': [test_score],
            'best_val_metric': val_score,
            'best_test_metric': test_score,
            'metric': metric_name,
            'is_classification': is_classification,
            'model': gbdt
        }
    except Exception as e:
        print(f"Error running XGBoost model: {str(e)}")
        # 返回一個基本值的結果
        is_classification = dataset_results['info']['task_type'] == 'classification'
        results = {
            'train_losses': [],
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
            'best_val_metric': float('-inf') if is_classification else float('inf'),
            'best_test_metric': float('-inf') if is_classification else float('inf'),
            'error': str(e),
        }
        return results