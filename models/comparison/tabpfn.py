# # import argparse
# # import os.path as osp

# # import numpy as np
# # import torch
# # from tabpfn import TabPFNClassifier
# # # Please run `pip install tabpfn` to install the package
# # from tqdm import tqdm

# # from torch_frame.data import DataLoader
# # from torch_frame.datasets import (
# #     ForestCoverType,
# #     KDDCensusIncome,
# #     Mushroom,
# #     Titanic,
# # )

# # parser = argparse.ArgumentParser(
# #     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
# # parser.add_argument(
# #     '--dataset', type=str, default="Titanic",
# #     choices=["Titanic", "Mushroom", "ForestCoverType", "KDDCensusIncome"])
# # parser.add_argument('--train_batch_size', type=int, default=4096)
# # parser.add_argument('--test_batch_size', type=int, default=128)
# # parser.add_argument('--seed', type=int, default=0)
# # args = parser.parse_args()

# # torch.manual_seed(args.seed)

# # # Prepare datasets
# # path = osp.join(osp.dirname(osp.realpath(__file__)), '..', 'data',
# #                 args.dataset)

# # if args.dataset == "Titanic":
# #     dataset = Titanic(root=path)
# # elif args.dataset == "ForestCoverType":
# #     dataset = ForestCoverType(root=path)
# # elif args.dataset == "KDDCensusIncome":
# #     dataset = KDDCensusIncome(root=path)
# # else:
# #     dataset = Mushroom(root=path)

# # dataset.materialize()
# # assert dataset.task_type.is_classification
# # dataset = dataset.shuffle()
# # train_dataset, test_dataset = dataset[:0.9], dataset[0.9:]
# # train_tensor_frame = train_dataset.tensor_frame
# # test_tensor_frame = test_dataset.tensor_frame
# # train_loader = DataLoader(
# #     train_tensor_frame,
# #     batch_size=args.train_batch_size,
# #     shuffle=True,
# # )
# # X_train = []
# # train_data = next(iter(train_loader))
# # for stype in train_data.stypes:
# #     X_train.append(train_data.feat_dict[stype])
# # X_train: torch.Tensor = torch.cat(X_train, dim=1)
# # clf = TabPFNClassifier()
# # clf.fit(X_train, train_data.y)
# # test_loader = DataLoader(test_tensor_frame, batch_size=args.test_batch_size)


# # @torch.no_grad()
# # def test() -> float:
# #     accum = total_count = 0
# #     for test_data in tqdm(test_loader):
# #         X_test = []
# #         for stype in train_data.stypes:
# #             X_test.append(test_data.feat_dict[stype])
# #         X_test = torch.cat(X_test, dim=1)
# #         pred: np.ndarray = clf.predict_proba(X_test)
# #         pred_class = pred.argmax(axis=-1)
# #         accum += float((test_data.y.numpy() == pred_class).sum())
# #         total_count += len(test_data.y)

# #     return accum / total_count


# # acc = test()
# # print(f"Accuracy: {acc:.4f}")

# def main(df, dataset_results, config):
#     """
#     主函數，運行TabPFN模型並返回結果
#     Args:
#         df: 資料集DataFrame
#         dataset_results: 資料集結果
#         config: 實驗配置
#     Returns:
#         dict: 實驗結果
#     """
#     # 這裡可以添加TabPFN模型的運行邏輯
#     # 例如，使用TabPFNClassifier進行訓練和預測
#     print("Running TabPFN model...")
#     print(f"df: {df.shape}")
#     # print(f"dataset_results: {dataset_results}")
#     print(f"config: {config}")
def main(train_df, val_df, test_df, dataset_results, config):
    """
    TabPFN主函數: 運行TabPFN模型，根據任務類型自動選擇評估指標
    
    Args:
        train_df: 訓練集DataFrame
        val_df: 驗證集DataFrame
        test_df: 測試集DataFrame
        dataset_results: 資料集結果與信息
        config: 實驗配置
    
    Returns:
        dict: 實驗結果
    """
    try:
        import numpy as np
        import torch
        from tqdm import tqdm
        from torchmetrics import AUROC, Accuracy, MeanSquaredError
        
        print("Running TabPFN model...")
        print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}, Test shape: {test_df.shape}")
        
        # 取得y與X
        target_col = 'target' if 'target' in train_df.columns else train_df.columns[-1]
        X_train = train_df.drop(columns=[target_col]).values
        y_train = train_df[target_col].values
        X_val = val_df.drop(columns=[target_col]).values
        y_val = val_df[target_col].values
        X_test = test_df.drop(columns=[target_col]).values
        y_test = test_df[target_col].values
        
        # 任務類型
        task_type = dataset_results['info']['task_type']
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # 分類/回歸
        is_classification = 'class' in task_type
        if is_classification:
            try:
                from tabpfn import TabPFNClassifier
                clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
                clf.fit(X_train, y_train)
                
                # 分類型任務
                n_classes = len(np.unique(y_train))
                is_binary = n_classes == 2
                if is_binary:
                    metric = AUROC(task='binary').to(device)
                    metric_name = 'AUC'
                else:
                    metric = Accuracy(task='multiclass', num_classes=n_classes).to(device)
                    metric_name = 'Acc'
                
                def eval_metric(X, y):
                    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                    y_tensor = torch.tensor(y).to(device)
                    pred_proba = torch.tensor(clf.predict_proba(X), dtype=torch.float32).to(device)
                    if is_binary:
                        metric.reset()
                        metric.update(pred_proba[:, 1], y_tensor)
                        return metric.compute().item()
                    else:
                        pred_class = pred_proba.argmax(dim=-1)
                        metric.reset()
                        metric.update(pred_class, y_tensor)
                        return metric.compute().item()
            except ImportError:
                print("TabPFN package not found. Please run `pip install tabpfn` to install it.")
                return {
                    'train_metrics': [],
                    'val_metrics': [],
                    'test_metrics': [],
                    'best_val_metric': 0.0,
                    'best_test_metric': 0.0,
                    'error': "TabPFN package not installed"
                }
        else:
            # 迴歸任務
            class PlaceholderRegressor:
                def __init__(self):
                    self.mean = 0
                
                def fit(self, X, y):
                    self.mean = y.mean()
                
                def predict(self, X):
                    return np.ones(len(X)) * self.mean
            
            clf = PlaceholderRegressor()
            clf.fit(X_train, y_train)
            metric = MeanSquaredError().to(device)
            metric_name = 'RMSE'
            
            def eval_metric(X, y):
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
                pred = torch.tensor(clf.predict(X), dtype=torch.float32).to(device)
                metric.reset()
                metric.update(pred, y_tensor)
                return metric.compute().item() ** 0.5
        
        # 計算metric
        train_metric = eval_metric(X_train, y_train)
        val_metric = eval_metric(X_val, y_val)
        test_metric = eval_metric(X_test, y_test)
        
        print(f"Train {metric_name}: {train_metric:.4f}")
        print(f"Val {metric_name}: {val_metric:.4f}")
        print(f"Test {metric_name}: {test_metric:.4f}")
        
        return {
            'train_metrics': [train_metric],
            'val_metrics': [val_metric],
            'test_metrics': [test_metric],
            'best_val_metric': val_metric,
            'best_test_metric': test_metric,
            'metric': metric_name,
            'is_classification': is_classification
        }
    except Exception as e:
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