
def main(train_df, val_df, test_df, dataset_results, config,gnn_stage=None):
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

        # helper: subsample large training sets to avoid TabPFN sample-limit warnings
        def _maybe_subsample(X, y, max_samples=10000, is_classification=True, random_state=42):
            """Return (X_sub, y_sub). If len(X) <= max_samples, return originals.

            For classification we perform stratified sampling to preserve class proportions.
            For regression we bin targets into quantiles and stratify by bins.
            """
            n = len(X)
            if n <= max_samples:
                return X, y
            try:
                from sklearn.model_selection import train_test_split
            except Exception:
                # if sklearn not available, simple random subsample
                import numpy as _np
                idx = _np.random.RandomState(random_state).permutation(n)[:max_samples]
                return X[idx], y[idx]

            if is_classification:
                # stratify by class labels
                X_sub, _, y_sub, _ = train_test_split(X, y, train_size=max_samples, stratify=y, random_state=random_state)
                return X_sub, y_sub
            else:
                # regression: bin targets into quantiles for approximate stratification
                import numpy as _np
                try:
                    n_bins = min(50, max(2, int(max_samples ** 0.25)))
                    bins = _np.quantile(y, q=_np.linspace(0, 1, n_bins + 1))
                    # digitize may produce 0..n_bins; map to 0..n_bins-1
                    y_binned = _np.digitize(y, bins[1:-1], right=True)
                    X_sub, _, y_sub, _ = train_test_split(X, y, train_size=max_samples, stratify=y_binned, random_state=random_state)
                    return X_sub, y_sub
                except Exception:
                    idx = _np.random.RandomState(random_state).permutation(n)[:max_samples]
                    return X[idx], y[idx]
        
        # 任務類型
                # determine max samples (config may be dict)
                max_samples = config.get('tabpfn_max_samples', 10000) if isinstance(config, dict) else getattr(config, 'tabpfn_max_samples', 10000)

        task_type = dataset_results['info']['task_type']
        device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        print(f"Using device: {device}")
        
        # 分類/回歸
        is_classification = 'class' in task_type
        if is_classification:
            try:
                from tabpfn import TabPFNClassifier
                clf = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
                # TabPFN has a recommended maximum number of training samples (e.g. 10000).
                # Compute local max_samples to avoid NameError if outer var missing.
                ms = config.get('tabpfn_max_samples', 10000) if isinstance(config, dict) else getattr(config, 'tabpfn_max_samples', 10000)
                X_fit, y_fit = _maybe_subsample(X_train, y_train, max_samples=ms, is_classification=True)
                if len(X_fit) < len(X_train):
                    print(f"TabPFN: subsampled training set from {len(X_train)} to {len(X_fit)} samples to respect model limits.")
                # Attempt to fit; if TabPFN raises about too many classes, try to use ManyClassClassifier wrapper
                try:
                    clf.fit(X_fit, y_fit)
                except Exception as e_fit:
                    msg = str(e_fit)
                    if 'exceeds the maximal number of classes' in msg or 'Number of classes' in msg and 'exceeds' in msg:
                        print('TabPFN reported too many classes. Attempting to use tabpfn_extensions.many_class.ManyClassClassifier as a wrapper...')
                        try:
                            from tabpfn_extensions.many_class import ManyClassClassifier
                            mc = ManyClassClassifier(estimator=clf)
                            mc.fit(X_fit, y_fit)
                            clf = mc
                            print('ManyClassClassifier wrapper fitted successfully and will be used for prediction.')
                        except Exception as e_wrap:
                            print('Failed to use ManyClassClassifier wrapper:', e_wrap)
                            # re-raise original exception so caller sees the root cause
                            raise
                    else:
                        raise
                
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
            except Exception as e:
                # capture detailed import error for easier debugging in logs
                import traceback
                tb = traceback.format_exc()
                print("TabPFN import or init failed:", e)
                print(tb)
                return {
                    'train_metrics': [],
                    'val_metrics': [],
                    'test_metrics': [],
                    'best_val_metric': 0.0,
                    'best_test_metric': 0.0,
                    'error': str(e),
                    'error_trace': tb,
                }
        else:
            # 迴歸任務 - 優先使用 TabPFNRegressor，若不可用則退回到簡單 placeholder
            try:
                from tabpfn import TabPFNRegressor
                # TabPFNRegressor may accept device args in newer versions; keep defaults
                clf = TabPFNRegressor()
                # Subsample large training sets for regressor too
                ms = config.get('tabpfn_max_samples', 10000) if isinstance(config, dict) else getattr(config, 'tabpfn_max_samples', 10000)
                X_fit, y_fit = _maybe_subsample(X_train, y_train, max_samples=ms, is_classification=False)
                if len(X_fit) < len(X_train):
                    print(f"TabPFNRegressor: subsampled training set from {len(X_train)} to {len(X_fit)} samples to respect model limits.")
                # fit on training data
                clf.fit(X_fit, y_fit)
            except Exception:
                # fallback simple regressor (keeps previous behaviour if tabpfn not installed)
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
                # Ensure predictions are numpy-like then convert to tensor for metric
                preds = clf.predict(X)
                preds = np.asarray(preds)
                X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
                y_tensor = torch.tensor(y, dtype=torch.float32).to(device)
                pred_tensor = torch.tensor(preds, dtype=torch.float32).to(device)
                metric.reset()
                metric.update(pred_tensor, y_tensor)
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