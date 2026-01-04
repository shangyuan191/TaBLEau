"""
DGM (Differentiable Graph Module) for Tabular Data
參考論文: "Differentiable Graph Module (DGM) for Graph Convolutional Networks"

DGM 是一個原生 GNN 模型，其核心特色是動態圖生成模組(DGM_d)，
能夠根據輸入特徵自動學習最佳的圖結構。

在 PyTorch-Frame 的五階段框架中，DGM 的位置分析：
1. materialize: 不涉及（DGM 直接操作原始特徵）
2. encoding: DGM_d 的 embed_f 將輸入特徵編碼到圖空間
3. columnwise: DGM_d 動態構建圖結構 + GNN 層進行節點間信息傳遞
4. decoding: 最終的 MLP 層將圖表徵解碼為預測

因此，DGM 的 GNN 主要相當於插在 encoding 和 columnwise 階段，
並且是端到端聯合訓練圖結構學習與節點表徵學習。
"""

import sys
import os
import time
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error, r2_score
import logging

logger = logging.getLogger(__name__)


def collate_dgm_full(batch):
    """自定义 collate 函数用于完整 DGM（处理 None edges）"""
    X, y, mask, edges = batch[0]
    return X.unsqueeze(0), y.unsqueeze(0), mask.unsqueeze(0), edges


def collate_dgm_simplified(batch):
    """自定义 collate 函数用于简化版 DGM（处理字典格式）"""
    # batch 是字典列表，但 __len__=1，所以只有一个元素
    data = batch[0]
    return {
        'X': data['X'].unsqueeze(0),  # [N, D] -> [1, N, D]
        'y': data['y'].unsqueeze(0),  # [N, C] -> [1, N, C]
        'mask': data['mask'].unsqueeze(0)  # [N] -> [1, N]
    }


# 先嘗試導入 PyG 的基礎組件（SimpleDGM 需要）
try:
    from torch_geometric.nn import GCNConv, GATConv
    PYG_AVAILABLE = True
except Exception as e:
    PYG_AVAILABLE = False
    logger.warning(f"PyTorch Geometric 導入失敗: {e}")

# 嘗試導入完整 DGM 相關模組
try:
    # 添加 DGM_pytorch 到路徑
    dgm_path = '/home/skyler/ModelComparison/DGM_pytorch'
    if dgm_path not in sys.path:
        sys.path.insert(0, dgm_path)
    
    # 添加 keops 到路徑（DGM 依賴）
    keops_path = os.path.join(dgm_path, 'keops')
    if keops_path not in sys.path:
        sys.path.insert(0, keops_path)
    
    from DGMlib.model_dDGM import DGM_Model
    from DGMlib.layers import DGM_d
    DGM_AVAILABLE = True
    logger.info("完整 DGM 模組導入成功")
except Exception as e:
    DGM_AVAILABLE = False
    logger.warning(f"完整 DGM 模組導入失敗: {e}")
    if PYG_AVAILABLE:
        logger.info("將使用簡化版 GNN 實現（基於 PyG）")
    else:
        logger.error("PyG 和 DGM 都不可用，無法運行模型")


class TabularDataset(Dataset):
    """表格數據集包裝器，用於 DGM 模型"""
    
    def __init__(self, X, y, mask=None, use_full_dgm=False):
        """
        Args:
            X: 特徵矩陣 [N, D]
            y: 標籤 [N] 或 [N, C]（one-hot）
            mask: 可選的掩碼 [N]
            use_full_dgm: 是否使用完整 DGM（需要返回 edges）
        """
        self.X = torch.FloatTensor(X)
        self.use_full_dgm = use_full_dgm
        
        # 處理標籤
        if len(y.shape) == 1:
            # 轉換為 one-hot
            num_classes = len(np.unique(y))
            y_onehot = np.eye(num_classes)[y.astype(int)]
            self.y = torch.FloatTensor(y_onehot)
        else:
            self.y = torch.FloatTensor(y)
        
        if mask is None:
            self.mask = torch.ones(len(X), dtype=torch.bool)
        else:
            self.mask = torch.BoolTensor(mask)
    
    def __len__(self):
        return 1  # DGM 使用 transductive learning，一次處理整個圖
    
    def __getitem__(self, idx):
        # DGM 使用 transductive learning，返回整個圖的數據
        # 返回時不添加額外維度，DataLoader 會自動 batch
        if self.use_full_dgm:
            # 完整 DGM 需要 edges（設置為 None，會在模型中生成）
            return (self.X, self.y, self.mask, None)
        else:
            # 簡化 DGM 使用 dict 格式
            return {
                'X': self.X,    # [N, D]
                'y': self.y,    # [N, C]
                'mask': self.mask  # [N]
            }


class SimpleDGM(nn.Module):
    """
    簡化版 DGM 模型（當無法導入完整 DGM 時使用）
    """
    
    def __init__(self, in_dim, hidden_dim=128, num_classes=2, k=5, num_layers=2, dropout=0.2):
        super().__init__()
        
        self.k = k
        self.num_layers = num_layers
        
        # 特徵編碼層
        self.pre_fc = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # GCN 層
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        # 預測頭
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        self.dropout = dropout
    
    def build_knn_graph(self, x, k):
        """構建 k-NN 圖"""
        # x: [B, N, D]
        B, N, D = x.shape
        
        # 計算距離矩陣
        dist = torch.cdist(x, x)  # [B, N, N]
        
        # 對每個節點找 k 個最近鄰
        _, indices = torch.topk(dist, k + 1, dim=-1, largest=False)
        indices = indices[:, :, 1:]  # 去掉自己 [B, N, k]
        
        # 構建邊索引
        src = torch.arange(N, device=x.device).unsqueeze(1).expand(N, k).reshape(-1)
        dst = indices[0].reshape(-1)  # 假設 batch_size=1
        
        edge_index = torch.stack([src, dst], dim=0)
        
        # 添加反向邊
        edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
        
        return edge_index
    
    def forward(self, x, edges=None):
        """
        Args:
            x: [B, N, D] 特徵矩陣
            edges: 可選的預定義邊
        
        Returns:
            pred: [B, N, num_classes] 預測
            None: 佔位符（與完整 DGM 保持接口一致）
        """
        # 確保輸入是 3D 的 [B, N, D]
        if len(x.shape) == 4:
            # 如果是 [1, B, N, D]，移除第一維
            x = x.squeeze(0)
        elif len(x.shape) == 2:
            # 如果是 [N, D]，添加 batch 維度
            x = x.unsqueeze(0)
        
        B, N, D = x.shape
        
        # 特徵編碼
        x = self.pre_fc(x)  # [B, N, hidden_dim]
        
        # 構建或使用現有的圖
        if edges is None:
            edges = self.build_knn_graph(x, self.k)
        
        # 移除 batch 維度進行 GCN
        x_flat = x.view(-1, x.size(-1))  # [B*N, hidden_dim]
        
        # GCN 層
        for conv in self.convs:
            x_flat = F.relu(conv(x_flat, edges))
            x_flat = F.dropout(x_flat, p=self.dropout, training=self.training)
        
        # 恢復 batch 維度
        x = x_flat.view(B, N, -1)  # [B, N, hidden_dim]
        
        # 預測
        pred = self.fc(x)  # [B, N, num_classes]
        
        return pred, None


def train_dgm_model(
    train_loader, 
    val_loader, 
    test_loader,
    model,
    device,
    epochs=200,
    lr=1e-3,
    patience=10,
    is_classification=True,
    use_full_dgm=False
):
    """
    訓練 DGM 模型
    
    Args:
        train_loader: 訓練數據加載器
        val_loader: 驗證數據加載器  
        test_loader: 測試數據加載器
        model: DGM 模型
        device: 設備
        epochs: 訓練輪數
        lr: 學習率
        patience: 早停耐心值
        is_classification: 是否為分類任務
        use_full_dgm: 是否使用完整 DGM
    
    Returns:
        dict: 訓練結果
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_val_metric = float('-inf') if is_classification else float('inf')
    best_test_metric = float('-inf') if is_classification else float('inf')
    patience_counter = 0
    
    train_metrics_history = []
    val_metrics_history = []
    test_metrics_history = []
    
    for epoch in range(epochs):
        # 訓練
        model.train()
        
        if use_full_dgm:
            # 完整 DGM 使用 tuple 格式
            for X, y, mask, edges in train_loader:
                X = X.to(device)
                y = y.to(device)
                mask = mask.to(device)
                
                optimizer.zero_grad()
                
                # 完整 DGM 前向傳播
                pred, lprobs = model(X, edges)  # pred: [B, N, C]
                
                # 只對訓練節點計算損失
                pred_flat = pred[0]  # [N, C]
                y_flat = y[0]  # [N, C]
                mask_flat = mask[0]  # [N]
                train_pred = pred_flat[mask_flat]  # [N_train, C]
                train_y = y_flat[mask_flat]  # [N_train, C]
                
                if is_classification:
                    loss = F.binary_cross_entropy_with_logits(train_pred, train_y)
                else:
                    loss = F.mse_loss(train_pred, train_y)
                
                loss.backward()
                optimizer.step()
        else:
            # 簡化 DGM 使用 dict 格式
            for batch in train_loader:
                X = batch['X'].to(device)
                y = batch['y'].to(device)
                mask = batch['mask'].to(device)
                
                optimizer.zero_grad()
                
                # 前向傳播（edges=None，模型會自動構建）
                # DataLoader 會添加 batch 維度: X: [B, N, D]
                pred, _ = model(X, None)  # pred: [B, N, C]
                
                # 只對訓練節點計算損失
                # mask: [B, N] after DataLoader batching
                # 對於 transductive learning batch_size=1，所以 B=1
                pred_flat = pred[0]  # [N, C]
                y_flat = y[0]  # [N, C]
                mask_flat = mask[0]  # [N]
                train_pred = pred_flat[mask_flat]  # [N_train, C]
                train_y = y_flat[mask_flat]  # [N_train, C]
                
                if is_classification:
                    loss = F.binary_cross_entropy_with_logits(train_pred, train_y)
                else:
                    loss = F.mse_loss(train_pred, train_y)
                
                loss.backward()
                optimizer.step()
        
        # 評估（每個 epoch 評估一次以支持真正的早停）
        train_metric, val_metric, test_metric = evaluate_dgm(
            model, train_loader, val_loader, test_loader, device, is_classification, use_full_dgm
        )
        
        train_metrics_history.append(train_metric)
        val_metrics_history.append(val_metric)
        test_metrics_history.append(test_metric)
        
        # 每 10 個 epoch 或最後一次 epoch 時打印日誌
        if (epoch + 1) % 10 == 0 or epoch == epochs - 1:
            logger.info(f"Epoch {epoch+1}/{epochs} - "
                       f"Train: {train_metric:.4f}, "
                       f"Val: {val_metric:.4f}, "
                       f"Test: {test_metric:.4f}")
        
        # 早停判斷（每個 epoch 檢查）
        if is_classification:
            improved = val_metric > best_val_metric
        else:
            improved = val_metric < best_val_metric
        
        if improved:
            best_val_metric = val_metric
            best_test_metric = test_metric
            patience_counter = 0
        else:
            patience_counter += 1
        
        # 當超過 patience epoch 數時停止
        if patience_counter >= patience:
            logger.info(f"Early stopping at epoch {epoch+1} (patience={patience})")
            break
    
    return {
        'train_metrics': train_metrics_history,
        'val_metrics': val_metrics_history,
        'test_metrics': test_metrics_history,
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'patience_counter': patience_counter,  # 沒有改進的連續 epoch 數
        'stopped_epoch': epoch + 1,  # 實際停止的 epoch 編號
    }


def evaluate_dgm(model, train_loader, val_loader, test_loader, device, is_classification=True, use_full_dgm=False):
    """評估 DGM 模型"""
    model.eval()
    
    metrics = []
    
    with torch.no_grad():
        for loader in [train_loader, val_loader, test_loader]:
            if use_full_dgm:
                # 完整 DGM 使用 tuple 格式
                for X, y, mask, edges in loader:
                    X = X.to(device)
                    y = y.to(device)
                    mask = mask.to(device)
                    
                    pred, _ = model(X, edges)  # pred: [1, N, C]
                    
                    # 提取對應集合的預測和標籤
                    pred_flat = pred[0]  # [N, C]
                    y_flat = y[0]  # [N, C]
                    mask_flat = mask[0]  # [N]
                    pred_set = pred_flat[mask_flat].cpu().numpy()  # [N_subset, C]
                    y_set = y_flat[mask_flat].cpu().numpy()  # [N_subset, C]
                    break  # transductive setting，一次處理整個數據集
            else:
                # 簡化 DGM 使用 dict 格式
                for batch in loader:
                    X = batch['X'].to(device)
                    y = batch['y'].to(device)
                    mask = batch['mask'].to(device)
                    
                    pred, _ = model(X, None)  # pred: [1, N, C]
                    
                    # 提取對應集合的預測和標籤
                    pred_flat = pred[0]  # [N, C]
                    y_flat = y[0]  # [N, C]
                    mask_flat = mask[0]  # [N]
                    pred_set = pred_flat[mask_flat].cpu().numpy()  # [N_subset, C]
                    y_set = y_flat[mask_flat].cpu().numpy()  # [N_subset, C]
                    break  # transductive setting，一次處理整個數據集
            
            # 計算指標
            if is_classification:
                # 分類任務
                y_pred_classes = pred_set.argmax(-1)
                y_true_classes = y_set.argmax(-1)
                
                if y_set.shape[-1] == 2:  # 二分類
                    # 使用 AUC
                    y_scores = torch.softmax(torch.FloatTensor(pred_set), dim=-1)[:, 1]
                    metric = roc_auc_score(y_true_classes, y_scores.numpy())
                else:  # 多分類
                    metric = accuracy_score(y_true_classes, y_pred_classes)
            else:
                # 迴歸任務 - 計算 RMSE
                mse = mean_squared_error(y_set.flatten(), pred_set.flatten())
                metric = np.sqrt(mse)  # RMSE
            
            metrics.append(metric)
    
    return metrics[0], metrics[1], metrics[2]


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage=None):
    """
    DGM 主函數: 運行 DGM 模型並返回結果
    
    Args:
        train_df: 訓練集 DataFrame
        val_df: 驗證集 DataFrame
        test_df: 測試集 DataFrame
        dataset_results: 資料集結果與信息
        config: 實驗配置
        gnn_stage: GNN 階段（DGM 不使用此參數，因為它是原生 GNN 模型，保留以兼容框架）
    
    Returns:
        dict: 實驗結果
    """
    try:
        logger.info("Running DGM model...")
        
        # 檢查依賴是否可用
        if not DGM_AVAILABLE and not PYG_AVAILABLE:
            raise ImportError("DGM 和 PyTorch Geometric 都不可用。請安裝其中之一：\n"
                            "  - 完整 DGM: 需要 pytorch_lightning 和 keops\n"
                            "  - 簡化版本: 需要 torch_geometric")
        
        # 獲取配置參數
        dataset_name = dataset_results['dataset']
        task_type = dataset_results['info']['task_type']
        seed = config.get('seed', 42)
        epochs = config.get('epochs', 200)
        lr = config.get('lr', 1e-3)
        patience = config.get('patience', 10)
        hidden_dim = config.get('gnn_hidden_dim', 128)
        num_layers = config.get('gnn_layers', 2)
        k = config.get('dgm_k', 10)  # k-NN 中的 k，從 dgm_k 參數讀取
        dropout = config.get('gnn_dropout', 0.2)
        
        # 設置隨機種子
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
        
        # 設備設置
        if 'device' in config:
            device = config['device']
        else:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {device}")
        
        # 確定任務類型
        is_classification = task_type.lower() in ['binclass', 'multiclass']
        
        # 合併所有數據（transductive setting）
        all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        
        # 分離特徵和標籤
        if 'target' in all_df.columns:
            target_col = 'target'
        elif 'label' in all_df.columns:
            target_col = 'label'
        else:
            # 假設最後一列是目標
            target_col = all_df.columns[-1]
        
        X = all_df.drop(columns=[target_col]).values
        y = all_df[target_col].values
        
        # 特徵標準化
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # 編碼標籤（如果需要）
        if is_classification:
            le = LabelEncoder()
            y = le.fit_transform(y)
            num_classes = len(np.unique(y))
        else:
            num_classes = 1
            y = y.reshape(-1, 1).astype(np.float32)
        
        # 創建掩碼
        n_train = len(train_df)
        n_val = len(val_df)
        n_test = len(test_df)
        
        train_mask = np.zeros(len(all_df), dtype=bool)
        val_mask = np.zeros(len(all_df), dtype=bool)
        test_mask = np.zeros(len(all_df), dtype=bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train+n_val] = True
        test_mask[n_train+n_val:] = True
        
        # 創建數據集 (根據是否使用完整 DGM 決定格式)
        # 創建模型前先检查内存需求
        in_dim = X.shape[1]
        num_samples = X.shape[0]
        
        # === 内存检查：大数据集自动降级到简化版本 ===
        # DGM 使用 transductive learning (batch_size=1)，需要 O(N²) 内存
        # 估算：N×N×4 bytes (float32)
        estimated_memory_gb = (num_samples ** 2 * 4) / (1024 ** 3)
        memory_threshold_gb = 10.0  # 设置阈值为 10GB
        
        use_full_dgm = DGM_AVAILABLE
        if DGM_AVAILABLE and estimated_memory_gb > memory_threshold_gb:
            logger.warning(
                f"数据集过大 (N={num_samples}, 估算需要 {estimated_memory_gb:.2f} GB 内存)，"
                f"超过阈值 {memory_threshold_gb} GB。自动降级到简化版 DGM（固定 k-NN 图）"
            )
            use_full_dgm = False  # 强制使用简化版本
        
        # 創建數據集
        train_dataset = TabularDataset(X, y, train_mask, use_full_dgm=use_full_dgm)
        val_dataset = TabularDataset(X, y, val_mask, use_full_dgm=use_full_dgm)
        test_dataset = TabularDataset(X, y, test_mask, use_full_dgm=use_full_dgm)
        
        # 創建數據加載器
        if use_full_dgm:
            # 完整 DGM 需要自定义 collate_fn 来处理 None edges
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_dgm_full)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_dgm_full)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_dgm_full)
        else:
            # 简化版本使用自定义 collate 处理字典格式
            train_loader = DataLoader(train_dataset, batch_size=1, shuffle=False, collate_fn=collate_dgm_simplified)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, collate_fn=collate_dgm_simplified)
            test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, collate_fn=collate_dgm_simplified)
        
        if use_full_dgm:
            # 使用完整的 DGM 模型
            from argparse import Namespace
            
            hparams = Namespace(
                pre_fc=[in_dim, hidden_dim],
                dgm_layers=[[hidden_dim, hidden_dim]],  # DGM_d 的特征变换层
                conv_layers=[[hidden_dim, hidden_dim]],  # GNN 卷积层
                fc_layers=[hidden_dim, hidden_dim // 2, num_classes],  # 最终分类层
                k=k,
                distance='euclidean',
                ffun='knn',  # 使用 'knn' 避免初始时 edges=None 的问题
                gfun='gcn',   # node_g 使用 GCN
                pooling='max',
                dropout=dropout,
                lr=lr,
                test_eval=1
            )
            
            model = DGM_Model(hparams)
            logger.info(f"使用完整 DGM 模型（动态图学习，N={num_samples}）")
        elif PYG_AVAILABLE:
            # 使用簡化版 GNN
            model = SimpleDGM(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                num_classes=num_classes,
                k=k,
                num_layers=num_layers,
                dropout=dropout
            )
            logger.info("使用簡化 GNN 模型（基於 PyG）")
        else:
            raise ImportError("無可用的模型實現")
        
        # 訓練模型
        logger.info(f"Training DGM with {epochs} epochs...")
        results = train_dgm_model(
            train_loader=train_loader,
            val_loader=val_loader,
            test_loader=test_loader,
            model=model,
            device=device,
            epochs=epochs,
            lr=lr,
            patience=patience,
            is_classification=is_classification,
            use_full_dgm=use_full_dgm
        )
        
        # 確定指標名稱
        if is_classification:
            metric_name = 'AUC' if num_classes == 2 else 'Acc'
        else:
            metric_name = 'RMSE'
        
        logger.info(f"Best Val {metric_name}: {results['best_val_metric']:.4f}")
        logger.info(f"Best Test {metric_name}: {results['best_test_metric']:.4f}")
        
        # 返回結果
        return {
            'train_metrics': results['train_metrics'],
            'val_metrics': results['val_metrics'],
            'test_metrics': results['test_metrics'],
            'best_val_metric': results['best_val_metric'],
            'best_test_metric': results['best_test_metric'],
            'metric': metric_name,
            'is_classification': is_classification,
            'model': model,
            'early_stop_epochs': results['stopped_epoch'],  # 訓練停止的實際 epoch 編號
            'patience_counter': results['patience_counter'],  # 沒有改進的連續 epoch 數
        }
        
    except Exception as e:
        logger.error(f"Error running DGM model: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        
        # 返回錯誤結果
        is_classification = dataset_results['info']['task_type'].lower() in ['binclass', 'multiclass']
        return {
            'train_metrics': [],
            'val_metrics': [],
            'test_metrics': [],
            'best_val_metric': float('-inf') if is_classification else float('inf'),
            'best_test_metric': float('-inf') if is_classification else float('inf'),
            'error': str(e),
        }


if __name__ == "__main__":
    # 測試代碼
    print("DGM model wrapper for TaBLEau framework")
    print(f"DGM library available: {DGM_AVAILABLE}")




#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models dgm --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models dgm --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models dgm --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models dgm --gnn_stages all --epochs 2
#  python main.py --dataset helena --models dgm --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models dgm --gnn_stages all --epochs 2