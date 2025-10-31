"""
TabM (Tabular Model with Multiple predictions) 集成到 TaBLEau 框架

五階段拆分設計：
- start: dummy stage（為了讓 GNN 能在 materialize 之前插入）
- materialize: 數據預處理 + 特徵嵌入，GNN 在這裡完整訓練並轉換所有數據
- encoding: EnsembleView + Backbone 前半部分，GNN 與這些層一起訓練
- columnwise: Backbone 後半部分，GNN 與這些層一起訓練
- decoding: 訓練完 encoder+columnwise 後，將所有數據通過並用 GNN 作為 decoder
"""

import sys
import os
from pathlib import Path

# 添加 TabM 包的路徑
tabm_path = '/home/shangyuan/ModelComparison/tabm'
if tabm_path not in sys.path:
    sys.path.insert(0, tabm_path)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.neighbors import NearestNeighbors
from torch_geometric.nn import GCNConv
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any

# TabM 相關導入
import tabm
from tabm import (
    TabM,
    EnsembleView,
    LinearEnsemble,
    MLPBackboneBatchEnsemble,
    ElementwiseAffine
)
import rtdl_num_embeddings
from rtdl_num_embeddings import (
    PiecewiseLinearEmbeddings,
    PeriodicEmbeddings,
    LinearReLUEmbeddings
)


class SimpleGCN(torch.nn.Module):
    """簡單的 GCN 用於 GNN 插入"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def knn_graph(x, k):
    """構建 KNN 圖"""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    N = x_np.shape[0]
    
    # 防護：如果樣本數太少，調整 k 值
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    
    # k 不能大於等於樣本數
    actual_k = min(k, N - 1)
    
    nbrs = NearestNeighbors(n_neighbors=actual_k+1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_index = []
    for i in range(N):
        for j in indices[i][1:]:
            edge_index.append([i, j])
    
    if len(edge_index) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def start_fn(train_df, val_df, test_df):
    """
    Start 階段 - dummy stage
    不做任何處理，只是為了讓 GNN 能在 materialize 之前插入
    """
    return train_df, val_df, test_df


def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    """
    在 start 和 materialize 之間插入 GNN
    GNN 完整訓練並將所有數據轉換為 node embedding，再組回 df 往下游傳遞
    """
    print("Executing GNN between start_fn and materialize_fn (TabM)")
    device = config.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    patience = config.get('gnn_patience', 10)
    
    # 合併三個 df
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values

    # 標籤處理
    if task_type == 'binclass':
        y = torch.tensor(y, dtype=torch.float32, device=device)
    elif task_type == 'multiclass':
        y = torch.tensor(y, dtype=torch.long, device=device)
    else:  # regression
        y = torch.tensor(y, dtype=torch.float32, device=device)

    # 建圖
    edge_index = knn_graph(x, k).to(device)
    in_dim = x.shape[1]
    out_dim = in_dim

    # 訓練 GNN（重構任務）
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    
    best_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    
    gnn.train()
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        loss = F.mse_loss(out, x)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if (epoch+1) % 10 == 0:
            print(f'GNN Epoch {epoch+1}/{gnn_epochs}, Loss: {loss.item():.4f}')
            
        if early_stop_counter >= patience:
            print(f"GNN Early stopping at epoch {epoch+1}")
            gnn_early_stop_epochs = epoch + 1
            break
    
    # 獲取最終嵌入
    gnn.eval()
    with torch.no_grad():
        final_emb = gnn(x, edge_index).cpu().numpy()
    
    # 分回三個 df
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    
    train_emb = final_emb[:n_train]
    val_emb = final_emb[n_train:n_train+n_val]
    test_emb = final_emb[n_train+n_val:]

    emb_cols = [f'N_feature_{i}' for i in range(1, out_dim+1)]
    train_df_gnn = pd.DataFrame(train_emb, columns=emb_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_emb, columns=emb_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_emb, columns=emb_cols, index=test_df.index)

    # 保留原標籤
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values

    return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs


def gnn_after_materialize_fn(X_train, y_train, X_val, y_val, X_test, y_test, config, task_type):
    """
    在 materialize 後插入 GNN
    GNN 完整訓練並將所有數據轉換為 node embedding
    """
    print("Executing GNN after materialize_fn (TabM)")
    device = config['device']
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    patience = config.get('gnn_patience', 10)
    
    # 合併所有數據
    X_all = torch.cat([X_train, X_val, X_test], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)
    
    # 建圖
    edge_index = knn_graph(X_all, k).to(device)
    in_dim = X_all.shape[1]
    out_dim = in_dim
    
    # 訓練 GNN
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    
    best_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    
    gnn.train()
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(X_all, edge_index)
        loss = F.mse_loss(out, X_all)
        loss.backward()
        optimizer.step()
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            
        if (epoch+1) % 10 == 0:
            print(f'GNN Epoch {epoch+1}/{gnn_epochs}, Loss: {loss.item():.4f}')
            
        if early_stop_counter >= patience:
            print(f"GNN Early stopping at epoch {epoch+1}")
            gnn_early_stop_epochs = epoch + 1
            break
    
    # 獲取最終嵌入
    gnn.eval()
    with torch.no_grad():
        X_all_gnn = gnn(X_all, edge_index)
    
    # 分回三個集合
    n_train = len(X_train)
    n_val = len(X_val)
    
    X_train_gnn = X_all_gnn[:n_train]
    X_val_gnn = X_all_gnn[n_train:n_train+n_val]
    X_test_gnn = X_all_gnn[n_train+n_val:]
    
    return X_train_gnn, X_val_gnn, X_test_gnn, gnn_early_stop_epochs


def materialize_fn(train_df, val_df, test_df, dataset_results, config):
    """
    Materialize 階段
    - 數據預處理（QuantileTransformer）
    - 創建特徵嵌入（PiecewiseLinearEmbeddings）
    - 標籤處理
    """
    print("TabM Materializing dataset...")
    device = config['device']
    task_type = dataset_results['info']['task_type']
    
    # 提取特徵和標籤
    X_train = train_df.iloc[:, :-1].values.astype(np.float32)
    y_train = train_df.iloc[:, -1].values
    X_val = val_df.iloc[:, :-1].values.astype(np.float32)
    y_val = val_df.iloc[:, -1].values
    X_test = test_df.iloc[:, :-1].values.astype(np.float32)
    y_test = test_df.iloc[:, -1].values
    
    n_num_features = X_train.shape[1]
    
    # 檢查並移除常數列（只有一個唯一值的列）
    unique_counts = np.array([len(np.unique(X_train[:, i])) for i in range(X_train.shape[1])])
    valid_cols = unique_counts > 1
    
    if not valid_cols.all():
        print(f"Warning: Removing {(~valid_cols).sum()} constant columns")
        X_train = X_train[:, valid_cols]
        X_val = X_val[:, valid_cols]
        X_test = X_test[:, valid_cols]
        n_num_features = X_train.shape[1]
    
    # 數值特徵標準化
    noise = np.random.default_rng(0).normal(0.0, 1e-5, X_train.shape).astype(np.float32)
    
    # 動態調整 n_quantiles：不能超過訓練樣本數
    n_quantiles = max(min(len(X_train) // 30, 1000), 10)
    n_quantiles = min(n_quantiles, len(X_train))
    
    preprocessor = QuantileTransformer(
        n_quantiles=n_quantiles,
        output_distribution='normal',
        subsample=10**9
    ).fit(X_train + noise)
    
    X_train = preprocessor.transform(X_train).astype(np.float32)
    X_val = preprocessor.transform(X_val).astype(np.float32)
    X_test = preprocessor.transform(X_test).astype(np.float32)
    
    # 標籤處理
    if task_type == 'regression':
        y_mean, y_std = y_train.mean(), y_train.std()
        y_train_normalized = (y_train - y_mean) / y_std
        n_classes = None
        label_stats = {'mean': y_mean, 'std': y_std}
    else:
        y_train_normalized = y_train
        n_classes = len(np.unique(y_train))
        label_stats = None
    
    # 轉換為 tensor
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32, device=device)
    y_train_tensor = torch.tensor(y_train_normalized, dtype=torch.float32 if task_type == 'regression' else torch.long, device=device)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32, device=device)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32 if task_type == 'regression' else torch.long, device=device)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32, device=device)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32 if task_type == 'regression' else torch.long, device=device)
    
    # 創建特徵嵌入（使用 PiecewiseLinearEmbeddings）
    # 動態調整 n_bins：必須 < len(X_train) 且 > 1
    n_bins = min(48, len(X_train_tensor) - 1)
    n_bins = max(n_bins, 2)  # 至少 2 個 bins
    
    num_embeddings = PiecewiseLinearEmbeddings(
        rtdl_num_embeddings.compute_bins(
            X_train_tensor,
            n_bins=n_bins
        ),
        d_embedding=16,
        activation=False,
        version='B'
    ).to(device)
    
    return {
        'X_train': X_train_tensor,
        'y_train': y_train_tensor,
        'X_val': X_val_tensor,
        'y_val': y_val_tensor,
        'X_test': X_test_tensor,
        'y_test': y_test_tensor,
        'n_num_features': n_num_features,
        'n_classes': n_classes,
        'task_type': task_type,
        'label_stats': label_stats,
        'num_embeddings': num_embeddings,
        'device': device
    }


def tabm_core_fn(material_outputs, config, gnn_stage):
    """
    TabM 核心訓練函數
    根據 gnn_stage 決定如何整合 GNN
    """
    device = material_outputs['device']
    task_type = material_outputs['task_type']
    n_classes = material_outputs['n_classes']
    num_embeddings = material_outputs['num_embeddings']
    
    X_train = material_outputs['X_train']
    y_train = material_outputs['y_train']
    X_val = material_outputs['X_val']
    y_val = material_outputs['y_val']
    X_test = material_outputs['X_test']
    y_test = material_outputs['y_test']
    
    # TabM 超參數
    k = config.get('tabm_k', 32)
    n_blocks = config.get('tabm_n_blocks', 2)
    d_block = config.get('tabm_d_block', 512)
    dropout = config.get('tabm_dropout', 0.1)
    
    # 計算輸入維度（經過 embedding 後）
    with torch.no_grad():
        sample_x = X_train[:1]
        if num_embeddings is not None:
            embedded = num_embeddings(sample_x)
            d_in = embedded.shape[-1] * embedded.shape[-2]
        else:
            d_in = material_outputs['n_num_features']
    
    # 創建 TabM 組件
    ensemble_view = EnsembleView(k=k)
    
    # 創建 Backbone 並分割
    backbone = MLPBackboneBatchEnsemble(
        d_in=d_in,
        n_blocks=n_blocks,
        d_block=d_block,
        dropout=dropout,
        k=k,
        tabm_init=True,
        scaling_init='normal',
        start_scaling_init_chunks=None,
    ).to(device)
    
    n_blocks_total = len(backbone.blocks)
    split_idx = max(1, n_blocks_total // 2)
    encoding_blocks = nn.ModuleList(backbone.blocks[:split_idx])
    columnwise_blocks = nn.ModuleList(backbone.blocks[split_idx:])
    
    # 輸出層
    d_out = 1 if task_type == 'regression' else n_classes
    output_layer = LinearEnsemble(d_block, d_out, k=k).to(device)
    
    # 根據 gnn_stage 創建 GNN
    gnn = None
    if gnn_stage in ['encoding', 'columnwise']:
        gnn_hidden = config.get('gnn_hidden', 64)
        gnn = SimpleGCN(d_block, gnn_hidden, d_block).to(device)
    
    # 定義前向傳播
    def forward(x_num, include_gnn=True):
        # Feature embedding
        if num_embeddings is not None:
            x = num_embeddings(x_num)
            x = x.flatten(1, -1)
        else:
            x = x_num
        
        # EnsembleView
        x = ensemble_view(x)
        
        # Encoding blocks
        for block in encoding_blocks:
            x = block(x)
        
        # GNN after encoding
        if gnn_stage == 'encoding' and gnn is not None and include_gnn:
            batch_size = x.shape[0]
            k_gnn = min(5, batch_size - 1)
            if k_gnn > 0 and batch_size > 1:
                x_mean = x.mean(dim=1)
                edge_index = knn_graph(x_mean, k_gnn).to(device)
                x_gnn = gnn(x_mean, edge_index)
                x = x_gnn.unsqueeze(1).expand(-1, k, -1)
        
        # Columnwise blocks
        for block in columnwise_blocks:
            x = block(x)
        
        # GNN after columnwise
        if gnn_stage == 'columnwise' and gnn is not None and include_gnn:
            batch_size = x.shape[0]
            k_gnn = min(5, batch_size - 1)
            if k_gnn > 0 and batch_size > 1:
                x_mean = x.mean(dim=1)
                edge_index = knn_graph(x_mean, k_gnn).to(device)
                x_gnn = gnn(x_mean, edge_index)
                x = x_gnn.unsqueeze(1).expand(-1, k, -1)
        
        return x
    
    # 如果是 decoding 階段，先訓練 encoder+columnwise，然後用 GNN 作為 decoder
    if gnn_stage == 'decoding':
        return tabm_decoding_with_gnn(
            X_train, y_train, X_val, y_val, X_test, y_test,
            num_embeddings, ensemble_view, encoding_blocks, columnwise_blocks,
            forward, task_type, material_outputs['label_stats'], k, d_block, config
        )
    
    # 正常訓練（encoding 或 columnwise 階段）
    # 收集所有參數
    all_params = []
    if num_embeddings is not None:
        all_params += list(num_embeddings.parameters())
    all_params += list(encoding_blocks.parameters())
    all_params += list(columnwise_blocks.parameters())
    all_params += list(output_layer.parameters())
    if gnn is not None:
        all_params += list(gnn.parameters())
    
    lr = config.get('lr', 0.002)
    weight_decay = config.get('weight_decay', 3e-4)
    optimizer = torch.optim.AdamW(all_params, lr=lr, weight_decay=weight_decay)
    
    # TabM 的損失函數
    def loss_fn(y_pred, y_true):
        y_pred_flat = y_pred.flatten(0, 1)
        y_true_repeated = y_true.repeat_interleave(k)
        
        if task_type == 'regression':
            return F.mse_loss(y_pred_flat.squeeze(-1), y_true_repeated)
        else:
            return F.cross_entropy(y_pred_flat, y_true_repeated)
    
    # 訓練循環
    batch_size = config.get('batch_size', 256)
    epochs = config.get('epochs', 200)
    patience = config.get('patience', 10)
    
    best_val_metric = float('-inf') if task_type != 'regression' else float('inf')
    best_test_metric = 0
    early_stop_counter = 0
    early_stop_epochs = 0
    
    for epoch in range(1, epochs + 1):
        # 訓練
        if num_embeddings is not None:
            num_embeddings.train()
        for block in encoding_blocks:
            block.train()
        for block in columnwise_blocks:
            block.train()
        output_layer.train()
        if gnn is not None:
            gnn.train()
        
        indices = torch.randperm(len(X_train), device=device)
        epoch_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            x_encoded = forward(x_batch, include_gnn=True)
            y_pred = output_layer(x_encoded)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        # 驗證
        with torch.no_grad():
            if num_embeddings is not None:
                num_embeddings.eval()
            for block in encoding_blocks:
                block.eval()
            for block in columnwise_blocks:
                block.eval()
            output_layer.eval()
            if gnn is not None:
                gnn.eval()
            
            x_val_encoded = forward(X_val, include_gnn=True)
            val_pred = output_layer(x_val_encoded)
            
            if task_type == 'regression':
                val_pred_mean = val_pred.mean(dim=1).squeeze(-1)
                label_stats = material_outputs['label_stats']
                val_pred_denorm = val_pred_mean * label_stats['std'] + label_stats['mean']
                val_metric = torch.sqrt(F.mse_loss(val_pred_denorm, y_val)).item()
            else:
                val_pred_prob = F.softmax(val_pred, dim=-1)
                val_pred_mean = val_pred_prob.mean(dim=1)
                val_pred_class = val_pred_mean.argmax(dim=-1)
                val_metric = (val_pred_class == y_val).float().mean().item()
        
        improved = (val_metric > best_val_metric) if task_type != 'regression' else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            early_stop_counter = 0
            
            # 測試
            with torch.no_grad():
                x_test_encoded = forward(X_test, include_gnn=True)
                test_pred = output_layer(x_test_encoded)
                if task_type == 'regression':
                    test_pred_mean = test_pred.mean(dim=1).squeeze(-1)
                    test_pred_denorm = test_pred_mean * label_stats['std'] + label_stats['mean']
                    best_test_metric = torch.sqrt(F.mse_loss(test_pred_denorm, y_test)).item()
                else:
                    test_pred_prob = F.softmax(test_pred, dim=-1)
                    test_pred_mean = test_pred_prob.mean(dim=1)
                    test_pred_class = test_pred_mean.argmax(dim=-1)
                    best_test_metric = (test_pred_class == y_test).float().mean().item()
        else:
            early_stop_counter += 1
        
        if epoch % 10 == 0:
            metric_name = 'RMSE' if task_type == 'regression' else 'Acc'
            print(f'Epoch {epoch}/{epochs}, Loss: {epoch_loss/n_batches:.4f}, Val {metric_name}: {val_metric:.4f}')
        
        if early_stop_counter >= patience:
            early_stop_epochs = epoch
            print(f"Early stopping at epoch {epoch}")
            break
    
    return {
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'early_stop_epochs': early_stop_epochs,
    }


def tabm_decoding_with_gnn(X_train, y_train, X_val, y_val, X_test, y_test,
                            num_embeddings, ensemble_view, encoding_blocks, columnwise_blocks,
                            forward, task_type, label_stats, k, d_block, config):
    """
    Decoding 階段：先訓練 encoder+columnwise，然後用 GNN 作為 decoder
    """
    print("TabM Decoding with GNN as decoder...")
    device = config['device']
    
    # 第一步：訓練 encoder + columnwise
    print("Step 1: Training encoder + columnwise...")
    all_params = []
    if num_embeddings is not None:
        all_params += list(num_embeddings.parameters())
    all_params += list(encoding_blocks.parameters())
    all_params += list(columnwise_blocks.parameters())
    
    # 臨時輸出層用於訓練 encoder
    n_classes = len(torch.unique(y_train)) if task_type != 'regression' else None
    d_out = 1 if task_type == 'regression' else n_classes
    temp_output = LinearEnsemble(d_block, d_out, k=k).to(device)
    all_params += list(temp_output.parameters())
    
    optimizer = torch.optim.AdamW(all_params, lr=config.get('lr', 0.002), weight_decay=config.get('weight_decay', 3e-4))
    
    def loss_fn(y_pred, y_true):
        y_pred_flat = y_pred.flatten(0, 1)
        y_true_repeated = y_true.repeat_interleave(k)
        if task_type == 'regression':
            return F.mse_loss(y_pred_flat.squeeze(-1), y_true_repeated)
        else:
            return F.cross_entropy(y_pred_flat, y_true_repeated)
    
    batch_size = config.get('batch_size', 256)
    encoder_epochs = config.get('epochs', 100)
    
    for epoch in range(1, encoder_epochs + 1):
        if num_embeddings is not None:
            num_embeddings.train()
        for block in encoding_blocks:
            block.train()
        for block in columnwise_blocks:
            block.train()
        temp_output.train()
        
        indices = torch.randperm(len(X_train), device=device)
        for i in range(0, len(X_train), batch_size):
            batch_indices = indices[i:i+batch_size]
            x_batch = X_train[batch_indices]
            y_batch = y_train[batch_indices]
            
            optimizer.zero_grad()
            x_encoded = forward(x_batch, include_gnn=False)
            y_pred = temp_output(x_encoded)
            loss = loss_fn(y_pred, y_batch)
            loss.backward()
            optimizer.step()
    
    print("Step 2: Extracting embeddings and training GNN decoder...")
    # 第二步：將所有數據通過 encoder+columnwise 獲取 embedding
    with torch.no_grad():
        if num_embeddings is not None:
            num_embeddings.eval()
        for block in encoding_blocks:
            block.eval()
        for block in columnwise_blocks:
            block.eval()
        
        X_train_emb = forward(X_train, include_gnn=False).mean(dim=1)  # 平均 k 個 ensemble
        X_val_emb = forward(X_val, include_gnn=False).mean(dim=1)
        X_test_emb = forward(X_test, include_gnn=False).mean(dim=1)
    
    # 合併所有 embedding
    X_all_emb = torch.cat([X_train_emb, X_val_emb, X_test_emb], dim=0)
    y_all = torch.cat([y_train, y_val, y_test], dim=0)
    
    # 建圖
    k_gnn = 5
    edge_index = knn_graph(X_all_emb, k_gnn).to(device)
    
    # 創建 GNN decoder
    gnn_hidden = config.get('gnn_hidden', 64)
    d_out = 1 if task_type == 'regression' else n_classes
    gnn_decoder = SimpleGCN(d_block, gnn_hidden, d_out).to(device)
    
    # 訓練 GNN
    optimizer_gnn = torch.optim.Adam(gnn_decoder.parameters(), lr=0.01)
    gnn_epochs = config.get('gnn_epochs', 200)
    patience = config.get('gnn_patience', 10)
    
    n_train = len(X_train)
    n_val = len(X_val)
    
    best_val_metric = float('-inf') if task_type != 'regression' else float('inf')
    best_test_metric = 0
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    
    for epoch in range(gnn_epochs):
        gnn_decoder.train()
        optimizer_gnn.zero_grad()
        
        out = gnn_decoder(X_all_emb, edge_index)
        
        if task_type == 'regression':
            loss = F.mse_loss(out[:n_train].squeeze(-1), y_train)
        else:
            loss = F.cross_entropy(out[:n_train], y_train)
        
        loss.backward()
        optimizer_gnn.step()
        
        # 驗證
        gnn_decoder.eval()
        with torch.no_grad():
            out_eval = gnn_decoder(X_all_emb, edge_index)
            val_out = out_eval[n_train:n_train+n_val]
            
            if task_type == 'regression':
                val_pred_denorm = val_out.squeeze(-1) * label_stats['std'] + label_stats['mean']
                val_metric = torch.sqrt(F.mse_loss(val_pred_denorm, y_val)).item()
            else:
                val_pred_class = val_out.argmax(dim=-1)
                val_metric = (val_pred_class == y_val).float().mean().item()
        
        improved = (val_metric > best_val_metric) if task_type != 'regression' else (val_metric < best_val_metric)
        if improved:
            best_val_metric = val_metric
            early_stop_counter = 0
            
            # 測試
            test_out = out_eval[n_train+n_val:]
            if task_type == 'regression':
                test_pred_denorm = test_out.squeeze(-1) * label_stats['std'] + label_stats['mean']
                best_test_metric = torch.sqrt(F.mse_loss(test_pred_denorm, y_test)).item()
            else:
                test_pred_class = test_out.argmax(dim=-1)
                best_test_metric = (test_pred_class == y_test).float().mean().item()
        else:
            early_stop_counter += 1
        
        if (epoch+1) % 10 == 0:
            metric_name = 'RMSE' if task_type == 'regression' else 'Acc'
            print(f'GNN Epoch {epoch+1}/{gnn_epochs}, Val {metric_name}: {val_metric:.4f}')
        
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"GNN Early stopping at epoch {epoch+1}")
            break
    
    return {
        'best_val_metric': best_val_metric,
        'best_test_metric': best_test_metric,
        'early_stop_epochs': encoder_epochs,
        'gnn_early_stop_epochs': gnn_early_stop_epochs,
    }


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    主函數：按五階段執行 TabM
    """
    print("=" * 60)
    print(f"TabM - Five Stage Execution (gnn_stage: {gnn_stage})")
    print("=" * 60)
    
    task_type = dataset_results['info']['task_type']
    gnn_early_stop_epochs = 0
    
    try:
        # 階段 0: Start (dummy)
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        
        # GNN after start
        if gnn_stage == 'start':
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(
                train_df, val_df, test_df, config, task_type
            )
        
        # 階段 1: Materialize
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        
        # GNN after materialize
        if gnn_stage == 'materialize':
            X_train_gnn, X_val_gnn, X_test_gnn, gnn_early_stop_epochs = gnn_after_materialize_fn(
                material_outputs['X_train'], material_outputs['y_train'],
                material_outputs['X_val'], material_outputs['y_val'],
                material_outputs['X_test'], material_outputs['y_test'],
                config, task_type
            )
            material_outputs['X_train'] = X_train_gnn
            material_outputs['X_val'] = X_val_gnn
            material_outputs['X_test'] = X_test_gnn
        
        # 階段 2-4: 核心訓練（encoding, columnwise, decoding 都在這裡處理）
        results = tabm_core_fn(material_outputs, config, gnn_stage)
        
        # 如果 results 中已經有 gnn_early_stop_epochs（decoding 階段），使用它
        # 否則使用之前階段設置的值（start/materialize）或默認值 0
        if 'gnn_early_stop_epochs' not in results:
            results['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        
    except Exception as e:
        import traceback
        print(f"Error in TabM: {str(e)}")
        traceback.print_exc()
        
        is_classification = task_type != 'regression'
        results = {
            'best_val_metric': float('-inf') if is_classification else float('inf'),
            'best_test_metric': float('-inf') if is_classification else float('inf'),
            'error': str(e),
            'early_stop_epochs': 0,
            'gnn_early_stop_epochs': 0,
        }
    
    return results


# 測試命令
#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models tabm --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models tabm --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models tabm --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models tabm --gnn_stages all --epochs 2
#  python main.py --dataset helena --models tabm --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models tabm --gnn_stages all --epochs 2
