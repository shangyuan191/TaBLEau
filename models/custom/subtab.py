"""
SubTab模型包裝器 - 參考excelformer.py的架構設計
支援GNN插入於start, materialize, encoding, columnwise, decoding階段
"""

import os
import gc
import copy
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader as TorchDataLoader
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score, mean_squared_error

# 內嵌必要元件，避免依賴外部 SubTab 專案
def set_seed(config):
    seed = int(config.get('seed', 42))
    import random
    import numpy as _np
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)
    _np.random.seed(seed)


class HiddenLayers(nn.Module):
    def __init__(self, dims, use_bn=False, use_dropout=False, dropout_rate=0.2):
        super().__init__()
        layers = []
        for i in range(1, len(dims) - 1):
            layers.append(nn.Linear(dims[i - 1], dims[i]))
            if use_bn:
                layers.append(nn.BatchNorm1d(dims[i]))
            layers.append(nn.LeakyReLU(inplace=False))
            if use_dropout:
                layers.append(nn.Dropout(dropout_rate))
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)


class AEWrapper(nn.Module):
    """簡化版 SubTab AE + Projection 包裝器，支援可變 dims。"""
    def __init__(self, options):
        super().__init__()
        self.options = copy.deepcopy(options)
        dims = list(self.options["dims"])  # [in, h1, ..., latent]
        use_bn = self.options.get("isBatchNorm", False)
        use_dropout = self.options.get("isDropout", False)
        dropout_rate = self.options.get("dropout_rate", 0.2)
        # Encoder hidden
        if len(dims) >= 3:
            self.encoder_hidden = HiddenLayers(dims[:-1], use_bn, use_dropout, dropout_rate)
        else:
            self.encoder_hidden = nn.Identity()
        # Latent
        self.latent = nn.Linear(dims[-2], dims[-1]) if len(dims) >= 2 else nn.Identity()
        # Projection head
        self.proj1 = nn.Linear(dims[-1], dims[-1])
        self.proj2 = nn.Linear(dims[-1], dims[-1])
        self.normalize = self.options.get("normalize", True)
        self.p_norm = self.options.get("p_norm", 2)
        # Decoder (mirror)
        dec_dims = list(reversed(dims))
        if len(dec_dims) >= 3:
            self.decoder_hidden = HiddenLayers(dec_dims[:-1], use_bn, use_dropout, dropout_rate)
        else:
            self.decoder_hidden = nn.Identity()
        self.decoder_out = nn.Linear(dec_dims[-2], dec_dims[-1]) if len(dec_dims) >= 2 else nn.Identity()
    
    def forward(self, x):
        h = self.encoder_hidden(x)
        latent = self.latent(h)
        z = F.leaky_relu(self.proj1(latent))
        z = self.proj2(z)
        if self.normalize:
            z = F.normalize(z, p=self.p_norm, dim=1)
        d = self.decoder_hidden(latent)
        x_recon = self.decoder_out(d)
        return z, latent, x_recon
    
    def decode(self, latent):
        """從 latent 向量解碼回重構數據"""
        d = self.decoder_hidden(latent)
        x_recon = self.decoder_out(d)
        return x_recon


class JointLoss(nn.Module):
    """簡化版 JointLoss：目前僅採用重構 MSE，保留擴充空間。"""
    def __init__(self, options):
        super().__init__()
        self.options = options
        self.device = options.get('device', torch.device('cpu'))
    def forward(self, z, x_recon, x_orig):
        recon_loss = F.mse_loss(x_recon, x_orig)
        closs = torch.tensor(0.0, device=self.device)
        zrecon_loss = torch.tensor(0.0, device=self.device)
        return recon_loss, closs, recon_loss, zrecon_loss


# ==================== GNN模組 ====================
class SimpleGCN(torch.nn.Module):
    """簡單的GCN模型，用於各階段插入"""
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        from torch_geometric.nn import GCNConv
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x


def knn_graph(x, k):
    """構建KNN圖"""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_index = []
    N = x_np.shape[0]
    for i in range(N):
        for j in indices[i][1:]:
            edge_index.append([i, j])
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


# ==================== 自定義Dataset ====================
class TabularDataset(Dataset):
    """表格數據集包裝器"""
    def __init__(self, df, target_col='target'):
        self.df = df.reset_index(drop=True)
        self.target_col = target_col
        
        # 分離特徵和標籤
        if target_col in df.columns:
            self.X = df.drop(columns=[target_col]).values.astype(np.float32)
            self.y = df[target_col].values
        else:
            self.X = df.values.astype(np.float32)
            self.y = np.zeros(len(df))
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ==================== GNN插入階段函數 ====================
def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    """在start和materialize之間插入GNN"""
    print("Executing GNN between start_fn and materialize_fn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    
    # 合併三個df
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values

    # 處理標籤
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
    
    # 設置mask
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    
    patience = config.get('gnn_patience', 10)
    best_loss = float('inf')
    early_stop_counter = 0
    
    # 訓練GNN
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
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
    
    # 將embedding分回三個df
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


def gnn_after_materialize_fn(train_loader, val_loader, test_loader, config, task_type):
    """在materialize和encoding之間插入GNN (在DataLoader創建後)"""
    print("Executing GNN after materialize_fn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    
    # 收集所有數據
    def collect_data(loader):
        X_list, y_list = [], []
        for x_batch, y_batch in loader:
            X_list.append(x_batch)
            y_list.append(y_batch)
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)
    
    train_X, train_y = collect_data(train_loader)
    val_X, val_y = collect_data(val_loader)
    test_X, test_y = collect_data(test_loader)
    
    # 合併所有數據
    all_X = torch.cat([train_X, val_X, test_X], dim=0).to(device)
    all_y = torch.cat([train_y, val_y, test_y], dim=0).to(device)
    
    # 建圖
    edge_index = knn_graph(all_X, k).to(device)
    in_dim = all_X.shape[1]
    out_dim = in_dim
    
    n_train = len(train_X)
    n_val = len(val_X)
    n_test = len(test_X)
    
    patience = config.get('gnn_patience', 10)
    best_loss = float('inf')
    early_stop_counter = 0
    
    # 訓練GNN
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    gnn_early_stop_epochs = 0
    
    gnn.train()
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(all_X, edge_index)
        loss = F.mse_loss(out, all_X)
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
        final_emb = gnn(all_X, edge_index).cpu().numpy()
    
    # 分割回三個集合並創建新的DataLoader
    train_emb = final_emb[:n_train]
    val_emb = final_emb[n_train:n_train+n_val]
    test_emb = final_emb[n_train+n_val:]
    
    train_y_np = train_y.cpu().numpy()
    val_y_np = val_y.cpu().numpy()
    test_y_np = test_y.cpu().numpy()
    
    # 創建新的DataFrame
    emb_cols = [f'N_feature_{i}' for i in range(1, out_dim+1)]
    train_df = pd.DataFrame(train_emb, columns=emb_cols)
    train_df['target'] = train_y_np
    val_df = pd.DataFrame(val_emb, columns=emb_cols)
    val_df['target'] = val_y_np
    test_df = pd.DataFrame(test_emb, columns=emb_cols)
    test_df['target'] = test_y_np
    
    # 創建新的DataLoader
    batch_size = config.get('batch_size', 32)
    train_dataset = TabularDataset(train_df)
    val_dataset = TabularDataset(val_df)
    test_dataset = TabularDataset(test_df)
    
    new_train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    new_val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    new_test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return new_train_loader, new_val_loader, new_test_loader, gnn_early_stop_epochs


def gnn_decoding_eval(train_loader, val_loader, test_loader, config, task_type, 
                      z_train, z_val, z_test, y_train, y_val, y_test):
    """在decoding階段使用GNN作為下游分類器/回歸器"""
    print("Executing GNN at decoding stage")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    
    # 合併所有嵌入
    all_z = np.concatenate([z_train, z_val, z_test], axis=0)
    all_y = np.concatenate([y_train, y_val, y_test], axis=0)
    
    x = torch.tensor(all_z, dtype=torch.float32, device=device)
    
    # 處理標籤
    if task_type == 'binclass':
        y = torch.tensor(all_y, dtype=torch.float32, device=device)
        num_classes = 1
    elif task_type == 'multiclass':
        y = torch.tensor(all_y, dtype=torch.long, device=device)
        num_classes = len(np.unique(all_y))
    else:  # regression
        y = torch.tensor(all_y, dtype=torch.float32, device=device)
        num_classes = 1
    
    # 建圖
    edge_index = knn_graph(x, k).to(device)
    
    # 設置mask
    n_train = len(z_train)
    n_val = len(z_val)
    n_test = len(z_test)
    N = n_train + n_val + n_test
    
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True
    
    in_dim = x.shape[1]
    out_dim = 1 if (task_type == 'regression' or task_type == 'binclass') else num_classes
    
    # 訓練GNN分類器
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    
    patience = config.get('gnn_patience', 10)
    best_val_metric = -float('inf') if task_type in ['binclass', 'multiclass'] else float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_test_metric = None
    
    gnn.train()
    for epoch in tqdm(range(gnn_epochs), desc="GNN Training"):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        
        # 計算損失
        if task_type == 'binclass':
            loss = F.binary_cross_entropy_with_logits(out[train_mask][:, 0], y[train_mask])
        elif task_type == 'multiclass':
            loss = F.cross_entropy(out[train_mask], y[train_mask])
        else:  # regression
            loss = F.mse_loss(out[train_mask][:, 0], y[train_mask])
        
        loss.backward()
        optimizer.step()
        
        # 驗證
        gnn.eval()
        with torch.no_grad():
            out_val = gnn(x, edge_index)
            
            if task_type == 'binclass':
                val_pred = torch.sigmoid(out_val[val_mask][:, 0])
                val_metric = roc_auc_score(y[val_mask].cpu().numpy(), val_pred.cpu().numpy())
                improved = val_metric > best_val_metric
            elif task_type == 'multiclass':
                val_pred = out_val[val_mask].argmax(dim=-1)
                val_metric = accuracy_score(y[val_mask].cpu().numpy(), val_pred.cpu().numpy())
                improved = val_metric > best_val_metric
            else:  # regression
                val_pred = out_val[val_mask][:, 0]
                val_metric = np.sqrt(mean_squared_error(y[val_mask].cpu().numpy(), val_pred.cpu().numpy()))
                improved = val_metric < best_val_metric
            
            if improved:
                best_val_metric = val_metric
                early_stop_counter = 0
                
                # 計算測試集指標
                if task_type == 'binclass':
                    test_pred = torch.sigmoid(out_val[test_mask][:, 0])
                    best_test_metric = roc_auc_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
                elif task_type == 'multiclass':
                    test_pred = out_val[test_mask].argmax(dim=-1)
                    best_test_metric = accuracy_score(y[test_mask].cpu().numpy(), test_pred.cpu().numpy())
                else:
                    test_pred = out_val[test_mask][:, 0]
                    best_test_metric = np.sqrt(mean_squared_error(y[test_mask].cpu().numpy(), test_pred.cpu().numpy()))
            else:
                early_stop_counter += 1
        
        gnn.train()
        
        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"GNN Early stopping at epoch {epoch+1}")
            break
    
    return best_val_metric, best_test_metric, gnn_early_stop_epochs


# ==================== SubTab核心訓練函數 ====================
def subtab_core_fn(train_loader, val_loader, test_loader, config, task_type, gnn_stage=None):
    """
    SubTab核心訓練函數
    整合encoding, columnwise (subsetting+contrastive), decoding階段
    """
    print("Executing subtab_core_fn")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 設置隨機種子
    set_seed(config)
    
    # 獲取輸入維度
    sample_batch, _ = next(iter(train_loader))
    input_dim = sample_batch.shape[1]
    
    # 計算子集大小（SubTab使用子集大小作為encoder的輸入維度）
    n_subsets = config.get('n_subsets', 4)
    overlap = config.get('overlap', 0.75)
    subset_size = int(input_dim / n_subsets)
    n_overlap = int(overlap * subset_size)
    # 計算實際子集大小（包含重疊）
    actual_subset_size = subset_size + 2 * n_overlap
    # 確保不超過總特徵數
    actual_subset_size = min(actual_subset_size, input_dim)
    
    # 更新配置中的維度（使用子集大小而不是完整輸入維度）
    if 'dims' not in config or len(config['dims']) == 0:
        # 默認架構: [subset_size, hidden1, hidden2, latent]
        # encoder_hidden: HiddenLayers([subset_size, hidden1, hidden2])會創建subset_size→hidden1→hidden2
        # latent: Linear(hidden2, latent)會創建hidden2→latent
        config['dims'] = [actual_subset_size, 128, 128, 64]
    else:
        config['dims'][0] = actual_subset_size
    
    config['device'] = device
    
    # 創建SubTab模型的配置
    subtab_config = {
        'dims': config['dims'],
        'device': device,
        'batch_size': config.get('batch_size', 32),
        'learning_rate': config.get('lr', 0.001),
        'epochs': config.get('epochs', 200),
        'scheduler': config.get('scheduler', False),
        'validate': config.get('validate', True),
        'nth_epoch': config.get('nth_epoch', 1),
        'seed': config.get('seed', 42),
        'shallow_architecture': config.get('shallow_architecture', True),
        'normalize': config.get('normalize', True),
        'p_norm': config.get('p_norm', 2),
        'n_subsets': config.get('n_subsets', 4),
        'overlap': config.get('overlap', 0.75),
        'add_noise': config.get('add_noise', True),
        'noise_type': config.get('noise_type', 'swap_noise'),
        'masking_ratio': config.get('masking_ratio', 0.3),
        'noise_level': config.get('noise_level', 0.1),
        'contrastive_loss': config.get('contrastive_loss', True),
        'distance_loss': config.get('distance_loss', True),
        'reconstruction': config.get('reconstruction', True),
        'reconstruct_subset': config.get('reconstruct_subset', False),
        'tau': config.get('tau', 0.1),
        'cosine_similarity': config.get('cosine_similarity', False),
        'aggregation': config.get('aggregation', 'mean'),
        'task_type': task_type,
        'isBatchNorm': config.get('isBatchNorm', False),
        'isDropout': config.get('isDropout', False),
        'dropout_rate': config.get('dropout_rate', 0.2),
    }
    
    # 創建模型和優化器
    encoder = AEWrapper(subtab_config).to(device)
    joint_loss = JointLoss(subtab_config)
    optimizer = torch.optim.Adam(encoder.parameters(), lr=subtab_config['learning_rate'])
    
    # 用於encoding/columnwise階段的GNN（如果需要）
    gnn_encoding = None
    gnn_columnwise = None
    gnn_early_stop_epochs = 0
    if gnn_stage == 'encoding':
        latent_dim = subtab_config['dims'][-1]
        gnn_encoding = SimpleGCN(latent_dim, config.get('gnn_hidden', 64), latent_dim).to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(gnn_encoding.parameters()),
            lr=subtab_config['learning_rate']
        )
    elif gnn_stage == 'columnwise':
        latent_dim = subtab_config['dims'][-1]
        gnn_columnwise = SimpleGCN(latent_dim, config.get('gnn_hidden', 64), latent_dim).to(device)
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(gnn_columnwise.parameters()),
            lr=subtab_config['learning_rate']
        )
    
    # 訓練循環
    patience = config.get('patience', 10)
    best_val_metric = float('inf')  # SubTab使用重構損失
    early_stop_counter = 0
    early_stop_epochs = 0
    
    train_losses = []
    val_losses = []
    
    epochs = config.get('epochs', 200)
    
    # SubTab的subset生成器
    def subset_generator(x_batch):
        """生成特徵子集 - 確保每個子集大小一致"""
        n_subsets = subtab_config['n_subsets']
        overlap = subtab_config['overlap']
        n_features = x_batch.shape[1]
        
        subset_size = int(n_features / n_subsets)
        n_overlap = int(overlap * subset_size)
        
        subsets = []
        for i in range(n_subsets):
            start_idx = max(0, i * subset_size - n_overlap)
            end_idx = min(n_features, (i + 1) * subset_size + n_overlap)
            subset = x_batch[:, start_idx:end_idx]
            
            # 確保所有子集大小一致（padding或truncate）
            current_size = subset.shape[1]
            if current_size < actual_subset_size:
                # Padding with zeros
                padding = torch.zeros(subset.shape[0], actual_subset_size - current_size, device=subset.device)
                subset = torch.cat([subset, padding], dim=1)
            elif current_size > actual_subset_size:
                # Truncate
                subset = subset[:, :actual_subset_size]
            
            # 添加噪聲（如果啟用）
            if subtab_config['add_noise']:
                noise = torch.randn_like(subset) * subtab_config['noise_level']
                subset = subset + noise
            
            subsets.append(subset)
        
        return subsets
    
    # 訓練階段用 val_loss 做 early stop（越小越好）
    best_val_loss = float('inf')
    early_stop_counter = 0
    early_stop_epochs = 0
    gnn_early_stop_epochs = 0
    for epoch in range(1, epochs + 1):
        encoder.train()
        if gnn_stage == 'encoding' and gnn_encoding is not None:
            gnn_encoding.train()
        epoch_loss = 0
        train_tqdm = tqdm(train_loader, desc=f'Epoch {epoch}/{epochs}')
        for x_batch, _ in train_tqdm:
            x_batch = x_batch.to(device)
            subsets = subset_generator(x_batch)
            latent_list = []
            for subset in subsets:
                _, latent, _ = encoder(subset)
                latent_list.append(latent)
            latent_all = torch.cat(latent_list, dim=0)
            if gnn_stage == 'encoding' and gnn_encoding is not None:
                edge_index = knn_graph(latent_all, k=min(5, latent_all.shape[0]-1)).to(device)
                latent_gnn = gnn_encoding(latent_all, edge_index)
                x_recon_all = encoder.decode(latent_gnn)
            else:
                x_recon_all = encoder.decode(latent_all)
            Xorig_all = torch.cat([s for s in subsets], dim=0)
            Xinput_all = torch.cat([s for s in subsets], dim=0)
            z_pair, latent_pair, x_recon_pair = encoder(Xinput_all)
            if gnn_stage == 'encoding' and gnn_encoding is not None:
                loss, _, _, _ = joint_loss(z_pair, x_recon_all, Xorig_all)
            else:
                loss, _, _, _ = joint_loss(z_pair, x_recon_pair, Xorig_all)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            train_tqdm.set_postfix({'loss': loss.item()})
        avg_train_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        # 驗證
        if config.get('validate', True) and epoch % config.get('nth_epoch', 1) == 0:
            encoder.eval()
            if gnn_stage == 'encoding' and gnn_encoding is not None:
                gnn_encoding.eval()
            val_loss = 0
            with torch.no_grad():
                for x_batch, _ in val_loader:
                    x_batch = x_batch.to(device)
                    subsets = subset_generator(x_batch)
                    latent_list = []
                    for subset in subsets:
                        _, latent, _ = encoder(subset)
                        latent_list.append(latent)
                    latent_all = torch.cat(latent_list, dim=0)
                    if gnn_stage == 'encoding' and gnn_encoding is not None:
                        edge_index = knn_graph(latent_all, k=min(5, latent_all.shape[0]-1)).to(device)
                        latent_gnn = gnn_encoding(latent_all, edge_index)
                        x_recon_all = encoder.decode(latent_gnn)
                    else:
                        x_recon_all = encoder.decode(latent_all)
                    Xorig_all = torch.cat([s for s in subsets], dim=0)
                    Xinput_all = torch.cat([s for s in subsets], dim=0)
                    z_pair, latent_pair, x_recon_pair = encoder(Xinput_all)
                    if gnn_stage == 'encoding' and gnn_encoding is not None:
                        loss, _, _, _ = joint_loss(z_pair, x_recon_all, Xorig_all)
                    else:
                        loss, _, _, _ = joint_loss(z_pair, x_recon_pair, Xorig_all)
                    val_loss += loss.item()
            avg_val_loss = val_loss / len(val_loader)
            val_losses.append(avg_val_loss)
            print(f'Epoch {epoch}: Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}')
            # Early stopping（用 val_loss 判斷，越小越好，加入容忍度避免浮點數精度問題）
            tolerance = 1e-8
            if avg_val_loss < best_val_loss - tolerance:
                best_val_loss = avg_val_loss
                early_stop_counter = 0
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience:
                early_stop_epochs = epoch
                print(f"Early stopping at epoch {epoch}")
                break
    
    # 提取嵌入用於評估
    def extract_embeddings(loader):
        encoder.eval()
        z_list, y_list = [], []
        
        with torch.no_grad():
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(device)
                subsets = subset_generator(x_batch)
                
                # 對所有子集提取latent並聚合
                latent_list = []
                for subset in subsets:
                    _, latent, _ = encoder(subset)
                    latent_list.append(latent)
                
                # 聚合方式：平均
                latent_aggregated = torch.stack(latent_list, dim=0).mean(dim=0)
                
                z_list.append(latent_aggregated.cpu().numpy())
                y_list.append(y_batch.numpy())
        
        return np.concatenate(z_list, axis=0), np.concatenate(y_list, axis=0)
    
    z_train, y_train = extract_embeddings(train_loader)
    z_val, y_val = extract_embeddings(val_loader)
    z_test, y_test = extract_embeddings(test_loader)
    
    return {
        'encoder': encoder,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'best_val_loss': best_val_loss,
        'z_train': z_train,
        'z_val': z_val,
        'z_test': z_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
        'early_stop_epochs': early_stop_epochs,
        'gnn_early_stop_epochs': gnn_early_stop_epochs,
    }


# ==================== 主函數 (對應excelformer.py的main) ====================
def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    """
    SubTab主函數 - 五階段執行
    對應excelformer.py的main函數簽名
    """
    print("SubTab - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    
    task_type = dataset_results['info']['task_type']
    
    try:
        # 設定 patience 預設值
        if 'patience' not in config:
            config['patience'] = 10

        gnn_early_stop_epochs = 0
        early_stop_epochs = 0

        # ===== 階段1: start =====
        if gnn_stage == 'start':
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(
                train_df, val_df, test_df, config, task_type
            )

        # ===== 階段2: materialize - 創建DataLoader =====
        batch_size = config.get('batch_size', 32)
        train_dataset = TabularDataset(train_df)
        val_dataset = TabularDataset(val_df)
        test_dataset = TabularDataset(test_df)

        train_loader = TorchDataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = TorchDataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        test_loader = TorchDataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        if gnn_stage == 'materialize':
            train_loader, val_loader, test_loader, gnn_early_stop_epochs = gnn_after_materialize_fn(
                train_loader, val_loader, test_loader, config, task_type
            )

        # ===== 階段3-5: encoding + columnwise + decoding (SubTab核心) =====
        core_results = subtab_core_fn(
            train_loader, val_loader, test_loader, config, task_type, gnn_stage=gnn_stage
        )
        early_stop_epochs = core_results.get('early_stop_epochs', 0)

        # ===== 階段5: decoding - 下游評估 =====
        if gnn_stage == 'decoding':
            best_val_metric, best_test_metric, gnn_epochs = gnn_decoding_eval(
                train_loader, val_loader, test_loader, config, task_type,
                core_results['z_train'], core_results['z_val'], core_results['z_test'],
                core_results['y_train'], core_results['y_val'], core_results['y_test']
            )
            gnn_early_stop_epochs = gnn_epochs
        else:
            z_train = core_results['z_train']
            z_val = core_results['z_val']
            z_test = core_results['z_test']
            y_train = core_results['y_train']
            y_val = core_results['y_val']
            y_test = core_results['y_test']

            if task_type == 'regression':
                reg = LinearRegression()
                reg.fit(z_train, y_train)
                y_val_pred = reg.predict(z_val)
                y_test_pred = reg.predict(z_test)
                best_val_metric = np.sqrt(mean_squared_error(y_val, y_val_pred))
                best_test_metric = np.sqrt(mean_squared_error(y_test, y_test_pred))
            elif task_type == 'binclass':
                clf = LogisticRegression(max_iter=1200, C=1.0)
                clf.fit(z_train, y_train)
                if hasattr(clf, 'predict_proba'):
                    y_val_pred_proba = clf.predict_proba(z_val)[:, 1]
                    y_test_pred_proba = clf.predict_proba(z_test)[:, 1]
                    best_val_metric = roc_auc_score(y_val, y_val_pred_proba)
                    best_test_metric = roc_auc_score(y_test, y_test_pred_proba)
                else:
                    y_val_pred = clf.predict(z_val)
                    y_test_pred = clf.predict(z_test)
                    best_val_metric = accuracy_score(y_val, y_val_pred)
                    best_test_metric = accuracy_score(y_test, y_test_pred)
            else:  # multiclass
                clf = LogisticRegression(max_iter=1200, C=1.0)
                clf.fit(z_train, y_train)
                y_val_pred = clf.predict(z_val)
                y_test_pred = clf.predict(z_test)
                best_val_metric = accuracy_score(y_val, y_val_pred)
                best_test_metric = accuracy_score(y_test, y_test_pred)

        # encoding/columnwise 階段 GNN 只 forward，不訓練，gnn_early_stop_epochs 必須為 0
        if gnn_stage in ['encoding', 'columnwise']:
            gnn_early_stop_epochs = 0

        # 構建返回結果
        results = {
            'train_losses': core_results['train_losses'],
            'val_losses': core_results['val_losses'],
            'best_val_metric': best_val_metric,
            'best_test_metric': best_test_metric,
            'early_stop_epochs': early_stop_epochs,
            'gnn_early_stop_epochs': gnn_early_stop_epochs,
        }
        
    except Exception as e:
        print(f"Error in SubTab training: {str(e)}")
        import traceback
        traceback.print_exc()
        
        results = {
            'train_losses': [],
            'val_losses': [],
            'best_val_metric': float('inf'),
            'best_test_metric': float('inf'),
            'error': str(e),
            'early_stop_epochs': 0,
            'gnn_early_stop_epochs': 0,
        }
    
    return results


# ==================== 便利包裝函數 ====================
def start_fn(train_df, val_df, test_df):
    """Start階段（預處理前）"""
    return train_df, val_df, test_df

def materialize_fn(train_df, val_df, test_df, dataset_results=None, config=None):
    """Materialize 階段：將 DataFrame 打包為簡單的 DataLoader（僅為相容掛鉤）。"""
    batch_size =  config.get('batch_size', 32) if isinstance(config, dict) else 32
    train_loader = TorchDataLoader(TabularDataset(train_df), batch_size=batch_size, shuffle=True)
    val_loader = TorchDataLoader(TabularDataset(val_df), batch_size=batch_size, shuffle=False)
    test_loader = TorchDataLoader(TabularDataset(test_df), batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

def encoding_fn(*args, **kwargs):
    """Encoding 階段（相容用，這裡用不上）。"""
    return args[0] if args else None

def columnwise_fn(*args, **kwargs):
    """Column-wise 階段（相容用，這裡用不上）。"""
    return args[0] if args else None

def decoding_fn(*args, **kwargs):
    """Decoding 階段（相容用，這裡用不上）。"""
    return args[0] if args else None


#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models subtab --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models subtab --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models subtab --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models subtab --gnn_stages all --epochs 2
#  python main.py --dataset helena --models subtab --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models subtab --gnn_stages all --epochs 2