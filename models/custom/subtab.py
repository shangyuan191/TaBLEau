"""
SubTab模型包裝器 - 參考excelformer.py的架構設計
支援GNN插入於start, materialize, encoding, columnwise, decoding階段
"""

import os
import gc
import copy
import time
import math
import sys
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


# ==================== ExcelFormer-style 共用工具 ====================

# DGM 動態建圖（與 excelformer.py 對齊；嚴格依賴，不提供 fallback）
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
from DGMlib.layers import DGM_d


def resolve_device(config: dict) -> torch.device:
    """統一的裝置選擇：優先使用 config['gpu'] 指定的 cuda:id。"""
    gpu_id = None
    try:
        gpu_id = config.get('gpu', None)
    except Exception:
        gpu_id = None
    if torch.cuda.is_available():
        if gpu_id is not None:
            try:
                device = torch.device(f'cuda:{int(gpu_id)}')
                print(f"[DEVICE][SUBTAB] resolve_device: Using cuda:{int(gpu_id)}")
                return device
            except Exception:
                device = torch.device('cuda')
                print(f"[DEVICE][SUBTAB] resolve_device: Using cuda (fallback from gpu={gpu_id})")
                return device
        print('[DEVICE][SUBTAB] resolve_device: Using default cuda')
        return torch.device('cuda')
    print('[DEVICE][SUBTAB] resolve_device: Using cpu')
    return torch.device('cpu')


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    """沿指定維度做 z-score 標準化。"""
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """對 edge_index 做對稱化並加入自迴路，移除重複邊。"""
    if edge_index.numel() == 0:
        loops = torch.arange(num_nodes, device=edge_index.device)
        return torch.stack([loops, loops], dim=0)

    device = edge_index.device
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    loops = torch.arange(num_nodes, device=device)
    self_edges = torch.stack([loops, loops], dim=0)
    ei = torch.cat([edge_index, rev, self_edges], dim=1)
    edge_ids = ei[0] * num_nodes + ei[1]
    unique_ids = torch.unique(edge_ids, sorted=False)
    return torch.stack([unique_ids // num_nodes, unique_ids % num_nodes], dim=0)


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
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers: int = 2):
        super().__init__()
        from torch_geometric.nn import GCNConv
        dims = [in_dim] + [hidden_dim] * (num_layers - 1) + [out_dim]
        self.layers = torch.nn.ModuleList()
        for i in range(len(dims) - 1):
            self.layers.append(GCNConv(dims[i], dims[i + 1]))

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = torch.relu(x)
        return x


def knn_graph(x, k):
    """構建KNN圖"""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    N = x_np.shape[0]
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    actual_k = min(int(k), N - 1)
    nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_index = []
    for i in range(N):
        for j in indices[i][1:]:
            edge_index.append([i, j])
            edge_index.append([j, i])
    if not edge_index:
        return torch.empty((2, 0), dtype=torch.long)
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index


def _run_attention_dgm_pipeline(train_df: pd.DataFrame,
                               val_df: pd.DataFrame,
                               test_df: pd.DataFrame,
                               config: dict,
                               task_type: str,
                               stage_tag: str = 'START'):
    """ExcelFormer-style: Self-Attn pooling -> DGM dynamic graph -> GCN -> Self-Attn decode.

    - Train uses train split only; early stopping uses val_loss; inference is inductive per split.
    - Output is reconstructed feature matrix with original columns.
    """

    device = resolve_device(config)
    tag = f"[SUBTAB][{stage_tag}-GNN-DGM]"
    print(f"{tag} Running attention + DGM pipeline (inductive)")

    feature_cols = [c for c in train_df.columns if c != 'target']
    num_cols = len(feature_cols)
    if num_cols <= 0:
        return train_df, val_df, test_df, 0

    # Fixed k for fairness (config override allowed), with safety cap per split.
    dgm_k = int(config.get('dgm_k', 10))
    dgm_distance = config.get('dgm_distance', 'euclidean')

    # Strictly align with excelformer.py: use config['epochs'] for GNN-stage training epochs.
    gnn_epochs = int(config.get('epochs', 200))
    patience = int(config.get('gnn_patience', 10))
    loss_threshold = float(config.get('gnn_loss_threshold', 1e-4))
    attn_dim = int(config.get('gnn_attn_dim', config.get('gnn_hidden', 64)))
    gnn_hidden = int(config.get('gnn_hidden', 64))
    gnn_out_dim = int(config.get('gnn_out_dim', attn_dim))
    attn_heads = int(config.get('gnn_num_heads', config.get('gnn_attn_heads', 4)))
    lr = float(config.get('gnn_lr', 1e-3))

    x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)

    y_train_np = train_df['target'].values
    y_val_np = val_df['target'].values
    y_test_np = test_df['target'].values

    # Determine out_dim for supervised head (align with excelformer/tabm)
    if task_type in ['binclass', 'multiclass']:
        y_all_np = np.concatenate([y_train_np, y_val_np, y_test_np])
        num_classes = len(pd.unique(y_all_np))
        if task_type == 'binclass' and num_classes != 2:
            num_classes = 2
        out_dim = int(num_classes)
    else:
        out_dim = 1

    n_train = x_train.shape[0]
    if n_train <= 1:
        raise ValueError(f"{tag} Need at least 2 train samples for DGM (got n_train={n_train})")
    dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))

    # Modules
    try:
        attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    except Exception as e:
        print(f"{tag} Attention init failed ({e}); fallback attn_dim=64, heads=4")
        attn_dim = 64
        attn_heads = 4
        attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)

    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    gnn = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=2).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    pred_head = torch.nn.Linear(gnn_out_dim, out_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)
    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    class DGMEmbedWrapper(torch.nn.Module):
        def forward(self, x, A=None):
            return x

    dgm_embed_f = DGMEmbedWrapper()
    dgm_module = DGM_d(dgm_embed_f, k=dgm_k_train, distance=dgm_distance).to(device)

    params = list(attn_in.parameters()) + list(attn_out.parameters()) + list(input_proj.parameters()) \
        + list(gnn.parameters()) + list(gcn_to_attn.parameters()) + list(pred_head.parameters()) \
        + list(out_proj.parameters()) + [column_embed, pool_query]
    params += list(dgm_module.parameters())

    optimizer = torch.optim.Adam(params, lr=lr)

    def forward_pass(x_tensor: torch.Tensor):
        Ns = x_tensor.shape[0]
        x_in = input_proj(x_tensor.unsqueeze(-1))  # [Ns, num_cols, attn_dim]
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)  # [Ns, attn_dim]
        row_emb_std = _standardize(row_emb, dim=0)

        row_emb_batched = row_emb_std.unsqueeze(0)
        row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module(row_emb_batched, A=None)
        row_emb_dgm = row_emb_dgm.squeeze(0)
        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)

        gcn_out = gnn(row_emb_dgm, edge_index_dgm)  # [Ns, gnn_out_dim]
        logits = pred_head(gcn_out)

        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)  # [Ns,1,attn_dim]
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)  # [Ns, num_cols]
        return logits, recon, logprobs_dgm

    best_val_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    best_states = None

    for epoch in range(gnn_epochs):
        # train
        attn_in.train(); attn_out.train(); input_proj.train(); gnn.train(); gcn_to_attn.train(); pred_head.train(); out_proj.train()
        dgm_module.train()

        optimizer.zero_grad()
        logits_train, _, logprobs_dgm = forward_pass(x_train)
        if task_type in ['binclass', 'multiclass']:
            y_train = torch.tensor(y_train_np, dtype=torch.long, device=device)
            train_loss = F.cross_entropy(logits_train, y_train)
        else:
            y_train = torch.tensor(y_train_np, dtype=torch.float32, device=device)
            train_loss = F.mse_loss(logits_train.squeeze(), y_train)
        dgm_reg = -logprobs_dgm.mean() * 0.01
        train_loss = train_loss + dgm_reg
        train_loss.backward()
        optimizer.step()

        # val
        attn_in.eval(); attn_out.eval(); input_proj.eval(); gnn.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
        dgm_module.eval()
        with torch.no_grad():
            logits_val, _, _ = forward_pass(x_val)
            if task_type in ['binclass', 'multiclass']:
                y_val = torch.tensor(y_val_np, dtype=torch.long, device=device)
                val_loss = F.cross_entropy(logits_val, y_val)
            else:
                y_val = torch.tensor(y_val_np, dtype=torch.float32, device=device)
                val_loss = F.mse_loss(logits_val.squeeze(), y_val)

        val_loss_val = float(val_loss.item())
        improved = val_loss_val < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss_val
            early_stop_counter = 0
            best_states = {
                'attn_in': attn_in.state_dict(),
                'attn_out': attn_out.state_dict(),
                'input_proj': input_proj.state_dict(),
                'gnn': gnn.state_dict(),
                'gcn_to_attn': gcn_to_attn.state_dict(),
                'pred_head': pred_head.state_dict(),
                'out_proj': out_proj.state_dict(),
                'column_embed': column_embed.detach().clone(),
                'pool_query': pool_query.detach().clone(),
            }
            best_states['dgm_module'] = dgm_module.state_dict()
        else:
            early_stop_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"{tag} Epoch {epoch+1}/{gnn_epochs} train_loss={train_loss.item():.4f} val_loss={val_loss_val:.4f}")

        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"{tag} Early stopping at epoch {gnn_early_stop_epochs}")
            break

    # restore
    if best_states is not None:
        attn_in.load_state_dict(best_states['attn_in'])
        attn_out.load_state_dict(best_states['attn_out'])
        input_proj.load_state_dict(best_states['input_proj'])
        gnn.load_state_dict(best_states['gnn'])
        gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
        pred_head.load_state_dict(best_states['pred_head'])
        out_proj.load_state_dict(best_states['out_proj'])
        with torch.no_grad():
            column_embed.copy_(best_states['column_embed'])
            pool_query.copy_(best_states['pool_query'])
        dgm_module.load_state_dict(best_states['dgm_module'])

    attn_in.eval(); attn_out.eval(); input_proj.eval(); gnn.eval(); gcn_to_attn.eval(); pred_head.eval(); out_proj.eval()
    dgm_module.eval()

    with torch.no_grad():
        _, recon_train, _ = forward_pass(x_train)
        _, recon_val, _ = forward_pass(x_val)
        _, recon_test, _ = forward_pass(x_test)

    train_df_gnn = pd.DataFrame(recon_train.cpu().numpy(), columns=feature_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(recon_val.cpu().numpy(), columns=feature_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(recon_test.cpu().numpy(), columns=feature_cols, index=test_df.index)
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values

    return train_df_gnn, val_df_gnn, test_df_gnn, int(gnn_early_stop_epochs)


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
    """在 start 與 materialize 之間套用 ExcelFormer-style 注意力 + DGM 管線（inductive）。"""
    return _run_attention_dgm_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        task_type=task_type,
        stage_tag='START',
    )


def gnn_after_materialize_fn(train_loader, val_loader, test_loader, config, task_type):
    """在 materialize 後套用 ExcelFormer-style 注意力 + DGM 管線（inductive）。"""
    # 收集所有數據（保持 split 分離）
    def collect_data(loader):
        X_list, y_list = [], []
        for x_batch, y_batch in loader:
            X_list.append(x_batch)
            y_list.append(y_batch)
        return torch.cat(X_list, dim=0), torch.cat(y_list, dim=0)

    train_X, train_y = collect_data(train_loader)
    val_X, val_y = collect_data(val_loader)
    test_X, test_y = collect_data(test_loader)

    # 轉成 DataFrame 以便重用 pipeline（欄位名使用 N_feature_i，與 wrapper 內一致）
    in_dim = int(train_X.shape[1])
    feature_cols = [f'N_feature_{i}' for i in range(1, in_dim + 1)]

    train_df = pd.DataFrame(train_X.cpu().numpy(), columns=feature_cols)
    val_df = pd.DataFrame(val_X.cpu().numpy(), columns=feature_cols)
    test_df = pd.DataFrame(test_X.cpu().numpy(), columns=feature_cols)
    train_df['target'] = train_y.cpu().numpy()
    val_df['target'] = val_y.cpu().numpy()
    test_df['target'] = test_y.cpu().numpy()

    train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs = _run_attention_dgm_pipeline(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        config=config,
        task_type=task_type,
        stage_tag='MATERIALIZE',
    )

    # 重新包回 DataLoader
    batch_size = config.get('batch_size', 32)
    new_train_loader = TorchDataLoader(TabularDataset(train_df_gnn), batch_size=batch_size, shuffle=True)
    new_val_loader = TorchDataLoader(TabularDataset(val_df_gnn), batch_size=batch_size, shuffle=False)
    new_test_loader = TorchDataLoader(TabularDataset(test_df_gnn), batch_size=batch_size, shuffle=False)

    return new_train_loader, new_val_loader, new_test_loader, int(gnn_early_stop_epochs)





# ==================== SubTab核心訓練函數 ====================
def subtab_core_fn(train_loader, val_loader, test_loader, config, task_type, gnn_stage=None):
    """
    SubTab核心訓練函數
    整合encoding, columnwise (subsetting+contrastive), decoding階段
    """
    print("Executing subtab_core_fn")
    device = resolve_device(config)
    
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

    # decoding: strict alignment with excelformer_core_fn(decoding)
    # - GNN replaces downstream decoder/head
    # - Jointly trained with encoder in the same loop
    # - Minibatch Self-Attn -> pooling -> DGM -> GCN(out_dim)
    use_decoding_gnn = gnn_stage == 'decoding'
    decoding_modules = None
    if use_decoding_gnn:
        latent_dim = int(subtab_config['dims'][-1])
        gnn_hidden = int(config.get('gnn_hidden', 64))
        attn_heads = int(config.get('gnn_num_heads', config.get('gnn_attn_heads', 4)))
        dgm_k = int(config.get('dgm_k', config.get('gnn_knn', 5)))
        dgm_distance = config.get('dgm_distance', 'euclidean')

        # Determine out_dim from labels (train split)
        if task_type in ['binclass', 'multiclass']:
            y_all = []
            for _, yb in train_loader:
                y_all.append(yb.detach().cpu().view(-1))
            y_all = torch.cat(y_all, dim=0) if y_all else torch.empty(0, dtype=torch.long)
            num_classes = int(torch.unique(y_all.long()).numel()) if y_all.numel() > 0 else 0
            if task_type == 'binclass' and num_classes != 2:
                num_classes = 2
            out_dim = int(max(2, num_classes)) if task_type == 'binclass' else int(max(1, num_classes))
        else:
            out_dim = 1

        self_attn = torch.nn.MultiheadAttention(embed_dim=latent_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(latent_dim).to(device)
        subset_embed = torch.nn.Parameter(torch.randn(int(n_subsets), latent_dim, device=device))
        pool_query = torch.nn.Parameter(torch.randn(latent_dim, device=device))

        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x

        dgm_embed_f = DGMEmbedWrapper()
        dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
        gnn_pred = SimpleGCN(latent_dim, gnn_hidden, out_dim, num_layers=2).to(device)

        decoding_modules = {
            'out_dim': out_dim,
            'self_attn': self_attn,
            'attn_norm': attn_norm,
            'subset_embed': subset_embed,
            'pool_query': pool_query,
            'dgm_module': dgm_module,
            'gnn_pred': gnn_pred,
        }

        optimizer = torch.optim.Adam(
            list(encoder.parameters())
            + list(self_attn.parameters())
            + list(attn_norm.parameters())
            + [subset_embed, pool_query]
            + list(dgm_module.parameters())
            + list(gnn_pred.parameters()),
            lr=subtab_config['learning_rate']
        )
    
    # encoding/columnwise: strict alignment with excelformer_core_fn -> mini-batch Self-Attn -> pooling -> DGM -> GCN -> inject back.
    gnn_early_stop_epochs = 0  # encoding/columnwise 階段不做獨立 GNN 早停統計
    use_latent_gnn = gnn_stage in ['encoding', 'columnwise']
    latent_gnn_modules = None
    if use_latent_gnn:
        latent_dim = int(subtab_config['dims'][-1])
        gnn_hidden = int(config.get('gnn_hidden', 64))
        attn_heads = int(config.get('gnn_num_heads', config.get('gnn_attn_heads', 4)))
        gnn_dropout = float(config.get('gnn_dropout', 0.1))
        dgm_k = int(config.get('dgm_k', config.get('gnn_knn', 5)))
        dgm_distance = config.get('dgm_distance', 'euclidean')

        self_attn = torch.nn.MultiheadAttention(embed_dim=latent_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_norm = torch.nn.LayerNorm(latent_dim).to(device)
        self_attn_out = torch.nn.MultiheadAttention(embed_dim=latent_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_out_norm = torch.nn.LayerNorm(latent_dim).to(device)
        subset_embed = torch.nn.Parameter(torch.randn(int(n_subsets), latent_dim, device=device))
        pool_query = torch.nn.Parameter(torch.randn(latent_dim, device=device))

        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x

        dgm_embed_f = DGMEmbedWrapper()
        dgm_module = DGM_d(dgm_embed_f, k=dgm_k, distance=dgm_distance).to(device)
        gnn = SimpleGCN(latent_dim, gnn_hidden, latent_dim, num_layers=2).to(device)
        gcn_to_attn = torch.nn.Linear(latent_dim, latent_dim).to(device)
        ffn_pre = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(gnn_dropout),
            torch.nn.Linear(latent_dim * 2, latent_dim),
        ).to(device)
        ffn_post = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, latent_dim * 2),
            torch.nn.GELU(),
            torch.nn.Dropout(gnn_dropout),
            torch.nn.Linear(latent_dim * 2, latent_dim),
        ).to(device)
        fusion_alpha_param = torch.nn.Parameter(torch.tensor(-0.847, device=device))

        latent_gnn_modules = {
            'self_attn': self_attn,
            'attn_norm': attn_norm,
            'self_attn_out': self_attn_out,
            'attn_out_norm': attn_out_norm,
            'subset_embed': subset_embed,
            'pool_query': pool_query,
            'dgm_module': dgm_module,
            'gnn': gnn,
            'gcn_to_attn': gcn_to_attn,
            'ffn_pre': ffn_pre,
            'ffn_post': ffn_post,
            'fusion_alpha_param': fusion_alpha_param,
        }

        optimizer = torch.optim.Adam(
            list(encoder.parameters())
            + list(self_attn.parameters())
            + list(attn_norm.parameters())
            + list(self_attn_out.parameters())
            + list(attn_out_norm.parameters())
            + [subset_embed, pool_query, fusion_alpha_param]
            + list(dgm_module.parameters())
            + list(gnn.parameters())
            + list(gcn_to_attn.parameters())
            + list(ffn_pre.parameters())
            + list(ffn_post.parameters()),
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
    
    # 訓練階段用 val_loss 做 early stop（越小越好）並恢復最佳權重（ExcelFormer-style）
    best_val_loss = float('inf')
    best_states = None
    early_stop_counter = 0
    early_stop_epochs = 0

    # decoding stage: end-to-end supervised training with GNN as decoder/head
    if use_decoding_gnn and decoding_modules is not None:
        tag = '[SUBTAB][DECODING-JOINT]'
        patience = int(config.get('patience', 10))
        loss_threshold = float(config.get('loss_threshold', 1e-4))
        best_val_metric = None
        best_test_metric = None

        def forward_decode(latent_tokens: torch.Tensor) -> torch.Tensor:
            # latent_tokens: [B, S, D]
            B = int(latent_tokens.shape[0])
            S = int(latent_tokens.shape[1])
            D = int(latent_tokens.shape[2])
            if B <= 1:
                raise ValueError(f"{tag} Need batch size >= 2 for DGM (got B={B})")
            tokens = latent_tokens + decoding_modules['subset_embed'].unsqueeze(0)
            tokens_norm = decoding_modules['attn_norm'](tokens)
            tokens_attn, _ = decoding_modules['self_attn'](tokens_norm, tokens_norm, tokens_norm)
            tokens_attn = tokens + tokens_attn

            pool_logits = (tokens_attn * decoding_modules['pool_query']).sum(dim=-1) / math.sqrt(D)
            pool_weights = torch.softmax(pool_logits, dim=1)
            x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)  # [B,D]

            x_std = _standardize(x_pooled, dim=0)
            x_batched = x_std.unsqueeze(0)  # [1,B,D]
            if hasattr(decoding_modules['dgm_module'], 'k'):
                decoding_modules['dgm_module'].k = int(min(int(decoding_modules['dgm_module'].k), max(1, B - 1)))
            x_dgm, edge_index_dgm, _ = decoding_modules['dgm_module'](x_batched, A=None)
            x_dgm = x_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, B)
            logits = decoding_modules['gnn_pred'](x_dgm, edge_index_dgm)
            return logits

        def eval_split(loader) -> tuple[float, float, np.ndarray, np.ndarray]:
            # returns (avg_loss, metric, y_true_np, y_pred_np/prob)
            losses = []
            y_true = []
            y_out = []
            encoder.eval(); decoding_modules['self_attn'].eval(); decoding_modules['attn_norm'].eval()
            decoding_modules['dgm_module'].eval(); decoding_modules['gnn_pred'].eval()
            with torch.no_grad():
                for xb, yb in loader:
                    xb = xb.to(device)
                    yb = yb.to(device)
                    subsets = subset_generator(xb)
                    latent_list = []
                    for subset in subsets:
                        _, latent, _ = encoder(subset)
                        latent_list.append(latent)
                    latent_tokens = torch.stack(latent_list, dim=1)
                    B = int(latent_tokens.shape[0])
                    if B <= 1:
                        continue
                    logits = forward_decode(latent_tokens)

                    if task_type in ['binclass', 'multiclass']:
                        yb_long = yb.long()
                        loss = F.cross_entropy(logits, yb_long)
                        y_true.append(yb_long.detach().cpu().numpy())
                        y_out.append(logits.detach().cpu().numpy())
                    else:
                        yb_f = yb.float()
                        loss = F.mse_loss(logits.squeeze(), yb_f)
                        y_true.append(yb_f.detach().cpu().numpy())
                        y_out.append(logits.squeeze().detach().cpu().numpy())
                    losses.append(float(loss.item()) * int(B))

            if not losses:
                return float('inf'), float('nan'), np.array([]), np.array([])
            avg_loss = float(np.sum(losses) / max(1, sum([len(a) for a in y_true])))

            y_true_np = np.concatenate(y_true, axis=0) if y_true else np.array([])
            y_out_np = np.concatenate(y_out, axis=0) if y_out else np.array([])
            if task_type == 'binclass' and y_out_np.size:
                prob = torch.softmax(torch.tensor(y_out_np), dim=-1).numpy()[:, 1]
                metric_val = float(roc_auc_score(y_true_np, prob))
            elif task_type == 'multiclass' and y_out_np.size:
                pred = np.argmax(y_out_np, axis=-1)
                metric_val = float(accuracy_score(y_true_np, pred))
            elif task_type == 'regression' and y_out_np.size:
                metric_val = float(np.sqrt(mean_squared_error(y_true_np, y_out_np)))
            else:
                metric_val = float('nan')
            return avg_loss, metric_val, y_true_np, y_out_np

        for epoch in range(1, epochs + 1):
            encoder.train(); decoding_modules['self_attn'].train(); decoding_modules['attn_norm'].train()
            decoding_modules['dgm_module'].train(); decoding_modules['gnn_pred'].train()

            train_loss_sum = 0.0
            train_count = 0
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                subsets = subset_generator(xb)
                latent_list = []
                for subset in subsets:
                    _, latent, _ = encoder(subset)
                    latent_list.append(latent)
                latent_tokens = torch.stack(latent_list, dim=1)
                B = int(latent_tokens.shape[0])
                if B <= 1:
                    continue
                logits = forward_decode(latent_tokens)
                if task_type in ['binclass', 'multiclass']:
                    loss = F.cross_entropy(logits, yb.long())
                else:
                    loss = F.mse_loss(logits.squeeze(), yb.float())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss_sum += float(loss.item()) * B
                train_count += B

            avg_train_loss = float(train_loss_sum / max(1, train_count))
            train_losses.append(avg_train_loss)

            # validate
            val_loss, val_metric, _, _ = eval_split(val_loader)
            val_losses.append(float(val_loss))

            improved = float(val_loss) < best_val_loss - loss_threshold
            if improved:
                best_val_loss = float(val_loss)
                best_val_metric = float(val_metric)
                early_stop_counter = 0
                best_states = {
                    'encoder': copy.deepcopy(encoder.state_dict()),
                    'self_attn': copy.deepcopy(decoding_modules['self_attn'].state_dict()),
                    'attn_norm': copy.deepcopy(decoding_modules['attn_norm'].state_dict()),
                    'dgm_module': copy.deepcopy(decoding_modules['dgm_module'].state_dict()),
                    'gnn_pred': copy.deepcopy(decoding_modules['gnn_pred'].state_dict()),
                    'subset_embed': decoding_modules['subset_embed'].detach().clone(),
                    'pool_query': decoding_modules['pool_query'].detach().clone(),
                }
                # evaluate test metric under current best
                _, test_metric, _, _ = eval_split(test_loader)
                best_test_metric = float(test_metric)
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                early_stop_epochs = epoch
                print(f"{tag} Early stopping at epoch {epoch}")
                break

        # restore best
        if best_states is not None:
            encoder.load_state_dict(best_states['encoder'])
            decoding_modules['self_attn'].load_state_dict(best_states['self_attn'])
            decoding_modules['attn_norm'].load_state_dict(best_states['attn_norm'])
            decoding_modules['dgm_module'].load_state_dict(best_states['dgm_module'])
            decoding_modules['gnn_pred'].load_state_dict(best_states['gnn_pred'])
            with torch.no_grad():
                decoding_modules['subset_embed'].copy_(best_states['subset_embed'])
                decoding_modules['pool_query'].copy_(best_states['pool_query'])

        # extract embeddings for consistency (used by reporting/baselines in other stages)
        def extract_embeddings(loader):
            encoder.eval()
            z_list, y_list = [], []
            with torch.no_grad():
                for x_batch, y_batch in loader:
                    x_batch = x_batch.to(device)
                    subsets = subset_generator(x_batch)
                    latent_list = []
                    for subset in subsets:
                        _, latent, _ = encoder(subset)
                        latent_list.append(latent)
                    latent_aggregated = torch.stack(latent_list, dim=0).mean(dim=0)
                    z_list.append(latent_aggregated.cpu().numpy())
                    y_list.append(y_batch.numpy())
            return np.concatenate(z_list, axis=0), np.concatenate(y_list, axis=0)

        z_train, y_train_arr = extract_embeddings(train_loader)
        z_val, y_val_arr = extract_embeddings(val_loader)
        z_test, y_test_arr = extract_embeddings(test_loader)

        return {
            'encoder': encoder,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'best_val_loss': best_val_loss,
            'z_train': z_train,
            'z_val': z_val,
            'z_test': z_test,
            'y_train': y_train_arr,
            'y_val': y_val_arr,
            'y_test': y_test_arr,
            'early_stop_epochs': int(early_stop_epochs),
            'gnn_early_stop_epochs': 0,
            'best_val_metric': float(best_val_metric) if best_val_metric is not None else float('nan'),
            'best_test_metric': float(best_test_metric) if best_test_metric is not None else float('nan'),
        }

    for epoch in range(1, epochs + 1):
        encoder.train()
        if use_latent_gnn and latent_gnn_modules is not None:
            latent_gnn_modules['self_attn'].train()
            latent_gnn_modules['attn_norm'].train()
            latent_gnn_modules['self_attn_out'].train()
            latent_gnn_modules['attn_out_norm'].train()
            latent_gnn_modules['dgm_module'].train()
            latent_gnn_modules['gnn'].train()
            latent_gnn_modules['gcn_to_attn'].train()
            latent_gnn_modules['ffn_pre'].train()
            latent_gnn_modules['ffn_post'].train()
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
            if use_latent_gnn and latent_gnn_modules is not None:
                # latent_tokens: [B, n_subsets, latent_dim]
                latent_tokens = torch.stack(latent_list, dim=1)
                B = int(latent_tokens.shape[0])
                S = int(latent_tokens.shape[1])
                D = int(latent_tokens.shape[2])

                tokens = latent_tokens + latent_gnn_modules['subset_embed'].unsqueeze(0)  # [B,S,D]
                tokens_norm = latent_gnn_modules['attn_norm'](tokens)
                attn_out1, _ = latent_gnn_modules['self_attn'](tokens_norm, tokens_norm, tokens_norm)
                tokens_attn = tokens + attn_out1
                tokens_attn = tokens_attn + latent_gnn_modules['ffn_pre'](latent_gnn_modules['attn_norm'](tokens_attn))

                pool_logits = (tokens_attn * latent_gnn_modules['pool_query']).sum(dim=-1) / math.sqrt(D)
                pool_weights = torch.softmax(pool_logits, dim=1)
                x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)  # [B,D]

                x_pooled_std = _standardize(x_pooled, dim=0)
                x_batched = x_pooled_std.unsqueeze(0)  # [1,B,D]
                if hasattr(latent_gnn_modules['dgm_module'], 'k'):
                    latent_gnn_modules['dgm_module'].k = int(min(int(latent_gnn_modules['dgm_module'].k), max(1, B - 1)))
                x_dgm, edge_index_dgm, _ = latent_gnn_modules['dgm_module'](x_batched, A=None)
                x_dgm = x_dgm.squeeze(0)
                edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, B)
                x_gnn_out = latent_gnn_modules['gnn'](x_dgm, edge_index_dgm)  # [B,D]

                gcn_ctx = latent_gnn_modules['gcn_to_attn'](x_gnn_out).unsqueeze(1)  # [B,1,D]
                tokens_with_ctx = tokens_attn + gcn_ctx
                tokens_ctx_norm = latent_gnn_modules['attn_out_norm'](tokens_with_ctx)
                attn_out2, _ = latent_gnn_modules['self_attn_out'](tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
                tokens_mid = tokens_with_ctx + attn_out2
                tokens_out = tokens_mid + latent_gnn_modules['ffn_post'](latent_gnn_modules['attn_out_norm'](tokens_mid))

                fusion_alpha = torch.sigmoid(latent_gnn_modules['fusion_alpha_param'])
                latent_tokens = latent_tokens + fusion_alpha * tokens_out
                latent_all_updated = latent_tokens.reshape(B * S, D)
                x_recon_all = encoder.decode(latent_all_updated)
            else:
                x_recon_all = encoder.decode(latent_all)
            Xorig_all = torch.cat([s for s in subsets], dim=0)
            Xinput_all = torch.cat([s for s in subsets], dim=0)
            z_pair, latent_pair, x_recon_pair = encoder(Xinput_all)
            if use_latent_gnn and latent_gnn_modules is not None:
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
            if use_latent_gnn and latent_gnn_modules is not None:
                latent_gnn_modules['self_attn'].eval()
                latent_gnn_modules['attn_norm'].eval()
                latent_gnn_modules['self_attn_out'].eval()
                latent_gnn_modules['attn_out_norm'].eval()
                latent_gnn_modules['dgm_module'].eval()
                latent_gnn_modules['gnn'].eval()
                latent_gnn_modules['gcn_to_attn'].eval()
                latent_gnn_modules['ffn_pre'].eval()
                latent_gnn_modules['ffn_post'].eval()
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
                    if use_latent_gnn and latent_gnn_modules is not None:
                        latent_tokens = torch.stack(latent_list, dim=1)
                        B = int(latent_tokens.shape[0])
                        S = int(latent_tokens.shape[1])
                        D = int(latent_tokens.shape[2])

                        tokens = latent_tokens + latent_gnn_modules['subset_embed'].unsqueeze(0)
                        tokens_norm = latent_gnn_modules['attn_norm'](tokens)
                        attn_out1, _ = latent_gnn_modules['self_attn'](tokens_norm, tokens_norm, tokens_norm)
                        tokens_attn = tokens + attn_out1
                        tokens_attn = tokens_attn + latent_gnn_modules['ffn_pre'](latent_gnn_modules['attn_norm'](tokens_attn))

                        pool_logits = (tokens_attn * latent_gnn_modules['pool_query']).sum(dim=-1) / math.sqrt(D)
                        pool_weights = torch.softmax(pool_logits, dim=1)
                        x_pooled = (pool_weights.unsqueeze(-1) * tokens_attn).sum(dim=1)

                        x_pooled_std = _standardize(x_pooled, dim=0)
                        x_batched = x_pooled_std.unsqueeze(0)
                        if hasattr(latent_gnn_modules['dgm_module'], 'k'):
                            latent_gnn_modules['dgm_module'].k = int(min(int(latent_gnn_modules['dgm_module'].k), max(1, B - 1)))
                        x_dgm, edge_index_dgm, _ = latent_gnn_modules['dgm_module'](x_batched, A=None)
                        x_dgm = x_dgm.squeeze(0)
                        edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, B)
                        x_gnn_out = latent_gnn_modules['gnn'](x_dgm, edge_index_dgm)

                        gcn_ctx = latent_gnn_modules['gcn_to_attn'](x_gnn_out).unsqueeze(1)
                        tokens_with_ctx = tokens_attn + gcn_ctx
                        tokens_ctx_norm = latent_gnn_modules['attn_out_norm'](tokens_with_ctx)
                        attn_out2, _ = latent_gnn_modules['self_attn_out'](tokens_ctx_norm, tokens_ctx_norm, tokens_ctx_norm)
                        tokens_mid = tokens_with_ctx + attn_out2
                        tokens_out = tokens_mid + latent_gnn_modules['ffn_post'](latent_gnn_modules['attn_out_norm'](tokens_mid))

                        fusion_alpha = torch.sigmoid(latent_gnn_modules['fusion_alpha_param'])
                        latent_tokens = latent_tokens + fusion_alpha * tokens_out
                        latent_all_updated = latent_tokens.reshape(B * S, D)
                        x_recon_all = encoder.decode(latent_all_updated)
                    else:
                        x_recon_all = encoder.decode(latent_all)
                    Xorig_all = torch.cat([s for s in subsets], dim=0)
                    Xinput_all = torch.cat([s for s in subsets], dim=0)
                    z_pair, latent_pair, x_recon_pair = encoder(Xinput_all)
                    if use_latent_gnn and latent_gnn_modules is not None:
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
                best_states = {
                    'encoder': copy.deepcopy(encoder.state_dict()),
                }
                if use_latent_gnn and latent_gnn_modules is not None:
                    best_states['latent_gnn'] = {
                        'self_attn': copy.deepcopy(latent_gnn_modules['self_attn'].state_dict()),
                        'attn_norm': copy.deepcopy(latent_gnn_modules['attn_norm'].state_dict()),
                        'self_attn_out': copy.deepcopy(latent_gnn_modules['self_attn_out'].state_dict()),
                        'attn_out_norm': copy.deepcopy(latent_gnn_modules['attn_out_norm'].state_dict()),
                        'dgm_module': copy.deepcopy(latent_gnn_modules['dgm_module'].state_dict()),
                        'gnn': copy.deepcopy(latent_gnn_modules['gnn'].state_dict()),
                        'gcn_to_attn': copy.deepcopy(latent_gnn_modules['gcn_to_attn'].state_dict()),
                        'ffn_pre': copy.deepcopy(latent_gnn_modules['ffn_pre'].state_dict()),
                        'ffn_post': copy.deepcopy(latent_gnn_modules['ffn_post'].state_dict()),
                        'subset_embed': latent_gnn_modules['subset_embed'].detach().clone(),
                        'pool_query': latent_gnn_modules['pool_query'].detach().clone(),
                        'fusion_alpha_param': latent_gnn_modules['fusion_alpha_param'].detach().clone(),
                    }
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience:
                early_stop_epochs = epoch
                print(f"Early stopping at epoch {epoch}")
                break

    # 恢復最佳權重（若有）
    if best_states is not None:
        encoder.load_state_dict(best_states['encoder'])
        if use_latent_gnn and latent_gnn_modules is not None and 'latent_gnn' in best_states:
            lg = best_states['latent_gnn']
            latent_gnn_modules['self_attn'].load_state_dict(lg['self_attn'])
            latent_gnn_modules['attn_norm'].load_state_dict(lg['attn_norm'])
            latent_gnn_modules['self_attn_out'].load_state_dict(lg['self_attn_out'])
            latent_gnn_modules['attn_out_norm'].load_state_dict(lg['attn_out_norm'])
            latent_gnn_modules['dgm_module'].load_state_dict(lg['dgm_module'])
            latent_gnn_modules['gnn'].load_state_dict(lg['gnn'])
            latent_gnn_modules['gcn_to_attn'].load_state_dict(lg['gcn_to_attn'])
            latent_gnn_modules['ffn_pre'].load_state_dict(lg['ffn_pre'])
            latent_gnn_modules['ffn_post'].load_state_dict(lg['ffn_post'])
            with torch.no_grad():
                latent_gnn_modules['subset_embed'].copy_(lg['subset_embed'])
                latent_gnn_modules['pool_query'].copy_(lg['pool_query'])
                latent_gnn_modules['fusion_alpha_param'].copy_(lg['fusion_alpha_param'])
    
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
            # Strict alignment: decoding is trained end-to-end inside subtab_core_fn with GNN as decoder/head.
            best_val_metric = float(core_results.get('best_val_metric', float('nan')))
            best_test_metric = float(core_results.get('best_test_metric', float('nan')))
            gnn_early_stop_epochs = 0
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

        # encoding/columnwise 階段：GNN 可能在 core 內隨訓練更新，但此處不做「單獨 GNN 早停統計」，gnn_early_stop_epochs 維持 0
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