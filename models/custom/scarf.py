import math
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.custom.scarf_lib.dataset import SCARFDataset
from models.custom.scarf_lib.loss import NTXent
from models.custom.scarf_lib.model import SCARF
from models.custom.scarf_lib.utils import fix_seed

import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors

# DGM 動態圖模組（可選）
sys.path.insert(0, '/home/skyler/ModelComparison/DGM_pytorch')
try:
    from DGMlib.layers import DGM_d
    DGM_AVAILABLE = True
except ImportError:
    DGM_AVAILABLE = False
    print("[WARNING] DGM_d not available, some features will be disabled")


def resolve_device(config: dict) -> torch.device:
    """Unified device resolver with optional GPU index from config."""
    gpu_id = None
    try:
        gpu_id = config.get('gpu', None)
    except Exception:
        gpu_id = None
    if torch.cuda.is_available():
        if gpu_id is not None:
            try:
                device = torch.device(f'cuda:{int(gpu_id)}')
                print(f"[DEVICE] resolve_device: Using cuda:{int(gpu_id)}")
                return device
            except Exception:
                device = torch.device('cuda')
                print(f"[DEVICE] resolve_device: Using cuda (fallback from gpu={gpu_id})")
                return device
        device = torch.device('cuda')
        print("[DEVICE] resolve_device: Using default cuda")
        return device
    print("[DEVICE] resolve_device: Using cpu")
    return torch.device('cpu')


def compute_adaptive_dgm_k(num_samples, num_features, dataset_name: str = ''):
    """Heuristic for adaptive dgm_k; mirrors ExcelFormer alignment."""
    base_k = int(np.sqrt(max(num_samples, 1)))
    feature_factor = 1.0 + np.log1p(max(num_features, 1)) / 10
    adjusted_k = int(base_k * feature_factor)
    if num_samples < 500:
        density_factor = 1.3
    elif num_samples > 5000:
        density_factor = 0.9
    else:
        density_factor = 1.0
    adaptive_k = int(adjusted_k * density_factor)
    upper_limit = min(30, max(15, int(4 * np.sqrt(max(num_samples, 1)))))
    if num_samples < 1000:
        upper_limit = min(20, int(3 * np.sqrt(max(num_samples, 1))))
    adaptive_k = max(5, min(adaptive_k, upper_limit))
    print(f"[DGM-K] dataset={dataset_name or 'unknown'} N={num_samples} D={num_features} -> k={adaptive_k}")
    return adaptive_k


def _standardize(x: torch.Tensor, dim: int = 0, eps: float = 1e-6) -> torch.Tensor:
    mean = x.mean(dim=dim, keepdim=True)
    std = x.std(dim=dim, keepdim=True).clamp_min(eps)
    return (x - mean) / std


def _symmetrize_and_self_loop(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    rev = torch.stack([edge_index[1], edge_index[0]], dim=0)
    loops = torch.arange(num_nodes, device=edge_index.device)
    self_edges = torch.stack([loops, loops], dim=0)
    ei = torch.cat([edge_index, rev, self_edges], dim=1)
    edge_ids = ei[0] * num_nodes + ei[1]
    unique_ids = torch.unique(edge_ids, sorted=False)
    return torch.stack([unique_ids // num_nodes, unique_ids % num_nodes], dim=0)



class SimpleGCN(torch.nn.Module):
    """Multi-layer GCN used in graph-enhanced stages."""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers: int = 2):
        super().__init__()
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


def knn_graph(x, k, directed: bool = False):
    """k-NN graph with optional symmetric edges and auto-guarded k."""
    x_np = x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x
    N = x_np.shape[0]
    if N <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    actual_k = min(k, N - 1)
    nbrs = NearestNeighbors(n_neighbors=actual_k + 1, algorithm='auto').fit(x_np)
    _, indices = nbrs.kneighbors(x_np)
    edge_list = []
    for i in range(N):
        for j in indices[i][1:]:
            edge_list.append([i, j])
            if not directed:
                edge_list.append([j, i])
    if not edge_list:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edge_list, dtype=torch.long).t().contiguous()


def _run_attention_dgm_pipeline(train_df, val_df, test_df, config, dataset_name: str = '', task_type: str | None = None):
    """Self-attention + DGM + GCN autoencoder used for GNN stages."""
    print("[SCARF][GNN] Running attention + DGM pipeline (inductive)...")
    device = resolve_device(config)

    feature_cols = [c for c in train_df.columns if c != 'target']
    num_cols = len(feature_cols)

    # adaptive dgm_k unless explicitly set
    auto_k = compute_adaptive_dgm_k(len(train_df), num_cols, dataset_name)
    dgm_k = int(config.get('dgm_k', auto_k))
    dgm_distance = config.get('dgm_distance', 'euclidean')
    gnn_epochs = config.get('gnn_epochs', 200)
    patience = config.get('gnn_patience', 10)
    loss_threshold = config.get('gnn_loss_threshold', 1e-4)
    attn_dim = config.get('gnn_attn_dim', config.get('gnn_hidden', 64))
    gnn_hidden = config.get('gnn_hidden', 64)
    attn_heads = config.get('gnn_attn_heads', 4)
    lr = config.get('gnn_lr', 1e-3)

    # tensors
    x_train = torch.tensor(train_df[feature_cols].values, dtype=torch.float32, device=device)
    x_val = torch.tensor(val_df[feature_cols].values, dtype=torch.float32, device=device)
    x_test = torch.tensor(test_df[feature_cols].values, dtype=torch.float32, device=device)

    n_train = len(train_df)
    n_val = len(val_df)

    try:
        attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
        attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=attn_heads, batch_first=True).to(device)
    except Exception as e:
        print(f"[SCARF][GNN] Attention init failed ({e}); falling back to dim=64, heads=4")
        attn_dim = 64
        attn_in = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=4, batch_first=True).to(device)
        attn_out = torch.nn.MultiheadAttention(embed_dim=attn_dim, num_heads=4, batch_first=True).to(device)

    input_proj = torch.nn.Linear(1, attn_dim).to(device)
    gnn_out_dim = num_cols
    gnn_model = SimpleGCN(attn_dim, gnn_hidden, gnn_out_dim, num_layers=2).to(device)
    gcn_to_attn = torch.nn.Linear(gnn_out_dim, attn_dim).to(device)
    out_proj = torch.nn.Linear(attn_dim, 1).to(device)
    column_embed = torch.nn.Parameter(torch.randn(num_cols, attn_dim, device=device))
    pool_query = torch.nn.Parameter(torch.randn(attn_dim, device=device))

    # DGM module (optional)
    if DGM_AVAILABLE and n_train > 1:
        class DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x

        dgm_embed_f = DGMEmbedWrapper()
        dgm_k_train = int(min(dgm_k, max(1, n_train - 1)))
        dgm_module = DGM_d(dgm_embed_f, k=dgm_k_train, distance=dgm_distance).to(device)
    else:
        dgm_module = None

    def forward_pass(x_tensor, use_dgm: bool = True):
        Ns = x_tensor.shape[0]
        x_in = input_proj(x_tensor.unsqueeze(-1))  # [Ns, num_cols, attn_dim]
        tokens = x_in + column_embed.unsqueeze(0)
        tokens_attn, _ = attn_in(tokens, tokens, tokens)

        pool_logits = (tokens_attn * pool_query).sum(dim=2) / math.sqrt(attn_dim)
        pool_weights = torch.softmax(pool_logits, dim=1)
        row_emb = (pool_weights.unsqueeze(2) * tokens_attn).sum(dim=1)
        row_emb_std = _standardize(row_emb, dim=0)

        if use_dgm and dgm_module is not None:
            row_emb_batched = row_emb_std.unsqueeze(0)
            row_emb_dgm, edge_index_dgm, logprobs_dgm = dgm_module(row_emb_batched, A=None)
            row_emb_dgm = row_emb_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)
        else:
            row_emb_dgm = row_emb_std
            edge_index_dgm = knn_graph(row_emb_dgm, k=min(5, max(1, Ns - 1))).to(device)
            logprobs_dgm = torch.tensor(0.0, device=device)

        gcn_out = gnn_model(row_emb_dgm, edge_index_dgm)
        gcn_ctx = gcn_to_attn(gcn_out).unsqueeze(1)
        tokens_with_ctx = tokens_attn + gcn_ctx
        tokens_out, _ = attn_out(tokens_with_ctx, tokens_with_ctx, tokens_with_ctx)
        recon = out_proj(tokens_out).squeeze(-1)
        return recon, logprobs_dgm

    params = list(attn_in.parameters()) + list(attn_out.parameters()) + \
             list(input_proj.parameters()) + list(gnn_model.parameters()) + \
             list(gcn_to_attn.parameters()) + list(out_proj.parameters()) + \
             [column_embed, pool_query]
    if dgm_module is not None:
        params.extend(list(dgm_module.parameters()))

    optimizer = torch.optim.Adam(params, lr=lr)
    best_val_loss = float('inf')
    early_stop_counter = 0
    best_states = {}
    gnn_early_stop_epochs = 0

    for epoch in range(gnn_epochs):
        attn_in.train(); attn_out.train(); gnn_model.train()
        if dgm_module is not None:
            dgm_module.train()
        optimizer.zero_grad()
        recon_train, logprobs_dgm = forward_pass(x_train, use_dgm=True)
        recon_loss = F.mse_loss(recon_train, x_train)
        dgm_reg = -logprobs_dgm.mean() * 0.01 if isinstance(logprobs_dgm, torch.Tensor) else 0.0
        train_loss = recon_loss + dgm_reg
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        optimizer.step()

        attn_in.eval(); attn_out.eval(); gnn_model.eval()
        if dgm_module is not None:
            dgm_module.eval()
        with torch.no_grad():
            recon_val, _ = forward_pass(x_val, use_dgm=True)
            val_loss = F.mse_loss(recon_val, x_val)

        improved = val_loss.item() < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss.item()
            early_stop_counter = 0
            best_states = {
                'attn_in': attn_in.state_dict(),
                'attn_out': attn_out.state_dict(),
                'input_proj': input_proj.state_dict(),
                'gnn_model': gnn_model.state_dict(),
                'gcn_to_attn': gcn_to_attn.state_dict(),
                'out_proj': out_proj.state_dict(),
                'column_embed': column_embed.detach().clone(),
                'pool_query': pool_query.detach().clone(),
            }
            if dgm_module is not None:
                best_states['dgm_module'] = dgm_module.state_dict()
        else:
            early_stop_counter += 1

        if (epoch + 1) % 10 == 0:
            print(f"[SCARF][GNN] Epoch {epoch+1}/{gnn_epochs} train_loss={train_loss.item():.4f} val_loss={val_loss.item():.4f}")

        if early_stop_counter >= patience:
            gnn_early_stop_epochs = epoch + 1
            print(f"[SCARF][GNN] Early stopping at epoch {gnn_early_stop_epochs}")
            break

    if best_states:
        attn_in.load_state_dict(best_states['attn_in'])
        attn_out.load_state_dict(best_states['attn_out'])
        input_proj.load_state_dict(best_states['input_proj'])
        gnn_model.load_state_dict(best_states['gnn_model'])
        gcn_to_attn.load_state_dict(best_states['gcn_to_attn'])
        out_proj.load_state_dict(best_states['out_proj'])
        column_embed.data = best_states['column_embed']
        pool_query.data = best_states['pool_query']
        if dgm_module is not None and 'dgm_module' in best_states:
            dgm_module.load_state_dict(best_states['dgm_module'])

    attn_in.eval(); attn_out.eval(); gnn_model.eval()
    if dgm_module is not None:
        dgm_module.eval()

    with torch.no_grad():
        recon_train, _ = forward_pass(x_train, use_dgm=True)
        recon_val, _ = forward_pass(x_val, use_dgm=True)
        recon_test, _ = forward_pass(x_test, use_dgm=True)

    train_df_gnn = pd.DataFrame(recon_train.cpu().numpy(), columns=feature_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(recon_val.cpu().numpy(), columns=feature_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(recon_test.cpu().numpy(), columns=feature_cols, index=test_df.index)

    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values

    return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs

seed = 42
fix_seed(seed)
def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    dataset_name = config.get('dataset', '')
    return _run_attention_dgm_pipeline(train_df, val_df, test_df, config, dataset_name, task_type)



def gnn_after_materialize_fn(material_outputs, dataset_results, config, task_type):
    print("[SCARF][GNN] Materialize-stage attention + DGM pipeline")

    train_ds = material_outputs['train_ds']
    val_ds = material_outputs['val_ds']
    test_ds = material_outputs['test_ds']
    batch_size = material_outputs.get('batch_size', config.get('batch_size', 128))
    dataset_name = dataset_results.get('info', {}).get('dataset', '') if isinstance(dataset_results, dict) else ''

    def ds_to_df(ds: SCARFDataset):
        cols = ds.columns if ds.columns is not None else [f'feature_{i}' for i in range(ds.shape[1])]
        df = pd.DataFrame(ds.data, columns=cols)
        df['target'] = ds.target
        return df

    train_df = ds_to_df(train_ds)
    val_df = ds_to_df(val_ds)
    test_df = ds_to_df(test_ds)

    train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs = _run_attention_dgm_pipeline(
        train_df, val_df, test_df, config, dataset_name, task_type
    )

    feature_cols = [c for c in train_df_gnn.columns if c != 'target']
    train_ds_gnn = SCARFDataset(train_df_gnn[feature_cols].to_numpy(), train_df_gnn['target'].to_numpy(), columns=feature_cols)
    val_ds_gnn = SCARFDataset(val_df_gnn[feature_cols].to_numpy(), val_df_gnn['target'].to_numpy(), columns=feature_cols)
    test_ds_gnn = SCARFDataset(test_df_gnn[feature_cols].to_numpy(), test_df_gnn['target'].to_numpy(), columns=feature_cols)

    train_loader_gnn = DataLoader(train_ds_gnn, batch_size=batch_size, shuffle=True)
    val_loader_gnn = DataLoader(val_ds_gnn, batch_size=batch_size, shuffle=False)
    test_loader_gnn = DataLoader(test_ds_gnn, batch_size=batch_size, shuffle=False)

    material_outputs.update({
        'train_loader': train_loader_gnn,
        'val_loader': val_loader_gnn,
        'test_loader': test_loader_gnn,
        'train_ds': train_ds_gnn,
        'val_ds': val_ds_gnn,
        'test_ds': test_ds_gnn,
        'gnn_early_stop_epochs': gnn_early_stop_epochs,
    })
    return material_outputs
def start_fn(train_df, val_df, test_df):
    print("Starting Scarf model...")
    print(f"train_df: {train_df.shape}, val_df: {val_df.shape}, test_df: {test_df.shape}")
    return train_df, val_df, test_df

def materialize_fn(train_df, val_df, test_df, dataset_results, config):
    print("Materializing data...")
    print(f"train_df: {train_df.shape}, val_df: {val_df.shape}, test_df: {test_df.shape}")
    print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")

    batch_size = config.get('batch_size', 128)
    epochs = config.get('epochs', 200)
    device = resolve_device(config)

    # 預處理
    train_data = train_df.drop(columns=['target'])
    train_target = train_df['target']
    val_data = val_df.drop(columns=['target'])
    val_target = val_df['target']
    test_data = test_df.drop(columns=['target'])
    test_target = test_df['target']

    # 移除常數欄位
    constant_cols = [c for c in train_data.columns if train_data[c].nunique() == 1]
    train_data.drop(columns=constant_cols, inplace=True)
    val_data.drop(columns=constant_cols, inplace=True)
    test_data.drop(columns=constant_cols, inplace=True)

    # 標準化
    scaler = StandardScaler()
    train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    val_data = pd.DataFrame(scaler.transform(val_data), columns=val_data.columns)
    test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    # SCARFDataset
    train_ds = SCARFDataset(train_data.to_numpy(), train_target.to_numpy(), columns=train_data.columns)
    val_ds = SCARFDataset(val_data.to_numpy(), val_target.to_numpy(), columns=val_data.columns)
    test_ds = SCARFDataset(test_data.to_numpy(), test_target.to_numpy(), columns=test_data.columns)

    # DataLoader
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # 判斷任務型態
    task_type = dataset_results['info']['task_type'] if 'info' in dataset_results else 'binclass'
    is_classification = task_type in ['binclass', 'multiclass']
    out_channels = len(np.unique(train_target)) if is_classification else 1
    is_binary_class = is_classification and out_channels == 2

    # 過濾 low >= high 欄位
    features_low = train_ds.features_low
    features_high = train_ds.features_high
    valid_cols = features_low < features_high
    if not np.all(valid_cols):
        print(f"Filtering out {np.sum(~valid_cols)} columns where low >= high")
        keep_idx = np.where(valid_cols)[0]
        train_data = train_data.iloc[:, keep_idx]
        val_data = val_data.iloc[:, keep_idx]
        test_data = test_data.iloc[:, keep_idx]
        train_ds = SCARFDataset(train_data.to_numpy(), train_target.to_numpy(), columns=train_data.columns)
        val_ds = SCARFDataset(val_data.to_numpy(), val_target.to_numpy(), columns=val_data.columns)
        test_ds = SCARFDataset(test_data.to_numpy(), test_target.to_numpy(), columns=test_data.columns)
    return {
        'batch_size': batch_size,
        'epochs': epochs,
        'device': device,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'train_ds': train_ds,
        'val_ds': val_ds,
        'test_ds': test_ds,
        'task_type': task_type,
        'is_classification': is_classification,
        'is_binary_class': is_binary_class,
        'out_channels': out_channels,
    }
    

def scarf_core_fn(material_outputs, config, task_type, gnn_stage):

    print("[SCARF] Joint contrastive + supervised training")
    print(f"[SCARF] gnn_stage={gnn_stage}")

    epochs = int(material_outputs['epochs'])
    device = material_outputs['device']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    train_ds = material_outputs['train_ds']
    task_type = material_outputs['task_type']
    out_channels = int(material_outputs['out_channels'])

    # ---- hyperparams ----
    dim_hidden_encoder = int(config.get('dim_hidden_encoder', 8))
    num_hidden_encoder = int(config.get('num_hidden_encoder', 3))
    dim_hidden_head = int(config.get('dim_hidden_head', 24))
    num_hidden_head = int(config.get('num_hidden_head', 2))
    dropout = float(config.get('dropout', 0.1))

    gnn_hidden = int(config.get('gnn_hidden', 64))
    gnn_knn = int(config.get('gnn_knn', 5))
    dgm_k = int(config.get('dgm_k', 10))
    dgm_distance = config.get('dgm_distance', 'euclidean')
    use_dgm = bool(config.get('use_dgm', True))
    dgm_reg_weight = float(config.get('dgm_reg_weight', 0.01))

    lr = float(config.get('lr', 1e-3))
    weight_decay = float(config.get('weight_decay', 1e-5))
    sup_loss_weight = float(config.get('sup_loss_weight', 1.0))
    corruption_rate = float(config.get('corruption_rate', 0.6))
    patience = int(config.get('patience', 10))
    loss_threshold = float(config.get('loss_threshold', 1e-4))

    # supervised output dim
    if task_type == 'multiclass':
        sup_out_dim = out_channels
        metric_name = 'ACC'
    elif task_type == 'binclass':
        sup_out_dim = 1
        metric_name = 'AUC'
    else:
        sup_out_dim = 1
        metric_name = 'RMSE'

    from models.custom.scarf_lib.model import ScarfEncoder, ScarfColumnwiseLayer, ScarfDecoder
    encoder = ScarfEncoder(train_ds.shape[1], dim_hidden_encoder, num_hidden_encoder, dropout).to(device)
    columnwise = ScarfColumnwiseLayer(dim_hidden_encoder).to(device)
    decoder = ScarfDecoder(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout).to(device)

    # GNN modules (mini-batch inductive)
    gnn_encoding = SimpleGCN(dim_hidden_encoder, gnn_hidden, dim_hidden_encoder).to(device) if gnn_stage == 'encoding' else None
    gnn_columnwise = SimpleGCN(dim_hidden_encoder, gnn_hidden, dim_hidden_encoder).to(device) if gnn_stage == 'columnwise' else None
    gnn_decoder = SimpleGCN(dim_hidden_encoder, gnn_hidden, sup_out_dim).to(device) if gnn_stage == 'decoding' else None

    # non-graph supervised head (used when not decoding-stage)
    sup_head = torch.nn.Linear(dim_hidden_encoder, sup_out_dim).to(device) if gnn_decoder is None else None

    # DGM module (optional; used for mini-batch dynamic graph)
    dgm_module = None
    dgm_k_train = int(min(dgm_k, max(1, int(material_outputs.get('batch_size', 128)) - 1)))
    if use_dgm and DGM_AVAILABLE and dgm_k_train >= 1:
        class _DGMEmbedWrapper(torch.nn.Module):
            def forward(self, x, A=None):
                return x

        dgm_module = DGM_d(_DGMEmbedWrapper(), k=dgm_k_train, distance=dgm_distance).to(device)

    def _as_targets(t: torch.Tensor) -> torch.Tensor:
        t = torch.as_tensor(t, device=device)
        if task_type == 'multiclass':
            return t.view(-1).long()
        if task_type == 'binclass':
            return t.view(-1).float()
        return t.view(-1).float()

    def _build_graph(node_feats: torch.Tensor):
        """Return (node_feats_for_gnn, edge_index, dgm_reg_scalar)."""
        Ns = int(node_feats.size(0))
        if Ns <= 1:
            return node_feats, torch.empty((2, 0), dtype=torch.long, device=device), torch.tensor(0.0, device=device)

        node_feats_std = _standardize(node_feats, dim=0)

        if dgm_module is not None and Ns > dgm_k_train:
            node_batched = node_feats_std.unsqueeze(0)
            node_dgm, edge_index_dgm, logprobs_dgm = dgm_module(node_batched, A=None)
            node_dgm = node_dgm.squeeze(0)
            edge_index_dgm = _symmetrize_and_self_loop(edge_index_dgm, Ns)
            dgm_reg = -logprobs_dgm.mean() if isinstance(logprobs_dgm, torch.Tensor) else torch.tensor(0.0, device=device)
            return node_dgm, edge_index_dgm, dgm_reg

        edge_index_knn = knn_graph(node_feats_std.detach(), k=min(gnn_knn, Ns - 1), directed=False).to(device)
        return node_feats_std, edge_index_knn, torch.tensor(0.0, device=device)

    def _apply_stage_gnn(z: torch.Tensor, gnn: torch.nn.Module | None):
        if gnn is None:
            return z, torch.tensor(0.0, device=device)
        z_gnn_in, edge_index, dgm_reg = _build_graph(z)
        if edge_index.numel() == 0:
            return z, dgm_reg
        return gnn(z_gnn_in, edge_index), dgm_reg

    def _supervised_forward(z_clean: torch.Tensor):
        """Return (logits, dgm_reg)"""
        if gnn_decoder is not None:
            z_gnn_in, edge_index, dgm_reg = _build_graph(z_clean)
            if edge_index.numel() == 0:
                logits = torch.zeros((z_clean.size(0), sup_out_dim), device=device)
            else:
                logits = gnn_decoder(z_gnn_in, edge_index)
            return logits, dgm_reg
        return sup_head(z_clean), torch.tensor(0.0, device=device)

    def _supervised_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        if task_type == 'multiclass':
            return F.cross_entropy(logits, y)
        if task_type == 'binclass':
            return F.binary_cross_entropy_with_logits(logits.view(-1), y)
        return F.mse_loss(logits.view(-1), y)

    def _metric_from_logits(logits: torch.Tensor, y: torch.Tensor):
        y_np = y.detach().cpu().numpy()
        if task_type == 'multiclass':
            pred = logits.softmax(dim=-1).argmax(dim=-1).detach().cpu().numpy()
            return float(accuracy_score(y_np, pred))
        if task_type == 'binclass':
            probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
            try:
                return float(roc_auc_score(y_np, probs))
            except Exception:
                return float('nan')
        pred = logits.view(-1).detach().cpu().numpy()
        return float(root_mean_squared_error(y_np, pred))

    # optimizer params
    params = list(encoder.parameters()) + list(columnwise.parameters()) + list(decoder.parameters())
    if gnn_encoding is not None:
        params += list(gnn_encoding.parameters())
    if gnn_columnwise is not None:
        params += list(gnn_columnwise.parameters())
    if gnn_decoder is not None:
        params += list(gnn_decoder.parameters())
    if sup_head is not None:
        params += list(sup_head.parameters())
    if dgm_module is not None:
        params += list(dgm_module.parameters())

    optimizer = Adam(params, lr=lr, weight_decay=weight_decay)
    ntxent_loss = NTXent()

    # SCARF-style feature-wise marginals for corruption sampling
    feat_low = torch.tensor(train_ds.features_low, dtype=torch.float32, device=device)
    feat_high = torch.tensor(train_ds.features_high, dtype=torch.float32, device=device)
    feat_range = (feat_high - feat_low)

    best_val_loss = float('inf')
    best_val_metric = None
    best_state = None
    early_stop_counter = 0
    stopped_early = False
    stop_epoch = epochs

    for epoch in range(1, epochs + 1):
        encoder.train(); columnwise.train(); decoder.train()
        if gnn_encoding is not None:
            gnn_encoding.train()
        if gnn_columnwise is not None:
            gnn_columnwise.train()
        if gnn_decoder is not None:
            gnn_decoder.train()
        if sup_head is not None:
            sup_head.train()
        if dgm_module is not None:
            dgm_module.train()

        train_total = 0.0
        train_count = 0

        for features, targets in train_loader:
            x = torch.as_tensor(features, device=device)
            y = _as_targets(targets)

            # ---- clean path ----
            z = encoder(x)
            z, dgm_reg1 = _apply_stage_gnn(z, gnn_encoding)
            z = columnwise(z)
            z, dgm_reg2 = _apply_stage_gnn(z, gnn_columnwise)

            # ---- contrastive head ----
            anchor = decoder(z)

            # ---- corruption path (contrastive) ----
            corruption_mask = torch.rand_like(x) > corruption_rate
            # Per-feature uniform sampling in [low, high]
            x_random = feat_low + torch.rand_like(x) * feat_range
            x_corrupt = torch.where(corruption_mask, x_random, x)

            z_c = encoder(x_corrupt)
            z_c, _ = _apply_stage_gnn(z_c, gnn_encoding)
            z_c = columnwise(z_c)
            z_c, _ = _apply_stage_gnn(z_c, gnn_columnwise)
            positive = decoder(z_c)

            contrastive = ntxent_loss(anchor, positive)
            logits, dgm_reg3 = _supervised_forward(z)
            sup_loss = _supervised_loss(logits, y)

            total_loss = contrastive + sup_loss_weight * sup_loss + dgm_reg_weight * (dgm_reg1 + dgm_reg2 + dgm_reg3)

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()

            train_total += float(total_loss) * len(x)
            train_count += len(x)

        train_total = train_total / max(1, train_count)

        # ---- validation: use supervised val_loss for early stopping ----
        encoder.eval(); columnwise.eval(); decoder.eval()
        if gnn_encoding is not None:
            gnn_encoding.eval()
        if gnn_columnwise is not None:
            gnn_columnwise.eval()
        if gnn_decoder is not None:
            gnn_decoder.eval()
        if sup_head is not None:
            sup_head.eval()
        if dgm_module is not None:
            dgm_module.eval()

        val_loss_sum = 0.0
        val_count = 0
        val_logits_all = []
        val_y_all = []

        with torch.no_grad():
            for features, targets in val_loader:
                x = torch.as_tensor(features, device=device)
                y = _as_targets(targets)
                z = encoder(x)
                z, _ = _apply_stage_gnn(z, gnn_encoding)
                z = columnwise(z)
                z, _ = _apply_stage_gnn(z, gnn_columnwise)
                logits, _ = _supervised_forward(z)
                vloss = _supervised_loss(logits, y)
                val_loss_sum += float(vloss) * len(x)
                val_count += len(x)
                val_logits_all.append(logits.detach().cpu())
                val_y_all.append(y.detach().cpu())

        val_loss = val_loss_sum / max(1, val_count)
        if val_logits_all:
            val_logits_cat = torch.cat(val_logits_all, dim=0).to(device)
            val_y_cat = torch.cat(val_y_all, dim=0).to(device)
            val_metric = _metric_from_logits(val_logits_cat, val_y_cat)
        else:
            val_metric = float('nan')

        improved = val_loss < best_val_loss - loss_threshold
        if improved:
            best_val_loss = val_loss
            best_val_metric = val_metric
            early_stop_counter = 0
            best_state = {
                'encoder': encoder.state_dict(),
                'columnwise': columnwise.state_dict(),
                'decoder': decoder.state_dict(),
                'gnn_encoding': gnn_encoding.state_dict() if gnn_encoding is not None else None,
                'gnn_columnwise': gnn_columnwise.state_dict() if gnn_columnwise is not None else None,
                'gnn_decoder': gnn_decoder.state_dict() if gnn_decoder is not None else None,
                'sup_head': sup_head.state_dict() if sup_head is not None else None,
                'dgm_module': dgm_module.state_dict() if dgm_module is not None else None,
            }
        else:
            early_stop_counter += 1

        if epoch % 10 == 0 or epoch == 1:
            print(f"[SCARF] epoch {epoch}/{epochs} train_total={train_total:.4f} val_loss={val_loss:.4f} val_{metric_name}={val_metric:.4f} (early_stop {early_stop_counter}/{patience})")

        if early_stop_counter >= patience:
            stopped_early = True
            stop_epoch = epoch
            print(f"[SCARF] Early stopping at epoch {epoch} (val_loss={val_loss:.4f})")
            break

    # ---- restore best weights ----
    if best_state is not None:
        encoder.load_state_dict(best_state['encoder'])
        columnwise.load_state_dict(best_state['columnwise'])
        decoder.load_state_dict(best_state['decoder'])
        if gnn_encoding is not None and best_state['gnn_encoding'] is not None:
            gnn_encoding.load_state_dict(best_state['gnn_encoding'])
        if gnn_columnwise is not None and best_state['gnn_columnwise'] is not None:
            gnn_columnwise.load_state_dict(best_state['gnn_columnwise'])
        if gnn_decoder is not None and best_state['gnn_decoder'] is not None:
            gnn_decoder.load_state_dict(best_state['gnn_decoder'])
        if sup_head is not None and best_state['sup_head'] is not None:
            sup_head.load_state_dict(best_state['sup_head'])
        if dgm_module is not None and best_state['dgm_module'] is not None:
            dgm_module.load_state_dict(best_state['dgm_module'])

    # ---- test metric ----
    encoder.eval(); columnwise.eval(); decoder.eval()
    if gnn_encoding is not None:
        gnn_encoding.eval()
    if gnn_columnwise is not None:
        gnn_columnwise.eval()
    if gnn_decoder is not None:
        gnn_decoder.eval()
    if sup_head is not None:
        sup_head.eval()
    if dgm_module is not None:
        dgm_module.eval()

    test_logits_all = []
    test_y_all = []
    with torch.no_grad():
        for features, targets in test_loader:
            x = torch.as_tensor(features, device=device)
            y = _as_targets(targets)
            z = encoder(x)
            z, _ = _apply_stage_gnn(z, gnn_encoding)
            z = columnwise(z)
            z, _ = _apply_stage_gnn(z, gnn_columnwise)
            logits, _ = _supervised_forward(z)
            test_logits_all.append(logits.detach().cpu())
            test_y_all.append(y.detach().cpu())

    if test_logits_all:
        test_logits_cat = torch.cat(test_logits_all, dim=0).to(device)
        test_y_cat = torch.cat(test_y_all, dim=0).to(device)
        test_metric = _metric_from_logits(test_logits_cat, test_y_cat)
    else:
        test_metric = float('nan')

    pre_stage_gnn_stop = int(material_outputs.get('gnn_early_stop_epochs', 0))
    core_stage_gnn_stop = int(stop_epoch) if stopped_early else 0
    reported_gnn_stop = pre_stage_gnn_stop if gnn_stage in ('start', 'materialize') else core_stage_gnn_stop

    return {
        'best_val_metric': best_val_metric,
        'best_test_metric': test_metric,
        'metric_name': metric_name,
        'early_stop_epochs': int(stop_epoch),
        'gnn_early_stop_epochs': material_outputs.get('gnn_early_stop_epochs', 0),
        'encoder': encoder,
        'columnwise': columnwise,
        'decoder': decoder,
    }


def main(train_df, val_df, test_df, dataset_results, config, gnn_stage):
    print("Scarf - 五階段執行")
    print(f"gnn_stage: {gnn_stage}")
    task_type = dataset_results['info']['task_type']
    # print(f"train_df: {train_df.shape}, val_df: {val_df.shape}, test_df: {test_df.shape}")
    try:
        train_df, val_df, test_df = start_fn(train_df, val_df, test_df)
        gnn_early_stop_epochs = 0
        if gnn_stage=='start':
            train_df, val_df, test_df, gnn_early_stop_epochs = gnn_after_start_fn(train_df, val_df, test_df, config, task_type)
        # scarf 不支援 GNN，直接跳過 GNN 階段
        material_outputs = materialize_fn(train_df, val_df, test_df, dataset_results, config)
        material_outputs['gnn_early_stop_epochs'] = gnn_early_stop_epochs
        if gnn_stage=='materialize':
            material_outputs = gnn_after_materialize_fn(material_outputs,dataset_results, config, task_type)
        results = scarf_core_fn(material_outputs, config, task_type, gnn_stage=gnn_stage)
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



#  small+binclass
#  python main.py --dataset kaggle_Audit_Data --models scarf --gnn_stages all --epochs 2
#  small+regression
#  python main.py --dataset openml_The_Office_Dataset --models scarf --gnn_stages all --epochs 2
#  large+binclass
#  python main.py --dataset credit --models scarf --gnn_stages all --epochs 2
#  large+multiclass
#  python main.py --dataset eye --models scarf --gnn_stages all --epochs 2
#  python main.py --dataset helena --models scarf --gnn_stages all --epochs 2
#  large+regression
#  python main.py --dataset house --models scarf --gnn_stages all --epochs 2