import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.metrics import root_mean_squared_error
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader
from models.custom.scarf_lib.loss import NTXent
from models.custom.scarf_lib.model import SCARF
from models.custom.scarf_lib.dataset import SCARFDataset
from models.custom.scarf_lib.utils import get_device, fix_seed, train_epoch
import torch
import numpy as np
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from sklearn.neighbors import NearestNeighbors
import pandas as pd



class SimpleGCN(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = torch.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

def knn_graph(x, k):
    x_np = x.cpu().numpy() if isinstance(x, torch.Tensor) else x
    N = x_np.shape[0]
    
    # 防護：如果樣本數太少，調整 k 值
    if N <= 1:
        # 只有1個或0個樣本，返回空圖
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
        # 如果沒有邊，返回空圖
        return torch.empty((2, 0), dtype=torch.long)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    return edge_index

seed = 42
fix_seed(seed)
def gnn_after_start_fn(train_df, val_df, test_df, config, task_type):
    print("GNN after start function (no-op for Scarf)...")
    print(f"train_df: {train_df.shape}, val_df: {val_df.shape}, test_df: {test_df.shape}")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    # 合併三個df
    all_df = pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True)
    print(f"all_df.head():\n{all_df.head()}")
    feature_cols = [c for c in all_df.columns if c != 'target']
    x = torch.tensor(all_df[feature_cols].values, dtype=torch.float32, device=device)
    y = all_df['target'].values

    # 自動計算 num_classes
    if task_type in ['binclass', 'multiclass']:
        num_classes = len(pd.unique(y))
        print(f"Detected num_classes: {num_classes}")
    else:
        num_classes = 1
    # label 處理
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
    # mask
    n_train = len(train_df)
    n_val = len(val_df)
    n_test = len(test_df)
    N = n_train + n_val + n_test
    train_mask = torch.zeros(N, dtype=torch.bool, device=device)
    val_mask = torch.zeros(N, dtype=torch.bool, device=device)
    test_mask = torch.zeros(N, dtype=torch.bool, device=device)
    train_mask[:n_train] = True
    val_mask[n_train:n_train+n_val] = True
    test_mask[n_train+n_val:] = True

    patience = config.get('gnn_patience', 10)
    best_loss = float('inf')
    early_stop_counter = 0
    # 建立並訓練GNN
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    gnn.train()
    gnn_early_stop_epochs = 0
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(x, edge_index)
        loss = torch.nn.functional.mse_loss(out, x)
        loss.backward()
        optimizer.step()
        # Early stopping check
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
    gnn.eval()
    with torch.no_grad():
        final_emb = gnn(x, edge_index).cpu().numpy()
    # print(f"Final embedding shape: {final_emb.shape}")
    # print(f"final embedding type: {type(final_emb)}")
    # print(f"final embedding head:\n{final_emb[:5]}")
    # 將final_emb分回三個df
    train_emb = final_emb[:n_train]
    val_emb = final_emb[n_train:n_train+n_val]
    test_emb = final_emb[n_train+n_val:]

    emb_cols = [f'N_feature_{i}' for i in range(1,out_dim+1)]
    train_df_gnn = pd.DataFrame(train_emb, columns=emb_cols, index=train_df.index)
    val_df_gnn = pd.DataFrame(val_emb, columns=emb_cols, index=val_df.index)
    test_df_gnn = pd.DataFrame(test_emb, columns=emb_cols, index=test_df.index)
    # print(f"train_df_gnn.shape: {train_df_gnn.shape}")
    # print(f"val_df_gnn.shape: {val_df_gnn.shape}")
    # print(f"test_df_gnn.shape: {test_df_gnn.shape}")
    # print(f"train_df_gnn.head():\n{train_df_gnn.head()}")
    # print(f"val_df_gnn.head():\n{val_df_gnn.head()}")
    # print(f"test_df_gnn.head():\n{test_df_gnn.head()}")

    # 保留原標籤
    train_df_gnn['target'] = train_df['target'].values
    val_df_gnn['target'] = val_df['target'].values
    test_df_gnn['target'] = test_df['target'].values
    # print(f"train_df_gnn.shape: {train_df_gnn.shape}")
    # print(f"val_df_gnn.shape: {val_df_gnn.shape}")
    # print(f"test_df_gnn.shape: {test_df_gnn.shape}")
    # print(f"train_df.head():\n{train_df.head()}")
    # print(f"train_df_gnn.head():\n{train_df_gnn.head()}")
    # print(f"val_df.head():\n{val_df.head()}")
    # print(f"val_df_gnn.head():\n{val_df_gnn.head()}")
    # print(f"test_df.head():\n{test_df.head()}")
    # print(f"test_df_gnn.head():\n{test_df_gnn.head()}")
    

    
    
    

    # 若需要將 num_classes 傳遞到下游，可 return
    return train_df_gnn, val_df_gnn, test_df_gnn, gnn_early_stop_epochs



def gnn_after_materialize_fn(material_outputs, dataset_results, config, task_type):
    print("GNN after materialize function (SCARF 版本，將特徵經 GCN 處理)")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    k = config.get('gnn_knn', 5)
    hidden_dim = config.get('gnn_hidden', 64)
    gnn_epochs = config.get('gnn_epochs', 200)
    patience = config.get('gnn_patience', 10)
    # 1. 從 DataLoader 取出所有特徵與標籤
    def extract_features_targets(loader):
        features_list, targets_list = [], []
        for batch in loader:
            features, targets = batch
            features_list.append(torch.as_tensor(features))
            targets_list.append(torch.as_tensor(targets))
        X = torch.cat(features_list, dim=0)
        y = torch.cat(targets_list, dim=0)
        return X, y
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    train_ds = material_outputs['train_ds']
    val_ds = material_outputs['val_ds']
    test_ds = material_outputs['test_ds']
    train_X, train_y = extract_features_targets(train_loader)
    val_X, val_y = extract_features_targets(val_loader)
    test_X, test_y = extract_features_targets(test_loader)
    # 2. 合併所有資料
    all_X = torch.cat([train_X, val_X, test_X], dim=0).to(device)
    all_y = torch.cat([train_y, val_y, test_y], dim=0)
    n_train, n_val, n_test = len(train_X), len(val_X), len(test_X)
    # 3. 建立 kNN 圖
    edge_index = knn_graph(all_X, k).to(device)
    in_dim = all_X.shape[1]
    out_dim = in_dim
    # 4. 訓練 GNN (SimpleGCN，自編碼器)
    gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device)
    optimizer = torch.optim.Adam(gnn.parameters(), lr=0.01)
    best_loss = float('inf')
    early_stop_counter = 0
    gnn_early_stop_epochs = 0
    gnn.train()
    for epoch in range(gnn_epochs):
        optimizer.zero_grad()
        out = gnn(all_X, edge_index)
        loss = torch.nn.functional.mse_loss(out, all_X)
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
    gnn.eval()
    with torch.no_grad():
        final_emb = gnn(all_X, edge_index).cpu().numpy()
    # 5. 分回 train/val/test
    train_emb = final_emb[:n_train]
    val_emb = final_emb[n_train:n_train+n_val]
    test_emb = final_emb[n_train+n_val:]
    # 6. 建立新的 SCARFDataset/DataLoader
    emb_cols = [f'N_feature_{i}' for i in range(1, out_dim+1)]
    train_df_gnn = pd.DataFrame(train_emb, columns=emb_cols)
    val_df_gnn = pd.DataFrame(val_emb, columns=emb_cols)
    test_df_gnn = pd.DataFrame(test_emb, columns=emb_cols)
    train_df_gnn['target'] = train_y.cpu().numpy()
    val_df_gnn['target'] = val_y.cpu().numpy()
    test_df_gnn['target'] = test_y.cpu().numpy()
    batch_size = material_outputs.get('batch_size', 128)
    train_ds_gnn = SCARFDataset(train_df_gnn[emb_cols].to_numpy(), train_df_gnn['target'].to_numpy(), columns=emb_cols)
    val_ds_gnn = SCARFDataset(val_df_gnn[emb_cols].to_numpy(), val_df_gnn['target'].to_numpy(), columns=emb_cols)
    test_ds_gnn = SCARFDataset(test_df_gnn[emb_cols].to_numpy(), test_df_gnn['target'].to_numpy(), columns=emb_cols)
    train_loader_gnn = DataLoader(train_ds_gnn, batch_size=batch_size, shuffle=True)
    val_loader_gnn = DataLoader(val_ds_gnn, batch_size=batch_size, shuffle=False)
    test_loader_gnn = DataLoader(test_ds_gnn, batch_size=batch_size, shuffle=False)
    # 更新 material_outputs
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
    device = get_device()

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

    print("Running Scarf core function (encoder/columnwise/decoder, with GNN stages)...")
    print(f"material_outputs: {material_outputs}")
    print(f"config: {config}")
    print(f"task_type: {task_type}")
    print(f"gnn_stage: {gnn_stage}")
    batch_size = material_outputs['batch_size']
    epochs = material_outputs['epochs']
    device = material_outputs['device']
    train_loader = material_outputs['train_loader']
    val_loader = material_outputs['val_loader']
    test_loader = material_outputs['test_loader']
    train_ds = material_outputs['train_ds']
    val_ds = material_outputs['val_ds']
    test_ds = material_outputs['test_ds']
    task_type = material_outputs['task_type']
    is_classification = material_outputs['is_classification']
    is_binary_class = material_outputs['is_binary_class']
    out_channels = material_outputs['out_channels']
    gnn_early_stop_epochs = material_outputs.get('gnn_early_stop_epochs', 0)

    # 參數
    dim_hidden_encoder = config.get('dim_hidden_encoder', 8)
    num_hidden_encoder = config.get('num_hidden_encoder', 3)
    dim_hidden_head = config.get('dim_hidden_head', 24)
    num_hidden_head = config.get('num_hidden_head', 2)
    dropout = config.get('dropout', 0.1)
    gnn_hidden = config.get('gnn_hidden', 64)
    gnn_knn = config.get('gnn_knn', 5)
    gnn_epochs = config.get('gnn_epochs', 200)
    gnn_patience = config.get('gnn_patience', 10)


    from models.custom.scarf_lib.model import ScarfEncoder, ScarfColumnwiseLayer, ScarfDecoder
    encoder = ScarfEncoder(train_ds.shape[1], dim_hidden_encoder, num_hidden_encoder, dropout).to(device)
    columnwise = ScarfColumnwiseLayer(dim_hidden_encoder).to(device)
    decoder = ScarfDecoder(dim_hidden_encoder, dim_hidden_head, num_hidden_head, dropout).to(device)

    # GNN 作為子模組
    gnn_encoding = None
    gnn_columnwise = None
    gnn_encoding_edge_index = None  # 移到這裡初始化
    gnn_columnwise_edge_index = None  # 移到這裡初始化
    if gnn_stage == 'encoding':
        gnn_encoding = SimpleGCN(dim_hidden_encoder, gnn_hidden, dim_hidden_encoder).to(device)
        gnn_encoding_edge_index = None
    if gnn_stage == 'columnwise':
        gnn_columnwise = SimpleGCN(dim_hidden_encoder, gnn_hidden, dim_hidden_encoder).to(device)
        gnn_columnwise_edge_index = None

    # 優化器包含 GNN 參數
    params = list(encoder.parameters()) + list(columnwise.parameters()) + list(decoder.parameters())
    if gnn_encoding is not None:
        params += list(gnn_encoding.parameters())
    if gnn_columnwise is not None:
        params += list(gnn_columnwise.parameters())
    optimizer = Adam(params, lr=config.get('lr', 1e-3), weight_decay=config.get('weight_decay', 1e-5))
    ntxent_loss = NTXent()
    patience = config.get('patience', 10)
    early_stop_counter = 0
    best_val_metric = None
    best_model_state = None

    def maybe_gnn(x, gnn, edge_index):
        if gnn is not None and edge_index is not None:
            return gnn(x, edge_index)
        return x

    def get_edge_index(x):
        # x: (batch, dim)
        with torch.no_grad():
            return knn_graph(x.detach(), gnn_knn).to(device)


    for epoch in range(1, epochs + 1):
        encoder.train(); columnwise.train(); decoder.train()
        if gnn_encoding is not None:
            gnn_encoding.train()
        if gnn_columnwise is not None:
            gnn_columnwise.train()
        epoch_loss = 0.0
        total_count = 0
        for features, _ in train_loader:
            features = torch.as_tensor(features).to(device)
                        # SCARF pretraining: anchor
            z = encoder(features)
            
            # encoding GNN
            if gnn_encoding is not None:
                # 重新計算 edge index (使用 detach 避免梯度)
                z_detached = z.detach()  # 用於圖構建
                gnn_encoding_edge_index = get_edge_index(z_detached)
                z = gnn_encoding(z, gnn_encoding_edge_index)
            
            z = columnwise(z)
            
            # columnwise GNN
            if gnn_columnwise is not None:
                # 重新計算 edge index (使用 detach 避免梯度)
                z_detached = z.detach()  # 用於圖構建
                gnn_columnwise_edge_index = get_edge_index(z_detached)
                z = gnn_columnwise(z, gnn_columnwise_edge_index)
            
            anchor = decoder(z)
            
            # corruption - 同樣的過程
            batch_size = features.size(0)
            corruption_mask = torch.rand_like(features, device=features.device) > 0.6
            x_random = torch.rand_like(features)
            features_corrupt = torch.where(corruption_mask, x_random, features)
            
            z_c = encoder(features_corrupt)
            
            if gnn_encoding is not None:
                z_c_detached = z_c.detach()
                gnn_encoding_edge_index = get_edge_index(z_c_detached)
                z_c = gnn_encoding(z_c, gnn_encoding_edge_index)
            
            z_c = columnwise(z_c)
            
            if gnn_columnwise is not None:
                z_c_detached = z_c.detach()
                gnn_columnwise_edge_index = get_edge_index(z_c_detached)
                z_c = gnn_columnwise(z_c, gnn_columnwise_edge_index)
            
            positive = decoder(z_c)
            
            loss = ntxent_loss(anchor, positive)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += float(loss) * len(features)
            total_count += len(features)
        
        epoch_loss = epoch_loss / total_count

        # Validation
        # Validation - 完全重寫避免梯度問題
        encoder.eval(); columnwise.eval(); decoder.eval()
        if gnn_encoding is not None:
            gnn_encoding.eval()
        if gnn_columnwise is not None:
            gnn_columnwise.eval()
            
        val_targets, val_preds = [], []
        
        with torch.no_grad():
            for features, targets in val_loader:
                features = torch.as_tensor(features).to(device)
                
                # Forward pass with explicit no_grad
                z = encoder(features)
                
                if gnn_encoding is not None:
                    edge_idx = get_edge_index(z)
                    z = gnn_encoding(z, edge_idx)
                
                z = columnwise(z)
                
                if gnn_columnwise is not None:
                    edge_idx = get_edge_index(z)
                    z = gnn_columnwise(z, edge_idx)
                
                preds = decoder(z)
                
                # 確保完全分離梯度
                preds_np = preds.detach().cpu().numpy()
                targets_np = targets.detach().cpu().numpy() if hasattr(targets, 'detach') else targets.cpu().numpy()
                
                val_targets.extend(np.ravel(targets_np))
                val_preds.extend(np.atleast_2d(preds_np))
        val_targets = np.array(val_targets)
        val_preds = np.array(val_preds)
        # downstream
        if task_type == 'binclass':
            clf = LogisticRegression(max_iter=1000)
            clf.fit(val_preds, val_targets)
            val_probs = clf.predict_proba(val_preds)[:, 1]
            val_metric = roc_auc_score(val_targets, val_probs)
            metric_name = 'AUC'
            is_better = best_val_metric is None or val_metric > best_val_metric
        elif task_type == 'multiclass':
            clf = LogisticRegression(max_iter=1000)
            clf.fit(val_preds, val_targets)
            val_pred_class = clf.predict(val_preds)
            val_metric = accuracy_score(val_targets, val_pred_class)
            metric_name = 'ACC'
            is_better = best_val_metric is None or val_metric > best_val_metric
        else:
            reg = LinearRegression()
            reg.fit(val_preds, val_targets)
            val_pred_reg = reg.predict(val_preds)
            val_metric = root_mean_squared_error(val_targets, val_pred_reg)
            metric_name = 'RMSE'
            is_better = best_val_metric is None or val_metric < best_val_metric

        if is_better:
            best_val_metric = val_metric
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        print(f"epoch {epoch}/{epochs} - train loss: {epoch_loss:.4f} - val {metric_name}: {val_metric:.4f} (early_stop: {early_stop_counter}/{patience})")
        if early_stop_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    # If we want to use a graph decoder on top of learned embeddings (decoding stage),
    # we replace the linear/logistic decoder with a supervised GNN trained on kNN graph
    # over instance embeddings.
    if gnn_stage == 'decoding':
        def get_all_embeddings_and_targets_scarf(loader, encoder, columnwise, device):
            encs, targets = [], []
            encoder.eval(); columnwise.eval()
            with torch.no_grad():
                for feats, ys in loader:
                    feats = torch.as_tensor(feats, device=device)
                    z = encoder(feats)
                    z = columnwise(z)
                    encs.append(z.detach().cpu())
                    targets.append(torch.as_tensor(ys).view(-1).cpu())
            return torch.cat(encs, dim=0), torch.cat(targets, dim=0)

        device_local = device
        train_emb, train_y = get_all_embeddings_and_targets_scarf(train_loader, encoder, columnwise, device_local)
        val_emb, val_y = get_all_embeddings_and_targets_scarf(val_loader, encoder, columnwise, device_local)
        test_emb, test_y = get_all_embeddings_and_targets_scarf(test_loader, encoder, columnwise, device_local)

        # Build graph on embeddings
        all_emb = torch.cat([train_emb, val_emb, test_emb], dim=0).to(device_local)
        all_y_np = np.concatenate([train_y.numpy(), val_y.numpy(), test_y.numpy()], axis=0)
        k = config.get('gnn_knn', 5)
        edge_index = knn_graph(all_emb.detach(), k).to(device_local)

        n_train, n_val, n_test = len(train_emb), len(val_emb), len(test_emb)
        N = n_train + n_val + n_test
        train_mask = torch.zeros(N, dtype=torch.bool, device=device_local); train_mask[:n_train] = True
        val_mask = torch.zeros(N, dtype=torch.bool, device=device_local); val_mask[n_train:n_train+n_val] = True
        test_mask = torch.zeros(N, dtype=torch.bool, device=device_local); test_mask[n_train+n_val:] = True

        # Prepare labels tensor
        if task_type == 'binclass':
            y_tensor = torch.tensor(all_y_np, dtype=torch.float32, device=device_local)
            out_dim = 1
            metric_name = 'AUC'
        elif task_type == 'multiclass':
            y_tensor = torch.tensor(all_y_np, dtype=torch.long, device=device_local)
            num_classes = int(np.unique(all_y_np).shape[0])
            out_dim = num_classes
            metric_name = 'ACC'
        else:
            y_tensor = torch.tensor(all_y_np, dtype=torch.float32, device=device_local)
            out_dim = 1
            metric_name = 'RMSE'

        in_dim = all_emb.shape[1]
        hidden_dim = config.get('gnn_hidden', 64)
        gnn_epochs = config.get('gnn_epochs', 200)
        patience = config.get('gnn_patience', 10)

        gnn = SimpleGCN(in_dim, hidden_dim, out_dim).to(device_local)
        opt = torch.optim.Adam(gnn.parameters(), lr=0.01)

        def compute_metric(logits):
            with torch.no_grad():
                if task_type == 'binclass':
                    probs = torch.sigmoid(logits.view(-1))
                    return roc_auc_score(y_tensor[test_mask].cpu().numpy(), probs[test_mask].cpu().numpy())
                elif task_type == 'multiclass':
                    pred = logits.softmax(dim=-1).argmax(dim=-1)
                    return accuracy_score(y_tensor[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy())
                else:
                    pred = logits.view(-1)
                    return root_mean_squared_error(y_tensor[test_mask].cpu().numpy(), pred[test_mask].cpu().numpy())
        
        best_val = -float('inf') if task_type in ['binclass','multiclass'] else float('inf')
        best_val_metric, best_test_metric = None, None
        early_stop_counter = 0
        best_epoch = 0
        stop_epoch = 0

        for ep in range(gnn_epochs):
            gnn.train()
            opt.zero_grad()
            logits = gnn(all_emb, edge_index)
            if task_type == 'binclass':
                loss = F.binary_cross_entropy_with_logits(logits.view(-1)[train_mask], y_tensor[train_mask])
            elif task_type == 'multiclass':
                loss = F.cross_entropy(logits[train_mask], y_tensor[train_mask])
            else:
                loss = F.mse_loss(logits.view(-1)[train_mask], y_tensor[train_mask])
            loss.backward()
            opt.step()

            # Validation
            gnn.eval()
            with torch.no_grad():
                logits = gnn(all_emb, edge_index)
                if task_type == 'binclass':
                    if val_mask.any():
                        val_probs = torch.sigmoid(logits.view(-1)[val_mask]).cpu().numpy()
                        val_score = roc_auc_score(y_tensor[val_mask].cpu().numpy(), val_probs)
                    else:
                        val_score = float('nan')
                    test_score = compute_metric(logits)
                    better = (best_val_metric is None) or (val_score > best_val)
                elif task_type == 'multiclass':
                    if val_mask.any():
                        val_pred = logits[val_mask].softmax(dim=-1).argmax(dim=-1)
                        val_score = accuracy_score(y_tensor[val_mask].cpu().numpy(), val_pred.cpu().numpy())
                    else:
                        val_score = float('nan')
                    test_score = compute_metric(logits)
                    better = (best_val_metric is None) or (val_score > best_val)
                else:
                    if val_mask.any():
                        val_pred = logits.view(-1)[val_mask]
                        val_score = root_mean_squared_error(y_tensor[val_mask].cpu().numpy(), val_pred.cpu().numpy())
                    else:
                        val_score = float('nan')
                    test_score = compute_metric(logits)
                    better = (best_val_metric is None) or (val_score < best_val)

            if better and not np.isnan(val_score):
                best_val = val_score
                best_val_metric = val_score
                best_test_metric = test_score
                early_stop_counter = 0
                best_epoch = ep + 1
            else:
                early_stop_counter += 1
            if early_stop_counter >= patience:
                stop_epoch = ep + 1
                print(f"GNN decoding early stopping at epoch {stop_epoch} (patience {patience})")
                break
        else:
            # no break: finished all epochs
            stop_epoch = gnn_epochs

        print(f"Decoding-stage best val {metric_name}: {best_val_metric:.4f}; test {best_test_metric:.4f} at epoch {best_epoch}")
        return {
            'best_val_metric': best_val_metric,
            'best_test_metric': best_test_metric,
            'metric_name': metric_name,
            # Keep base (SCARF pretraining) early_stop consistent with the outer loop's epoch var
            'early_stop_epochs': epoch,
            # Report GNN decoding stop epoch (when patience triggered or max epochs reached)
            'gnn_early_stop_epochs': stop_epoch,
            'encoder': encoder,
            'columnwise': columnwise,
            'decoder': decoder,
        }

    print(f"best val {metric_name}: {best_val_metric:.4f}")
    # Test
    encoder.eval(); columnwise.eval(); decoder.eval()
    if gnn_encoding is not None:
        gnn_encoding.eval()
    if gnn_columnwise is not None:
        gnn_columnwise.eval()
        
    test_targets, test_preds = [], []
    
    with torch.no_grad():
        for features, targets in test_loader:
            features = torch.as_tensor(features).to(device)
            
            z = encoder(features)
            
            if gnn_encoding is not None:
                edge_idx = get_edge_index(z)
                z = gnn_encoding(z, edge_idx)
            
            z = columnwise(z)
            
            if gnn_columnwise is not None:
                edge_idx = get_edge_index(z)
                z = gnn_columnwise(z, edge_idx)
            
            preds = decoder(z)
            
            # 確保完全分離梯度
            preds_np = preds.detach().cpu().numpy()
            targets_np = targets.detach().cpu().numpy() if hasattr(targets, 'detach') else targets.cpu().numpy()
            
            test_targets.extend(np.ravel(targets_np))
            test_preds.extend(np.atleast_2d(preds_np))
    test_targets = np.array(test_targets)
    test_preds = np.array(test_preds)
    if task_type == 'binclass':
        clf = LogisticRegression(max_iter=1000)
        clf.fit(test_preds, test_targets)
        test_probs = clf.predict_proba(test_preds)[:, 1]
        test_metric = roc_auc_score(test_targets, test_probs)
        metric_name = 'AUC'
    elif task_type == 'multiclass':
        clf = LogisticRegression(max_iter=1000)
        clf.fit(test_preds, test_targets)
        test_pred_class = clf.predict(test_preds)
        test_metric = accuracy_score(test_targets, test_pred_class)
        metric_name = 'ACC'
    else:
        reg = LinearRegression()
        reg.fit(test_preds, test_targets)
        test_pred_reg = reg.predict(test_preds)
        test_metric = root_mean_squared_error(test_targets, test_pred_reg)
        metric_name = 'RMSE'
    print(f"\nFinal test {metric_name}: {test_metric:.4f}")
    return {
        'best_val_metric': best_val_metric,
        'best_test_metric': test_metric,
        'metric_name': metric_name,
        'early_stop_epochs': epoch,
        'gnn_early_stop_epochs': 0 if gnn_stage != 'decoding' else gnn_early_stop_epochs,
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