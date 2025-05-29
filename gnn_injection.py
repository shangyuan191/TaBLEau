import torch
import torch.nn as nn
import numpy as np
import logging
from typing import Dict, Any, List, Tuple, Optional
from torch_geometric.nn import GCNConv

logger = logging.getLogger(__name__)

class TabularGNN(nn.Module):
    """
    表格資料GNN模型
    """
    def __init__(self, config):
        """
        初始化GNN模型
        
        Args:
            config: GNN配置
        """
        super().__init__()
        self.hidden_dim = config.get('hidden_dim', 64)
        self.num_layers = config.get('num_layers', 2)
        self.dropout = config.get('dropout', 0.2)
        
        # 延遲初始化，因為輸入維度在應用時才能確定
        self.input_dim = None
        self.output_dim = None
        self.gcn_layers = None
        self.dropout_layer = nn.Dropout(self.dropout)
        
    def _init_network(self, input_dim, output_dim=None):
        """
        初始化GNN網絡
        
        Args:
            input_dim: 輸入維度
            output_dim: 輸出維度 (如果為None，則使用輸入維度)
        """
        self.input_dim = input_dim
        self.output_dim = output_dim if output_dim is not None else input_dim
        
        # 創建GCN層
        gcn_layers = []
        
        # 輸入層
        gcn_layers.append(GCNConv(self.input_dim, self.hidden_dim))
        
        # 隱藏層
        for _ in range(self.num_layers - 2):
            gcn_layers.append(GCNConv(self.hidden_dim, self.hidden_dim))
            
        # 輸出層
        gcn_layers.append(GCNConv(self.hidden_dim, self.output_dim))
        
        self.gcn_layers = nn.ModuleList(gcn_layers)
        
    def _construct_graph(self, x):
        """
        根據輸入數據構造圖
        
        Args:
            x: 輸入數據
            
        Returns:
            tuple: (節點特徵, 邊索引)
        """
        # 根據輸入維度選擇合適的圖構造方法
        if x.dim() == 3:  # [batch_size, num_columns, feature_dim]
            return self._construct_column_graph(x)
        elif x.dim() == 2:  # [batch_size, feature_dim]
            return self._construct_row_graph(x)
        else:
            raise ValueError(f"不支持的輸入維度: {x.shape}")
    
    def _construct_column_graph(self, x):
        """
        構造基於列的圖
        
        Args:
            x: 輸入數據 [batch_size, num_columns, feature_dim]
            
        Returns:
            tuple: (節點特徵, 邊索引)
        """
        batch_size, num_columns, feature_dim = x.shape
        
        # 將每一列視為一個節點
        # 重塑為 [batch_size * num_columns, feature_dim]
        node_features = x.reshape(-1, feature_dim)
        
        # 構造完全連接圖，每個batch中的列互相連接
        edge_list = []
        
        for b in range(batch_size):
            offset = b * num_columns
            # 每個batch中的完全連接圖
            for i in range(num_columns):
                for j in range(num_columns):
                    if i != j:  # 不包括自環
                        edge_list.append([offset + i, offset + j])
        
        edge_index = torch.tensor(edge_list, device=x.device).t().contiguous()
        
        return node_features, edge_index
    
    def _construct_row_graph(self, x):
        """
        構造基於行的圖
        
        Args:
            x: 輸入數據 [batch_size, feature_dim]
            
        Returns:
            tuple: (節點特徵, 邊索引)
        """
        batch_size, feature_dim = x.shape
        
        # 每個樣本作為一個節點
        node_features = x
        
        # 構造k近鄰圖
        k = min(10, batch_size - 1)  # k不能大於樣本數-1
        
        # 計算樣本之間的歐氏距離
        dist = torch.cdist(x, x)
        
        # 獲取每個樣本的k個最近鄰
        _, indices = torch.topk(dist, k + 1, dim=1, largest=False)
        indices = indices[:, 1:]  # 排除自身
        
        # 構造邊
        edge_list = []
        for i in range(batch_size):
            for j in indices[i]:
                edge_list.append([i, j.item()])
        
        edge_index = torch.tensor(edge_list, device=x.device).t().contiguous()
        
        return node_features, edge_index
    
    def _reshape_output(self, node_features, original_x):
        """
        將GNN輸出重塑為原始形狀
        
        Args:
            node_features: GNN輸出的節點特徵
            original_x: 原始輸入
            
        Returns:
            重塑後的輸出
        """
        if original_x.dim() == 3:  # [batch_size, num_columns, feature_dim]
            batch_size, num_columns, _ = original_x.shape
            return node_features.reshape(batch_size, num_columns, -1)
        elif original_x.dim() == 2:  # [batch_size, feature_dim]
            return node_features
        else:
            raise ValueError(f"不支持的輸入維度: {original_x.shape}")
    
    def forward(self, x):
        """
        前向傳播
        
        Args:
            x: 輸入數據
            
        Returns:
            GNN處理後的輸出
        """
        # 延遲初始化
        if self.gcn_layers is None:
            if x.dim() == 3:
                self._init_network(x.size(2))
            elif x.dim() == 2:
                self._init_network(x.size(1))
            else:
                raise ValueError(f"不支持的輸入維度: {x.shape}")
        
        # 構造圖
        node_features, edge_index = self._construct_graph(x)
        
        # 應用GNN
        h = node_features
        
        for i, gcn in enumerate(self.gcn_layers[:-1]):
            h = gcn(h, edge_index)
            h = torch.relu(h)
            h = self.dropout_layer(h)
            
        h = self.gcn_layers[-1](h, edge_index)
        
        # 重塑輸出
        output = self._reshape_output(h, x)
        
        return output

def get_gnn_hook_fn(config, stage):
    """
    獲取GNN掛鉤函數
    
    Args:
        config: GNN配置
        stage: 注入階段
        
    Returns:
        掛鉤函數
    """
    # 創建GNN模型
    gnn = TabularGNN(config)
    
    def gnn_hook_fn(x, *args, **kwargs):
        """
        GNN掛鉤函數
        
        Args:
            x: 輸入數據
            
        Returns:
            GNN處理後的輸出
        """
        # 檢查輸入是否為張量
        if not isinstance(x, torch.Tensor):
            logger.warning(f"GNN掛鉤收到非張量輸入: {type(x)}")
            return x
            
        # 將GNN移至與輸入相同的設備
        device = x.device
        gnn.to(device)
        
        # 應用GNN
        return gnn(x)
    
    return gnn_hook_fn

class GNNInjector:
    """
    GNN注入器，用於向不同模型的不同階段注入GNN
    """
    def __init__(self, model_runner):
        """
        初始化GNN注入器
        
        Args:
            model_runner: 模型運行器
        """
        self.model_runner = model_runner
        # print(f"self.model_runner: {self.model_runner}")
        
    def inject(self, model_name, stage, config):
        """
        向指定模型的指定階段注入GNN
        
        Args:
            model_name: 模型名稱
            stage: 注入階段
            config: GNN配置
            
        Returns:
            是否注入成功
        """
        if stage == 'none':
            logger.info(f"跳過GNN注入: {model_name} 模型的 {stage} 階段")
            return True
            
        # 獲取GNN掛鉤函數
        hook_fn = get_gnn_hook_fn(config, stage)
        
        # 註冊掛鉤
        try:
            self.model_runner.register_gnn_hook(model_name, stage, hook_fn)
            return True
        except Exception as e:
            logger.error(f"GNN注入失敗: {model_name} 模型的 {stage} 階段: {str(e)}")
            return False

# 階段到函數名映射
STAGE_TO_FUNCTION = {
    'start': 'start',
    'materialize': 'materialize',
    'encoding': 'encode',
    'columnwise': 'column_interact',
    'decoding': 'decode'
}

def adapt_official_models():
    """
    適配官方模型，使其支持GNN注入
    
    這個函數需要在導入模型之前運行，它會修改sys.meta_path來攔截模型導入
    """
    import sys
    from importlib.abc import MetaPathFinder, Loader
    from importlib.machinery import ModuleSpec
    import types
    
    class ModelPatcher(MetaPathFinder, Loader):
        """
        模型修補器，用於在導入時修改模型
        """
        def __init__(self):
            self.original_meta_path = sys.meta_path.copy()
            
        def find_spec(self, fullname, path, target=None):
            """查找模組規範"""
            # 僅處理pytorch_frame下的模型
            if not fullname.startswith('models.pytorch_frame.'):
                return None
                
            # 查找原始規範
            for finder in self.original_meta_path:
                if finder is self:
                    continue
                    
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    # 替換加載器為自己
                    spec.loader = self
                    return spec
                    
            return None
            
        def create_module(self, spec):
            """創建模組"""
            return None  # 使用默認行為
            
        def exec_module(self, module):
            """執行模組"""
            # 獲取模組名稱
            model_name = module.__name__.split('.')[-1]
            
            # 導入原始模組
            original_module = types.ModuleType(module.__name__)
            original_module.__file__ = module.__file__
            
            with open(module.__file__, 'r') as f:
                code = compile(f.read(), module.__file__, 'exec')
                exec(code, original_module.__dict__)
            
            # 複製原始模組的屬性到新模組
            for name, attr in original_module.__dict__.items():
                if not name.startswith('__'):
                    setattr(module, name, attr)
            
            # 添加階段函數
            for stage, func_name in STAGE_TO_FUNCTION.items():
                # 檢查是否已有對應函數
                if not hasattr(module, f"{stage}_fn") and hasattr(module, func_name):
                    # 保存原始函數
                    original_fn = getattr(module, func_name)
                    
                    # 設置階段函數
                    setattr(module, f"{stage}_fn", original_fn)
                    
                    logger.info(f"為 {model_name} 模型添加了 {stage} 階段函數")
    
    # 安裝修補器
    patcher = ModelPatcher()
    sys.meta_path.insert(0, patcher)
    
    return patcher