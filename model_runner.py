import sys
import os
import importlib
import torch
import pandas as pd
import numpy as np
import logging
import time
from pathlib import Path
import importlib.util
from typing import Dict, Any, List, Tuple, Optional

# 設置日誌
logger = logging.getLogger(__name__)

class ModelRunner:
    """
    模型運行介面，負責運行不同的表格模型
    """
    def __init__(self, base_dir="./models"):
        """
        初始化模型運行器
        
        Args:
            base_dir: 模型目錄的基礎路徑
        """
        self.base_dir = Path(base_dir)
        self.available_models = self._scan_models()
        self.gnn_hooks = {}  # 階段掛鉤
        
    def _scan_models(self):
        """
        掃描可用的模型
        
        Returns:
            dict: 模型名稱到模型路徑的映射
        """
        available_models = {
            'pytorch_frame': {},
            'custom': {},
            'comparison': {}
        }
        
        # 掃描 pytorch_frame 模型
        pytorch_frame_dir = self.base_dir / "pytorch_frame"
        if pytorch_frame_dir.exists():
            for model_file in pytorch_frame_dir.glob("*.py"):
                model_name = model_file.stem
                available_models['pytorch_frame'][model_name] = str(model_file)
        
        # 掃描 custom 模型
        custom_dir = self.base_dir / "custom"
        if custom_dir.exists():
            for model_file in custom_dir.glob("*.py"):
                model_name = model_file.stem
                available_models['custom'][model_name] = str(model_file)
        
        # 掃描 comparison 模型
        comparison_dir = self.base_dir / "comparison"
        if comparison_dir.exists():
            for model_file in comparison_dir.glob("*.py"):
                model_name = model_file.stem
                available_models['comparison'][model_name] = str(model_file)
        
        logger.info(f"找到 {len(available_models['pytorch_frame'])} 個 pytorch_frame 模型: {list(available_models['pytorch_frame'].keys())}")
        logger.info(f"找到 {len(available_models['custom'])} 個 custom 模型: {list(available_models['custom'].keys())}")
        logger.info(f"找到 {len(available_models['comparison'])} 個 comparison 模型: {list(available_models['comparison'].keys())}")
        
        return available_models
        
    def _load_model_module(self, model_name, model_type):
        """
        動態加載模型模組
        
        Args:
            model_name: 模型名稱
            model_type: 模型類型（pytorch_frame, custom, comparison）
            
        Returns:
            模型模組
        """
        if model_name not in self.available_models[model_type]:
            raise ValueError(f"未找到模型: {model_name} 在類型: {model_type}")
            
        model_path = self.available_models[model_type][model_name]
        
        # 動態加載模組
        spec = importlib.util.spec_from_file_location(model_name, model_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        return module
    
    def register_gnn_hook(self, model_name, stage, hook_fn):
        """
        註冊GNN掛鉤
        
        Args:
            model_name: 模型名稱
            stage: 階段名稱
            hook_fn: 掛鉤函數
        """
        if model_name not in self.gnn_hooks:
            self.gnn_hooks[model_name] = {}
            
        self.gnn_hooks[model_name][stage] = hook_fn
        logger.info(f"已註冊GNN掛鉤: {model_name} 模型的 {stage} 階段")
    
    def _apply_gnn_hooks(self, model_name, module):
        """
        向模型應用GNN掛鉤
        
        Args:
            model_name: 模型名稱
            module: 模型模組
            
        Returns:
            應用掛鉤後的模型模組
        """
        if model_name not in self.gnn_hooks:
            # print(f"模型 {model_name} 沒有註冊GNN掛鉤，跳過掛鉤應用")
            return module
            
        model_hooks = self.gnn_hooks[model_name]
        
        # 為每個階段應用掛鉤
        for stage, hook_fn in model_hooks.items():
            # 這裡不再嘗試修改原始模組，而是在運行時添加掛鉤
            # 將原始的階段函數保存下來，然後替換成新函數
            if hasattr(module, f"{stage}_fn"):
                original_fn = getattr(module, f"{stage}_fn")
                def hooked_fn(*args, **kwargs):
                    # 運行原始函數
                    result = original_fn(*args, **kwargs)
                    # 應用GNN掛鉤
                    return hook_fn(result, *args, **kwargs)
                
                setattr(module, f"{stage}_fn", hooked_fn)
                logger.info(f"已應用GNN掛鉤: {model_name} 模型的 {stage} 階段")
            else:
                logger.warning(f"無法應用掛鉤: {model_name} 模型沒有 {stage} 階段函數")
        
        return module
    
    def run_model(self, model_name, train_df, val_df, test_df, dataset_results, config, model_type, gnn_stage):
        """
        運行指定的模型
        
        Args:
            model_name: 模型名稱
            train_df: 訓練集DataFrame
            val_df: 驗證集DataFrame
            test_df: 測試集DataFrame
            dataset_results: 當前資料集結果dict
            config: 實驗配置
            model_type: 模型類型
        Returns:
            dict: 實驗結果
        """
        logger.info(f"運行 {model_name} 模型")
        
        start_time = time.time()
        
        # 保存原始命令列參數
        original_argv = sys.argv
        sys.argv = [sys.argv[0]]  # 清空命令列參數，僅保留程式名稱
        
        try:
            # 加載模型模組
            module = self._load_model_module(model_name, model_type)
            # 應用GNN掛鉤
            module = self._apply_gnn_hooks(model_name, module)
            
            # 呼叫模型的main函數，傳入三份split
            if hasattr(module, 'main'):
                results = module.main(train_df, val_df, test_df, dataset_results, config, gnn_stage)
            elif hasattr(module, 'run_experiment'):
                results = module.run_experiment(train_df, val_df, test_df, config)
            else:
                for func_name in ['train_and_evaluate', 'run', 'execute']:
                    if hasattr(module, func_name):
                        results = getattr(module, func_name)(train_df, val_df, test_df, config)
                        break
                else:
                    raise ValueError(f"無法在 {model_name} 模型中找到可執行的函數")
        except Exception as e:
            logger.error(f"運行 {model_name} 模型時出錯: {str(e)}")
            return {
                'model': model_name,
                'error': str(e),
                'elapsed_time': time.time() - start_time
            }
        finally:
            # 恢復原始命令列參數
            sys.argv = original_argv
        
        elapsed_time = time.time() - start_time
        
        # 格式化結果
        if isinstance(results, dict):
            results['model'] = model_name
            results['elapsed_time'] = elapsed_time
        else:
            results = {
                'model': model_name,
                'raw_results': results,
                'elapsed_time': elapsed_time
            }
        
        logger.info(f"運行 {model_name} 模型完成，用時: {elapsed_time:.2f}秒")
        
        return results

