import os
import logging
import time
import json
import numpy as np
import pandas as pd
import torch
from pathlib import Path

from utils.data_utils import DatasetLoader
from injectors.injector_base import get_injector

logger = logging.getLogger(__name__)

class ExperimentConfig:
    """實驗配置類"""
    def __init__(self):
        # 數據配置
        self.dataset_size = None  # 'small_datasets' or 'large_datasets'
        self.task_type = None     # 'binclass', 'multiclass', or 'regression'
        self.feature_type = None  # 'numerical', 'categorical', or 'balanced'
        self.datasets = []        # 資料集名稱列表
        self.data_dir = './datasets'
        self.train_val_test_split_ratio = [0.7, 0.1, 0.2]  # 訓練、驗證、測試集比例
        self.few_shot = False
        self.few_shot_ratio = 0.1
        
        # 模型配置
        self.models = []  # 模型名稱列表
        self.gnn_stages = []  # GNN注入階段列表
        
        # GNN配置
        self.gnn_config = {
            'hidden_dim': 64,
            'num_layers': 2,
            'dropout': 0.2
        }
        
        # 訓練配置
        self.train_config = {
            'epochs': 100,
            'batch_size': 128,
            'lr': 0.001,
            'weight_decay': 1e-5,
            'patience': 10
        }
        
        # 其他配置
        self.seed = 42
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.output_dir = './results'
        self.exp_name = 'experiment'
        
    def to_dict(self):
        """轉換配置為字典"""
        return {k: v for k, v in self.__dict__.items() if not k.startswith('_')}

class ExperimentRunner:
    """實驗運行器類"""
    def __init__(self, config):
        """
        初始化實驗運行器
        
        Args:
            config: 實驗配置
        """
        self.config = config
        self.dataset_loader = DatasetLoader(config.data_dir)
        self.results = {}
        
    def get_model_instance(self, model_name):
        """
        獲取模型類
        
        Args:
            model_name: 模型名稱
            
        Returns:
            模型類
        """
        # 根據模型名稱導入對應的模型類
        if model_name.lower() == 'excelformer':
            from models.pytorch_frame.excelformer import ExcelFormer
            return ExcelFormer
        elif model_name.lower() == 'resnet':
            from models.pytorch_frame.resnet import ResNet
            return ResNet
        elif model_name.lower() == 'fttransformer':
            from models.pytorch_frame.fttransformer import FTTransformer
            return FTTransformer
        elif model_name.lower() == 'tabnet':
            from models.pytorch_frame.tabnet import TabNet
            return TabNet
        elif model_name.lower() == 'tabtransformer':
            from models.pytorch_frame.tabtransformer import TabTransformer
            return TabTransformer
        elif model_name.lower() == 'trompt':
            from models.pytorch_frame.trompt import Trompt
            return Trompt
        elif model_name.lower() == 'vime':
            from models.custom.vime_wrapper import VIMEWrapper
            return VIMEWrapper
        elif model_name.lower() == 'subtab':
            from models.custom.subtab_wrapper import SubTabWrapper
            return SubTabWrapper
        elif model_name.lower() == 'scarf':
            from models.custom.scarf_wrapper import ScarfWrapper
            return ScarfWrapper
        else:
            raise ValueError(f"不支持的模型: {model_name}")
    
    def _inject_gnn(self, model, model_name, gnn_stage):
        """
        向模型注入GNN
        
        Args:
            model: 原始模型
            model_name: 模型名稱
            gnn_stage: GNN注入階段
            
        Returns:
            注入GNN後的模型
        """
        if gnn_stage == 'none':
            return model
            
        # 獲取適當的注入器
        injector = get_injector(model_name, gnn_stage)
        
        # 注入GNN
        return injector.inject(model, self.config.gnn_config)
    
    def run_single_experiment(self, dataset_name, model_name, gnn_stage):
        """
        運行單個實驗
        
        Args:
            dataset_name: 資料集名稱
            model_name: 模型名稱
            gnn_stage: GNN注入階段
            
        Returns:
            實驗結果
        """
        logger.info(f"運行實驗: 資料集={dataset_name}, 模型={model_name}, GNN階段={gnn_stage}")
        
        # 加載資料集
        dataset_info = self.dataset_loader.load_dataset(dataset_name)
        df = dataset_info['df']
        
        # 獲取模型類
        ModelClass = self.get_model_instance(model_name)
        
        # 創建模型參數
        model_args = {
            'dataset': dataset_name,
            'df': df,
            'train_val_test_split_ratio': self.config.train_val_test_split_ratio,
            'batch_size': self.config.train_config['batch_size'],
            'epochs': self.config.train_config['epochs'],
            'lr': self.config.train_config['lr'],
            'weight_decay': self.config.train_config['weight_decay'],
            'patience': self.config.train_config['patience'],
            'device': self.config.device,
            'seed': self.config.seed,
            'few_shot': self.config.few_shot,
            'few_shot_ratio': self.config.few_shot_ratio
        }
        
        # 初始化模型
        model = ModelClass(**model_args)
        
        # 注入GNN (如果需要)
        if gnn_stage != 'none':
            model = self._inject_gnn(model, model_name, gnn_stage)
        
        # 訓練和評估模型
        # 注意：這裡假設每個模型類都有 train() 和 evaluate() 方法
        try:
            # 訓練模型
            train_results = model.train()
            
            # 評估模型
            eval_results = model.evaluate()
            
            # 合併結果
            results = {
                'dataset': dataset_name,
                'model': model_name,
                'gnn_stage': gnn_stage,
                'train_results': train_results,
                'eval_results': eval_results,
            }
            
            return results
        except Exception as e:
            logger.error(f"實驗失敗: {str(e)}")
            return {
                'dataset': dataset_name,
                'model': model_name,
                'gnn_stage': gnn_stage,
                'error': str(e)
            }
    
    def run(self):
        """
        運行所有實驗
        
        Returns:
            所有實驗結果
        """
        # 根據配置決定要運行的實驗
        all_results = []
        
        # 獲取資料集列表
        if self.config.datasets:
            datasets = self.config.datasets
        else:
            datasets = self.dataset_loader.get_datasets_by_category(
                self.config.dataset_size,
                self.config.task_type,
                self.config.feature_type
            )
            
        # 獲取模型列表
        if not self.config.models:
            self.config.models = ['excelformer', 'resnet', 'fttransformer', 
                                'tabnet', 'tabtransformer', 'trompt',
                                'vime', 'subtab', 'scarf']
            
        # 獲取GNN注入階段列表
        if not self.config.gnn_stages:
            self.config.gnn_stages = ['none', 'encoding', 'columnwise', 'decoding']
            
        # 運行所有模型和GNN注入階段的組合
        for dataset_name in datasets:
            dataset_results = {}
            
            for model_name in self.config.models:
                model_results = {}
                
                for gnn_stage in self.config.gnn_stages:
                    try:
                        # 運行實驗
                        result = self.run_single_experiment(dataset_name, model_name, gnn_stage)
                        model_results[gnn_stage] = result
                    except Exception as e:
                        logger.error(f"實驗失敗: {dataset_name}, {model_name}, {gnn_stage}, 錯誤: {str(e)}")
                        model_results[gnn_stage] = {'error': str(e)}
                
                dataset_results[model_name] = model_results
            
            all_results.append({
                'dataset': dataset_name,
                'results': dataset_results
            })
        
        # 計算排名
        self._compute_model_rankings(all_results)
        
        # 保存所有結果
        self.results = {
            'all_results': all_results,
            'rankings': self.rankings
        }
        
        return self.results
    
    def _compute_model_rankings(self, all_results):
        """
        計算不同模型和GNN注入階段的排名
        
        Args:
            all_results: 所有實驗結果
        """
        # 按照資料集類型組織結果
        categorized_results = {}
        
        for dataset_result in all_results:
            dataset_name = dataset_result['dataset']
            dataset_info = self.dataset_loader.get_dataset_info(dataset_name)
            
            # 創建類別鍵
            category_key = f"{dataset_info['size']}_{dataset_info['task_type']}_{dataset_info['feature_type']}"
            
            if category_key not in categorized_results:
                categorized_results[category_key] = []
                
            categorized_results[category_key].append(dataset_result)
        
        # 計算每個類別下的模型排名
        rankings = {}
        
        for category, results in categorized_results.items():
            category_rankings = {}
            
            # 收集所有模型和階段的性能指標
            model_performances = {}
            
            for dataset_result in results:
                for model_name, model_results in dataset_result['results'].items():
                    if model_name not in model_performances:
                        model_performances[model_name] = {}
                        
                    for gnn_stage, result in model_results.items():
                        if 'error' in result or 'eval_results' not in result:
                            continue
                            
                        if gnn_stage not in model_performances[model_name]:
                            model_performances[model_name][gnn_stage] = []
                        
                        # 獲取性能指標（假設eval_results中有performance鍵）
                        performance = result['eval_results'].get('performance', 0)
                        model_performances[model_name][gnn_stage].append({
                            'dataset': dataset_result['dataset'],
                            'performance': performance
                        })
            
            # 計算平均性能
            avg_performances = {}
            
            for model_name, stage_performances in model_performances.items():
                avg_performances[model_name] = {}
                
                for gnn_stage, performances in stage_performances.items():
                    if performances:
                        avg_performance = sum(p['performance'] for p in performances) / len(performances)
                        avg_performances[model_name][gnn_stage] = avg_performance
            
            # 建立排名
            # 1. 每個模型的最佳階段
            model_best_stages = {}
            
            for model_name, stage_performances in avg_performances.items():
                if stage_performances:
                    best_stage = max(stage_performances.items(), key=lambda x: x[1])
                    model_best_stages[model_name] = {
                        'stage': best_stage[0],
                        'performance': best_stage[1]
                    }
            
            # 2. 模型排名（基於最佳階段性能）
            model_ranking = sorted(model_best_stages.items(), key=lambda x: x[1]['performance'], reverse=True)
            
            # 3. GNN階段對每個模型的影響
            gnn_impact = {}
            
            for model_name, stage_performances in avg_performances.items():
                if 'none' in stage_performances and len(stage_performances) > 1:
                    base_performance = stage_performances['none']
                    
                    for gnn_stage, performance in stage_performances.items():
                        if gnn_stage != 'none':
                            relative_improvement = (performance - base_performance) / max(base_performance, 1e-10)
                            
                            if gnn_stage not in gnn_impact:
                                gnn_impact[gnn_stage] = []
                                
                            gnn_impact[gnn_stage].append({
                                'model': model_name,
                                'improvement': relative_improvement
                            })
            
            # 計算每個GNN階段的平均提升
            avg_gnn_impact = {}
            
            for gnn_stage, improvements in gnn_impact.items():
                if improvements:
                    avg_improvement = sum(imp['improvement'] for imp in improvements) / len(improvements)
                    avg_gnn_impact[gnn_stage] = avg_improvement
            
            # 排序GNN階段（基於平均提升）
            gnn_ranking = sorted(avg_gnn_impact.items(), key=lambda x: x[1], reverse=True)
            
            # 保存排名結果
            category_rankings['model_ranking'] = model_ranking
            category_rankings['gnn_ranking'] = gnn_ranking
            category_rankings['model_best_stages'] = model_best_stages
            category_rankings['avg_performances'] = avg_performances
            
            rankings[category] = category_rankings
        
        self.rankings = rankings