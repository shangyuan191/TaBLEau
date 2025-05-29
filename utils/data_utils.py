import os
import glob
import pandas as pd
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class DatasetLoader:
    """
    資料集加載器，僅負責掃描和加載原始CSV文件
    """
    def __init__(self, base_dir):
        """
        初始化資料集加載器
        
        Args:
            base_dir: 資料集的基礎目錄
        """
        self.base_dir = Path(base_dir)
        self.dataset_info = {}  # 存儲資料集信息
        self._scan_datasets()
    
    def _scan_datasets(self):
        """掃描所有可用的資料集"""
        # 掃描所有資料集
        for size in ['small_datasets', 'large_datasets']:
            size_dir = self.base_dir / size
            if not size_dir.exists():
                continue
            
            for task_type in ['binclass', 'multiclass', 'regression']:
                task_dir = size_dir / task_type
                if not task_dir.exists():
                    continue
                
                for feature_type in ['numerical', 'categorical', 'balanced']:
                    feature_dir = task_dir / feature_type
                    if not feature_dir.exists():
                        continue
                    
                    # 獲取此類型下的所有資料集
                    # dataset_dirs = [d for d in feature_dir.iterdir() if d.is_dir()]
                    dataset_dirs = [d for d in feature_dir.iterdir() if d.is_dir() and d.name != 'covtype']
                    
                    for dataset_dir in dataset_dirs:
                        dataset_name = dataset_dir.name
                        self.dataset_info[dataset_name] = {
                            'size': size,
                            'task_type': task_type,
                            'feature_type': feature_type,
                            'path': str(dataset_dir)
                        }
        
        logger.info(f"找到 {len(self.dataset_info)} 個資料集")
        # for dataset_name, info in self.dataset_info.items():
        #     print(f"資料集名稱: {dataset_name}, 大小: {info['size']}, 任務類型: {info['task_type']}, 特徵類型: {info['feature_type']}, 路徑: {info['path']}")
    def get_dataset_categories(self):
        """
        返回資料集類別統計
        
        Returns:
            dict: 包含各類別資料集數量的字典
        """
        categories = {
            'small_datasets': {'binclass': {'numerical': 0, 'categorical': 0, 'balanced': 0},
                              'multiclass': {'numerical': 0, 'categorical': 0, 'balanced': 0},
                              'regression': {'numerical': 0, 'categorical': 0, 'balanced': 0}},
            'large_datasets': {'binclass': {'numerical': 0, 'categorical': 0, 'balanced': 0},
                              'multiclass': {'numerical': 0, 'categorical': 0, 'balanced': 0},
                              'regression': {'numerical': 0, 'categorical': 0, 'balanced': 0}}
        }
        
        for info in self.dataset_info.values():
            size = info['size']
            task_type = info['task_type']
            feature_type = info['feature_type']
            categories[size][task_type][feature_type] += 1
            
        return categories
    
    def get_datasets_by_category(self, size=None, task_type=None, feature_type=None):
        """
        根據類別獲取資料集列表
        
        Args:
            size: 資料集大小 ('small_datasets' 或 'large_datasets')
            task_type: 任務類型 ('binclass', 'multiclass', 或 'regression')
            feature_type: 特徵類型 ('numerical', 'categorical', 或 'balanced')
            
        Returns:
            list: 符合條件的資料集名稱列表
        """
        filtered_datasets = []
        
        for name, info in self.dataset_info.items():
            if size and info['size'] != size:
                continue
            if task_type and info['task_type'] != task_type:
                continue
            if feature_type and info['feature_type'] != feature_type:
                continue
            
            filtered_datasets.append(name)
            
        return filtered_datasets
    
    def load_dataset(self, dataset_name):
        """
        加載指定的資料集的原始CSV文件
        
        Args:
            dataset_name: 資料集名稱
            
        Returns:
            dict: 包含資料集信息和原始DataFrame
        """
        if dataset_name not in self.dataset_info:
            raise ValueError(f"未找到資料集: {dataset_name}")
            
        info = self.dataset_info[dataset_name]
        dataset_path = Path(info['path'])
        
        # 嘗試查找CSV文件
        data_files = list(dataset_path.glob('*.csv'))
        if not data_files:
            data_files = list(dataset_path.glob('*.CSV'))
        if not data_files:
            data_files = list(dataset_path.glob('*.data'))
        if not data_files:
            data_files = list(dataset_path.glob('*.arff'))
            
        if not data_files:
            raise ValueError(f"在 {dataset_path} 中未找到數據文件")
            
        data_file = data_files[0]  # 使用找到的第一個數據文件
        
        # 嘗試加載數據
        try:
            if data_file.suffix.lower() == '.csv':
                df = pd.read_csv(data_file)
            elif data_file.suffix.lower() == '.arff':
                from scipy.io import arff
                data, meta = arff.loadarff(data_file)
                df = pd.DataFrame(data)
            else:
                # 嘗試用不同的分隔符加載
                for sep in [',', '\t', ' ', ';']:
                    try:
                        df = pd.read_csv(data_file, sep=sep)
                        # 檢查是否成功加載（至少有一行一列數據）
                        if df.shape[0] > 0 and df.shape[1] > 0:
                            break
                    except:
                        continue
        except Exception as e:
            raise ValueError(f"加載資料集 {dataset_name} 時出錯: {str(e)}")
            
        return {
            'name': dataset_name,
            'info': info,
            'df': df,
            'file_path': str(data_file)
        }
    
    def list_all_datasets(self):
        """
        列出所有資料集
        
        Returns:
            list: 所有資料集名稱列表
        """
        return list(self.dataset_info.keys())
    
    def get_dataset_info(self, dataset_name):
        """
        獲取指定資料集的信息
        
        Args:
            dataset_name: 資料集名稱
            
        Returns:
            dict: 資料集信息
        """
        if dataset_name not in self.dataset_info:
            raise ValueError(f"未找到資料集: {dataset_name}")
            
        return self.dataset_info[dataset_name]

# 使用示例
def example_usage():
    # 初始化資料集加載器
    loader = DatasetLoader('./datasets')
    
    # 獲取資料集類別統計
    categories = loader.get_dataset_categories()
    print("資料集類別統計:", categories)
    
    # 獲取小型二元分類數值資料集
    datasets = loader.get_datasets_by_category('small_datasets', 'binclass', 'numerical')
    print(f"找到 {len(datasets)} 個小型二元分類數值資料集")
    
    # 加載特定資料集
    if datasets:
        dataset = loader.load_dataset(datasets[0])
        print(f"已加載資料集: {dataset['name']}")
        print(f"資料集形狀: {dataset['df'].shape}")
        print(f"數據文件路徑: {dataset['file_path']}")