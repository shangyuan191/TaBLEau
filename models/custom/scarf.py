import matplotlib.pyplot as plt
import pandas as pd
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.optim import Adam
from torch.utils.data import DataLoader

from models.custom.scarf_lib.loss import NTXent
from models.custom.scarf_lib.model import SCARF
from models.custom.scarf_lib.dataset import SCARFDataset
from models.custom.scarf_lib.utils import get_device, dataset_embeddings, fix_seed, train_epoch

def start_fn(df, dataset_results, config):
    return df


def materialize_fn(df, dataset_results, config):
    """
    將資料集轉換為TensorFrame格式
    Args:
        df: 資料集DataFrame
        dataset_results: 資料集結果
        config: 實驗配置
    Returns:
        TensorFrame: 轉換後的TensorFrame
    """
    # 這裡可以添加將df轉換為TensorFrame的邏輯
    # 例如，使用PyTorch Geometric的DataLoader進行轉換
    print("Materializing dataset...")
    print(f"df: {df.shape}")
    # print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")

def encoding_fn(df, dataset_results, config):
    """
    將資料集編碼為TensorFrame格式
    Args:
        df: 資料集DataFrame
        dataset_results: 資料集結果
        config: 實驗配置
    Returns:
        TensorFrame: 編碼後的TensorFrame
    """
    # 這裡可以添加將df編碼為TensorFrame的邏輯
    # 例如，使用PyTorch Geometric的DataLoader進行編碼
    print("Encoding dataset...")
    print(f"df: {df.shape}")
    # print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")

def columnwise_fn(df, dataset_results, config):
    """
    將資料集按列處理
    Args:
        df: 資料集DataFrame
        dataset_results: 資料集結果
        config: 實驗配置
    Returns:
        TensorFrame: 列處理後的TensorFrame
    """
    # 這裡可以添加將df按列處理的邏輯
    # 例如，使用PyTorch Geometric的DataLoader進行處理
    print("Column-wise processing dataset...")
    print(f"df: {df.shape}")
    # print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")

def decoding_fn(df, dataset_results, config):
    """
    將資料集解碼為TensorFrame格式
    Args:
        df: 資料集DataFrame
        dataset_results: 資料集結果
        config: 實驗配置
    Returns:
        TensorFrame: 解碼後的TensorFrame
    """
    # 這裡可以添加將df解碼為TensorFrame的邏輯
    # 例如，使用PyTorch Geometric的DataLoader進行解碼
    print("Decoding dataset...")
    print(f"df: {df.shape}")
    # print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")
def main(df, dataset_results, config):
    """
    主函數，運行Scarf模型並返回結果
    Args:
        df: 資料集DataFrame
        dataset_results: 資料集結果
        config: 實驗配置
    Returns:
        dict: 實驗結果
    """
    print("Running Scarf model...")
    print(f"df: {df.shape}")
    # print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")
    df = start_fn(df, dataset_results, config)
    materialize_fn(df, dataset_results, config)
    encoding_fn(df, dataset_results, config)
    columnwise_fn(df, dataset_results, config)
    decoding_fn(df, dataset_results, config)
    



    # seed = 1234
    # fix_seed(seed)
    # data = datasets.load_breast_cancer(as_frame=True)
    # data, target = data["data"], data["target"]
    # train_data, test_data, train_target, test_target = train_test_split(
    #     data, target, test_size=0.2, stratify=target, random_state=seed
    # )

    # # preprocess
    # constant_cols = [c for c in train_data.columns if train_data[c].nunique() == 1]
    # train_data.drop(columns=constant_cols, inplace=True)
    # test_data.drop(columns=constant_cols, inplace=True)

    # scaler = StandardScaler()
    # train_data = pd.DataFrame(scaler.fit_transform(train_data), columns=train_data.columns)
    # test_data = pd.DataFrame(scaler.transform(test_data), columns=test_data.columns)

    # # to torch dataset
    # train_ds = SCARFDataset(train_data.to_numpy(), train_target.to_numpy(), columns=train_data.columns)
    # test_ds = SCARFDataset(test_data.to_numpy(), test_data.to_numpy(), columns=test_data.columns)

    # print(f"Train set: {train_ds.shape}")
    # print(f"Test set: {test_ds.shape}")
