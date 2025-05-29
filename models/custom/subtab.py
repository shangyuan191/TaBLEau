
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
    主函數，運行SubTab模型並返回結果
    Args:
        df: 資料集DataFrame
        dataset_results: 資料集結果
        config: 實驗配置
    Returns:
        dict: 實驗結果
    """
    # 這裡可以添加SubTab模型的運行邏輯
    # 例如，使用SubTab進行訓練和預測
    print("Running SubTab model...")
    print(f"df: {df.shape}")
    # print(f"dataset_results: {dataset_results}")
    print(f"config: {config}")
    df = start_fn(df, dataset_results, config)
    materialize_fn(df, dataset_results, config)
    encoding_fn(df, dataset_results, config)
    columnwise_fn(df, dataset_results, config)
    decoding_fn(df, dataset_results, config)