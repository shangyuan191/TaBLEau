#!/usr/bin/env python3
"""
拆分實驗結果檔案的腳本
將包含多個模型的實驗結果拆分成每個模型獨立的檔案
"""

import os
import re
from pathlib import Path


def split_results_file(input_file: str, output_dir: str = None):
    """
    拆分實驗結果檔案
    
    Args:
        input_file: 輸入檔案路徑
        output_dir: 輸出目錄（預設為輸入檔案所在目錄）
    """
    # 讀取輸入檔案
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 解析檔案名稱
    input_filename = Path(input_file).stem
    if output_dir is None:
        output_dir = Path(input_file).parent
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # 從檔案名稱中提取模型列表
    # 例如：dataset_size_all_task_type_all_feature_type_all_models_tabpfn_xgboost_catboost_lightgbm_gnn_stages_none_0.05_0.15_0.8.txt
    match = re.search(r'models_([\w_]+)_gnn_stages', input_filename)
    if not match:
        print(f"❌ 無法從檔案名稱中解析模型列表: {input_filename}")
        return
    
    models_str = match.group(1)
    models = models_str.split('_')
    print(f"📝 檢測到模型: {models}")
    
    # 為每個模型建立資料結構
    model_data = {model: [] for model in models}
    current_dataset = None
    current_model = None
    dataset_block = []
    
    # 逐行解析檔案
    lines = content.split('\n')
    
    for line in lines:
        # 檢測 dataset 行
        if line.startswith('dataset: '):
            # 如果有先前的 dataset block，先儲存
            if current_dataset and current_model and dataset_block:
                model_data[current_model].append('\n'.join(dataset_block))
            
            # 開始新的 dataset
            current_dataset = line
            dataset_block = []
            current_model = None
        
        # 檢測模型行
        elif line.strip().startswith('模型: '):
            # 儲存前一個模型的資料
            if current_model and dataset_block:
                model_data[current_model].append('\n'.join(dataset_block))
            
            # 提取模型名稱
            model_match = re.search(r'模型: (\w+)', line)
            if model_match:
                current_model = model_match.group(1)
                dataset_block = [current_dataset, line]
        
        # 繼續累積當前模型的資料
        elif current_model and line.strip():
            dataset_block.append(line)
    
    # 處理最後一個 dataset block
    if current_dataset and current_model and dataset_block:
        model_data[current_model].append('\n'.join(dataset_block))
    
    # 為每個模型寫出檔案
    base_filename = input_filename.replace(f'models_{models_str}', 'models_{}')
    
    for model in models:
        if not model_data[model]:
            print(f"⚠️  模型 {model} 沒有資料")
            continue
        
        output_filename = base_filename.format(model)
        output_path = output_dir / f"{output_filename}.txt"
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(model_data[model]))
        
        dataset_count = len(model_data[model])
        print(f"✅ 已建立 {output_path.name} (包含 {dataset_count} 個 datasets)")
    
    print(f"\n🎉 拆分完成！共拆分為 {len(models)} 個檔案")


def main():
    """主函數"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='拆分實驗結果檔案為每個模型獨立的檔案',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
範例:
  # 拆分單一檔案
  python split_results.py summary_results/dataset_size_all_task_type_all_feature_type_all_models_tabpfn_xgboost_catboost_lightgbm_gnn_stages_none_0.05_0.15_0.8.txt
  
  # 指定輸出目錄
  python split_results.py input.txt -o output_dir/
  
  # 處理目錄中所有符合模式的檔案
  python split_results.py summary_results/ --pattern "*_models_*"
        """
    )
    
    parser.add_argument(
        'input',
        help='輸入檔案路徑或目錄'
    )
    parser.add_argument(
        '-o', '--output-dir',
        help='輸出目錄（預設為輸入檔案所在目錄）'
    )
    parser.add_argument(
        '--pattern',
        default='*_models_*',
        help='當輸入為目錄時，用於匹配檔案的模式（預設: *_models_*）'
    )
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if input_path.is_file():
        # 處理單一檔案
        print(f"📂 處理檔案: {input_path}")
        split_results_file(str(input_path), args.output_dir)
    
    elif input_path.is_dir():
        # 處理目錄中的所有符合模式的檔案
        files = list(input_path.glob(args.pattern))
        if not files:
            print(f"❌ 在目錄 {input_path} 中找不到符合模式 '{args.pattern}' 的檔案")
            return
        
        print(f"📂 找到 {len(files)} 個檔案")
        for file in files:
            print(f"\n{'='*60}")
            print(f"處理: {file.name}")
            print('='*60)
            split_results_file(str(file), args.output_dir)
    
    else:
        print(f"❌ 路徑不存在: {input_path}")


if __name__ == '__main__':
    main()
