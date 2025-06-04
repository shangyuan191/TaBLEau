# python main.py --dataset_size all --task_type all --feature_type all --models all --gnn_stages all
# python main.py --dataset kaggle_Audit_Data --models all --gnn_stages all
# python main.py --dataset kaggle_Audit_Data --models excelformer --gnn_stages none
# python main.py --dataset_size all --task_type all --feature_type all --models all --gnn_stages none
# python main.py --dataset_size all --task_type all --feature_type all --models excelformer --gnn_stages none
import os
import argparse
import logging
import time
import json
import pandas as pd
import numpy as np
import torch
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# 首先，在導入模型之前適配官方模型
from gnn_injection import adapt_official_models
patcher = adapt_official_models()

# 導入自定義模塊
from utils.data_utils import DatasetLoader
from model_runner import ModelRunner
from gnn_injection import GNNInjector, STAGE_TO_FUNCTION

# 配置日誌
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("experiment.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def parse_args():
    """解析命令行參數"""
    parser = argparse.ArgumentParser(description='GNN插入表格模型實驗')
    
    # 資料集類別相關參數
    parser.add_argument('--dataset_size', type=str, default='small_datasets',
                        choices=['small_datasets', 'large_datasets', 'all'],
                        help='資料集大小類別')
    parser.add_argument('--task_type', type=str, default='binclass',
                        choices=['binclass', 'multiclass', 'regression', 'all'],
                        help='任務類型')
    parser.add_argument('--feature_type', type=str, default='numerical',
                        choices=['numerical', 'categorical', 'balanced', 'all'],
                        help='特徵類型')
    parser.add_argument('--dataset', type=str, default=None,
                        help='指定單一資料集名稱 (可選)')
    parser.add_argument('--data_dir', type=str, default='./datasets',
                        help='數據目錄路徑')
    
    # 模型相關參數
    parser.add_argument('--models', nargs='+', 
                        default=['excelformer'],
                        help='要測試的模型列表 (用空格分隔多個模型名稱)')
    parser.add_argument('--gnn_stages', nargs='+', 
                        default=['none', 'materialization', 'encoding', 'columnwise', 'decoding'],
                        help='GNN插入階段列表 (用空格分隔多個階段名稱)')
    
    # 實驗相關參數
    parser.add_argument('--train_ratio', type=float, default=0.80,
                        help='訓練集比例')
    parser.add_argument('--val_ratio', type=float, default=0.15,
                        help='驗證集比例')
    parser.add_argument('--test_ratio', type=float, default=0.05,
                        help='測試集比例')
    parser.add_argument('--few_shot', action='store_true',
                        help='是否使用few-shot學習設置')
    parser.add_argument('--few_shot_ratio', type=float, default=0.05,
                        help='few-shot學習中使用的訓練數據比例')
    
    # GNN相關參數
    parser.add_argument('--gnn_hidden_dim', type=int, default=256,
                        help='GNN隱藏層維度')
    parser.add_argument('--gnn_layers', type=int, default=2,
                        help='GNN層數')
    parser.add_argument('--gnn_dropout', type=float, default=0.2,
                        help='GNN Dropout比率')
    
    # 訓練相關參數
    parser.add_argument('--epochs', type=int, default=1,
                        help='訓練輪數')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='批次大小')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='學習率')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                        help='權重衰減')
    parser.add_argument('--patience', type=int, default=10,
                        help='早停patience')
    
    # 其他參數
    parser.add_argument('--seed', type=int, default=42,
                        help='隨機種子')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU ID (-1表示使用CPU)')
    parser.add_argument('--output_dir', type=str, default='./results',
                        help='結果輸出目錄')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='實驗名稱 (默認: dataset_type_timestamp)')
    
    return parser.parse_args()

def set_seed(seed):
    """設置隨機種子以確保實驗可重複性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_experiment_name(args):
    """根據參數生成實驗名稱"""
    if args.exp_name:
        return args.exp_name
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    if args.dataset:
        prefix = f"{args.dataset}"
    else:
        # 創建類別前綴
        size_prefix = args.dataset_size if args.dataset_size != 'all' else 'all_sizes'
        task_prefix = args.task_type if args.task_type != 'all' else 'all_tasks'
        feature_prefix = args.feature_type if args.feature_type != 'all' else 'all_features'
        prefix = f"{size_prefix}_{task_prefix}_{feature_prefix}"
    
    if args.few_shot:
        prefix += f"_fewshot{args.few_shot_ratio}"
        
    return f"{prefix}_{timestamp}"

def expand_all_option(option_list, all_options):
    """
    展開選項列表中的'all'選項
    
    Args:
        option_list: 選項列表
        all_options: 'all'對應的所有選項列表
        
    Returns:
        展開後的選項列表
    """
    if 'all' in option_list:
        return all_options
    return option_list

def run_experiment(args):
    """
    運行實驗
    
    Args:
        args: 命令行參數
        
    Returns:
        dict: 實驗結果
    """
    # 設置隨機種子
    set_seed(args.seed)
    

    # 初始化資料集加載器
    loader = DatasetLoader(args.data_dir)
    
    # 獲取資料集類別統計
    categories = loader.get_dataset_categories()
    print("資料集類別統計:", categories)
    

    # datasets = loader.get_datasets_by_category(None, None, None)
    # print(f"找到 {len(datasets)} 個資料集")

    # # 印出每個資料集的名稱和類型資訊
    # for dataset_name in datasets:
    #     dataset_info = loader.get_dataset_info(dataset_name)  # 獲取資料集的詳細資訊
    #     print(f"資料集名稱: {dataset_name}")
    #     print(f"  大小: {dataset_info['size']}")
    #     print(f"  任務類型: {dataset_info['task_type']}")
    #     print(f"  特徵種類: {dataset_info['feature_type']}")
        
    #     # 加載資料集
    #     dataset = loader.load_dataset(dataset_name)
    #     print(f"  已加載資料集: {dataset['name']}")
    #     print(f"  資料集形狀: {dataset['df'].shape}")
    #     print(f"  數據文件路徑: {dataset['file_path']}")

        
    
    # 初始化模型運行器
    model_runner = ModelRunner("./models")
    # print(f"model_runner: {model_runner}")
    # 初始化GNN注入器
    gnn_injector = GNNInjector(model_runner)
    # print(f"gnn_injector.model_runner: {gnn_injector.model_runner}")
    # 獲取可用模型列表，保留模型類型信息
    available_models = []
    model_type_mapping = {}  # 用於存儲模型名稱到類型的映射
    for model_type, models in model_runner.available_models.items():
        for model_name in models.keys():
            available_models.append(model_name)
            model_type_mapping[model_name] = model_type

    # 確定要測試的模型列表
    models_to_test = expand_all_option(args.models, available_models)
    print(f"models_to_test: {models_to_test}")
    # 確定要測試的GNN階段列表
    valid_stages = ['none'] + list(STAGE_TO_FUNCTION.keys())
    gnn_stages_to_test = expand_all_option(args.gnn_stages, valid_stages)
    # 確定要測試的資料集列表
    if args.dataset:
        datasets_to_test = [args.dataset]
    else:
        datasets_to_test = loader.get_datasets_by_category(
            None if args.dataset_size == 'all' else args.dataset_size,
            None if args.task_type == 'all' else args.task_type,
            None if args.feature_type == 'all' else args.feature_type
        )
    
    logger.info(f"將測試 {len(datasets_to_test)} 個資料集, {len(models_to_test)} 個模型, {len(gnn_stages_to_test)} 個GNN階段")
    
    # 準備實驗配置
    experiment_config = {
        'train_val_test_split_ratio': [args.train_ratio, args.val_ratio, args.test_ratio],
        'few_shot': args.few_shot,
        'few_shot_ratio': args.few_shot_ratio,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'lr': args.lr,
        'weight_decay': args.weight_decay,
        'patience': args.patience,
        'device': torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"),
        'seed': args.seed
    }
    
    # 準備GNN配置
    gnn_config = {
        'hidden_dim': args.gnn_hidden_dim,
        'num_layers': args.gnn_layers,
        'dropout': args.gnn_dropout
    }
    
    # 運行所有實驗組合
    all_results = []
    
    for dataset_name in datasets_to_test:
        logger.info(f"處理資料集: {dataset_name}")
        
        # 加載資料集
        try:
            dataset_info = loader.load_dataset(dataset_name)
            df = dataset_info['df']
            print(df.shape)
            from sklearn.model_selection import train_test_split
            # 取得 y
            y = df['target'] if 'target' in df.columns else df.iloc[:, -1]
            # 取得任務類型
            task_type = loader.get_dataset_info(dataset_name).get('task_type', 'binclass')
            # 根據 few_shot 決定切分比例
            if experiment_config.get('few_shot', False):
                split_ratio = [experiment_config['few_shot_ratio'], experiment_config['val_ratio'], 1 - experiment_config['few_shot_ratio'] - experiment_config['val_ratio']]
            else:
                split_ratio = [experiment_config['train_val_test_split_ratio'][0], experiment_config['train_val_test_split_ratio'][1], experiment_config['train_val_test_split_ratio'][2]]
            # 先切 test
            stratify_y = y if 'class' in task_type or 'binclass' in task_type else None
            train_val_df, test_df, train_val_y, test_y = train_test_split(
                df, y, test_size=split_ratio[2], stratify=stratify_y, random_state=experiment_config['seed'])
            # 再切 val
            val_ratio = split_ratio[1] / (split_ratio[0] + split_ratio[1])
            stratify_train_val = train_val_y if 'class' in task_type or 'binclass' in task_type else None
            train_df, val_df, train_y, val_y = train_test_split(
                train_val_df, train_val_y, test_size=val_ratio, stratify=stratify_train_val, random_state=experiment_config['seed'])
        except Exception as e:
            logger.error(f"加載資料集 {dataset_name} 失敗: {str(e)}")
            continue
            
        dataset_results = {
            'dataset': dataset_name,
            'info': loader.get_dataset_info(dataset_name),
            'models': {}
        }
        # print(f"dataset_results: {dataset_results}")
        # print(f"train_df shape: {train_df.shape}, val_df shape: {val_df.shape}, test_df shape: {test_df.shape}")
        # print(f"train_y shape: {train_y.shape}, val_y shape: {val_y.shape}, test_y shape: {test_y.shape}")
        for model_name in models_to_test:
            logger.info(f"處理模型: {model_name}")
            
            model_results = {}
            model_type = model_type_mapping[model_name]  # 獲取模型類型
            if model_type == 'comparison':
                # 如果是 comparison 類型，直接運行模型，跳過 GNN 階段
                logger.info(f"模型 {model_name} 是 comparison 類型，跳過 GNN 階段")
                try:
                    # 傳入 train_df, val_df, test_df
                    result = model_runner.run_model(model_name, train_df, val_df, test_df, dataset_results, experiment_config, model_type)
                    model_results['none'] = result  # 將結果存儲在 'none' 階段
                except Exception as e:
                    logger.error(f"運行 {model_name} 模型時出錯: {str(e)}")
                    model_results['none'] = {'error': str(e)}
            else:
                # 如果不是 comparison 類型，測試所有 GNN 階段
                for gnn_stage in gnn_stages_to_test:
                    logger.info(f"測試 {model_name} 模型的 {gnn_stage} 階段")
                    print("\n")
                    # 注入 GNN
                    if gnn_stage != 'none':
                        gnn_injector.inject(model_name, gnn_stage, gnn_config)
                    
                    # 運行模型
                    try:
                        result = model_runner.run_model(model_name, train_df, val_df, test_df, dataset_results, experiment_config, model_type)
                        model_results[gnn_stage] = result
                    except Exception as e:
                        logger.error(f"運行 {model_name} 模型的 {gnn_stage} 階段時出錯: {str(e)}")
                        model_results[gnn_stage] = {'error': str(e)}
            
            dataset_results['models'][model_name] = model_results
        
        all_results.append(dataset_results)

    # 保存結果到文件
    import sys
    args = parse_args()
    # 取得命令列參數並處理
    cmd_args = sys.argv
    # 去掉 'python' 和 'main.py'
    filtered = [x for x in cmd_args if not x.endswith('main.py') and x != 'python']
    # 處理參數格式
    cmd_str = '_'.join(
        x.replace('--', '').replace('=', '_') for x in filtered
    )
    # 加上切分比例
    split_str = f"{args.train_ratio}_{args.val_ratio}_{args.test_ratio}"
    results_file_name = f"{cmd_str}_{split_str}.txt"
    print(f"results_file_name: {results_file_name}")
    # with open(f'./results/{results_file_name}', 'w') as f:
    #     for result in all_results:
    #         f.write(f"dataset: {result['dataset']}\n")
    #         for model_name, model_results in result['models'].items():
    #             f.write(f"  模型: {model_name}\n")
    #             for gnn_stage, res in model_results.items():
    #                 f.write(f"    GNN階段: {gnn_stage}\n")
    #                 if 'best_val_metric' in res:
    #                     f.write(f"          Best val metric: {res['best_val_metric']}\n")
    #                 if 'best_test_metric' in res:
    #                     f.write(f"          Best test metric: {res['best_test_metric']}\n")
    #                 if 'error' in res:
    #                     f.write(f"          錯誤: {res['error']}\n")
    
    # # 計算排名和分析實驗結果
    # analysis_results = analyze_results(all_results, dataset_loader)
    
    # # 準備完整結果
    # experiment_results = {
    #     'all_results': all_results,
    #     'analysis': analysis_results,
    #     'config': {
    #         'experiment_config': experiment_config,
    #         'gnn_config': gnn_config,
    #         'args': vars(args)
    #     }
    # }
    
    # return experiment_results
    return None

def analyze_results(all_results, dataset_loader):
    """
    分析實驗結果
    
    Args:
        all_results: 所有實驗結果
        dataset_loader: 資料集加載器
        
    Returns:
        dict: 分析結果
    """
    # 按照資料集類別組織結果
    categorized_results = {}
    
    for dataset_result in all_results:
        dataset_name = dataset_result['dataset']
        dataset_info = dataset_result['info']
        
        # 創建類別鍵
        category_key = f"{dataset_info['size']}_{dataset_info['task_type']}_{dataset_info['feature_type']}"
        
        if category_key not in categorized_results:
            categorized_results[category_key] = []
            
        categorized_results[category_key].append(dataset_result)
    
    # 分析每個類別
    analysis = {}
    
    for category, results in categorized_results.items():
        category_analysis = {}
        
        # 收集所有模型和階段的性能指標
        model_performances = {}
        
        for dataset_result in results:
            for model_name, model_results in dataset_result['models'].items():
                if model_name not in model_performances:
                    model_performances[model_name] = {}
                    
                for gnn_stage, result in model_results.items():
                    if 'error' in result or 'performance' not in result:
                        continue
                        
                    if gnn_stage not in model_performances[model_name]:
                        model_performances[model_name][gnn_stage] = []
                        
                    model_performances[model_name][gnn_stage].append({
                        'dataset': dataset_result['dataset'],
                        'performance': result['performance']
                    })
        
        # 計算平均性能
        avg_performances = {}
        
        for model_name, stage_performances in model_performances.items():
            avg_performances[model_name] = {}
            
            for gnn_stage, performances in stage_performances.items():
                if performances:
                    avg_performance = sum(p['performance'] for p in performances) / len(performances)
                    avg_performances[model_name][gnn_stage] = avg_performance
        
        # 找出每個模型的最佳階段
        model_best_stages = {}
        
        for model_name, stage_performances in avg_performances.items():
            if stage_performances:
                best_stage = max(stage_performances.items(), key=lambda x: x[1])
                model_best_stages[model_name] = {
                    'stage': best_stage[0],
                    'performance': best_stage[1]
                }
        
        # 模型排名
        model_ranking = sorted(model_best_stages.items(), key=lambda x: x[1]['performance'], reverse=True)
        
        # 分析GNN對每個模型的影響
        gnn_impact = {}
        
        for model_name, stage_performances in avg_performances.items():
            if 'none' in stage_performances and len(stage_performances) > 1:
                base_performance = stage_performances['none']
                
                for gnn_stage, performance in stage_performances.items():
                    if gnn_stage != 'none':
                        relative_improvement = (performance - base_performance) / max(abs(base_performance), 1e-10)
                        
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
        
        # GNN階段排名
        gnn_ranking = sorted(avg_gnn_impact.items(), key=lambda x: x[1], reverse=True)
        
        # 保存分析結果
        category_analysis['model_ranking'] = model_ranking
        category_analysis['gnn_ranking'] = gnn_ranking
        category_analysis['model_best_stages'] = model_best_stages
        category_analysis['avg_performances'] = avg_performances
        category_analysis['gnn_impact'] = gnn_impact
        
        analysis[category] = category_analysis
    
    return analysis

def generate_visualizations(results, output_dir):
    """
    生成實驗結果可視化
    
    Args:
        results: 實驗結果
        output_dir: 輸出目錄
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 獲取分析數據
    analysis = results.get('analysis', {})
    
    for category, category_analysis in analysis.items():
        # 1. 模型排名可視化
        model_ranking = category_analysis.get('model_ranking', [])
        
        if model_ranking:
            plt.figure(figsize=(12, 6))
            models = [m[0] for m in model_ranking]
            performances = [m[1]['performance'] for m in model_ranking]
            
            # 創建條形圖
            ax = sns.barplot(x=models, y=performances)
            plt.title(f'模型排名 - {category}')
            plt.xlabel('模型')
            plt.ylabel('性能')
            plt.xticks(rotation=45)
            
            # 添加性能值標籤
            for i, p in enumerate(performances):
                ax.text(i, p + 0.01, f'{p:.4f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'model_ranking_{category}.png')
            plt.close()
        
        # 2. GNN階段排名可視化
        gnn_ranking = category_analysis.get('gnn_ranking', [])
        
        if gnn_ranking:
            plt.figure(figsize=(10, 6))
            stages = [s[0] for s in gnn_ranking]
            improvements = [s[1] for s in gnn_ranking]
            
            # 創建條形圖
            ax = sns.barplot(x=stages, y=improvements)
            plt.title(f'GNN階段提升比較 - {category}')
            plt.xlabel('GNN插入階段')
            plt.ylabel('相對提升')
            
            # 添加提升值標籤
            for i, imp in enumerate(improvements):
                ax.text(i, imp + 0.01, f'{imp:.4f}', ha='center')
            
            plt.tight_layout()
            plt.savefig(output_dir / f'gnn_impact_{category}.png')
            plt.close()
        
        # 3. 熱力圖：模型+GNN階段性能
        avg_performances = category_analysis.get('avg_performances', {})
        
        if avg_performances:
            # 轉換為DataFrame
            models = list(avg_performances.keys())
            stages = set()
            for model_perfs in avg_performances.values():
                stages.update(model_perfs.keys())
            stages = sorted(list(stages))
            
            data = []
            for model in models:
                row = []
                model_perfs = avg_performances.get(model, {})
                for stage in stages:
                    row.append(model_perfs.get(stage, float('nan')))
                data.append(row)
            
            df = pd.DataFrame(data, index=models, columns=stages)
            
            # 創建熱力圖
            plt.figure(figsize=(12, 8))
            sns.heatmap(df, annot=True, fmt='.4f', cmap='viridis')
            plt.title(f'模型和GNN階段性能熱力圖 - {category}')
            plt.tight_layout()
            plt.savefig(output_dir / f'performance_heatmap_{category}.png')
            plt.close()

def save_results(results, exp_name, output_dir='./results'):
    """
    保存實驗結果
    
    Args:
        results: 實驗結果
        exp_name: 實驗名稱
        output_dir: 輸出目錄
    """
    # 創建輸出目錄
    output_dir = Path(output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 保存結果JSON
    results_file = output_dir / "results.json"
    
    # 準備可序列化的結果
    serializable_results = {}
    
    # 處理分析結果
    serializable_results['analysis'] = results.get('analysis', {})
    
    # 處理配置
    serializable_results['config'] = results.get('config', {})
    
    # 處理所有實驗結果
    serializable_all_results = []
    
    for dataset_result in results.get('all_results', []):
        serializable_dataset_result = {
            'dataset': dataset_result['dataset'],
            'info': dataset_result['info'],
            'models': {}
        }
        
        for model_name, model_results in dataset_result.get('models', {}).items():
            serializable_dataset_result['models'][model_name] = {}
            
            for gnn_stage, result in model_results.items():
                # 過濾掉不可序列化的字段
                serializable_result = {k: v for k, v in result.items() if k not in ['model', 'optimizer']}
                serializable_dataset_result['models'][model_name][gnn_stage] = serializable_result
        
        serializable_all_results.append(serializable_dataset_result)
    
    serializable_results['all_results'] = serializable_all_results
    
    with open(results_file, 'w') as f:
        json.dump(serializable_results, f, indent=4)
    
    # 生成可視化
    generate_visualizations(results, output_dir)
    
    logger.info(f"實驗結果已保存至 {output_dir}")

def main():
    """主函數"""
    # 解析命令行參數
    args = parse_args()
    
    # 生成實驗名稱
    exp_name = get_experiment_name(args)
    
    logger.info(f"開始實驗: {exp_name}")
    
    # 記錄設備信息
    device = f"cuda:{args.gpu}" if args.gpu >= 0 and torch.cuda.is_available() else "cpu"
    logger.info(f"使用設備: {device}")
    
    # 運行實驗
    start_time = time.time()
    results = run_experiment(args)
    # print(f"results:", results)
    elapsed_time = time.time() - start_time
    
    logger.info(f"實驗完成，總用時: {elapsed_time:.2f}秒")
    
    # 保存結果
    # save_results(results, exp_name, args.output_dir)

if __name__ == "__main__":
    main()

