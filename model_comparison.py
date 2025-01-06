# import os
# import sys
# import csv
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# if __name__ == '__main__':
#     # Load data
#     dataset_sizes=['small','large']
#     task_types=['binclass','multiclass','regression']
#     feature_types=['numerical','categorical','balanced']
#     for dataset_size in dataset_sizes:
#         print(f"dataset_size: {dataset_size}\n\n\n")
#         for task_type in task_types:
#             print(f"task_type: {task_type}\n\n")
#             for feature_type in feature_types:
#                 print(f"feature_type: {feature_type}\n")
#                 for dataset_name in os.listdir(f'./datasets/{dataset_size}_datasets/{task_type}/{feature_type}'):
#                     dataset_path=f'./datasets/{dataset_size}_datasets/{task_type}/{feature_type}/{dataset_name}/{dataset_name}.csv'
#                     print(f"dataset_name: {dataset_name}")
#                     dataset=pd.read_csv(dataset_path)
#                     print(f"dataset shape: {dataset.shape}")


import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, root_mean_squared_error
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier, XGBRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm
import matplotlib.pyplot as plt

# Function to evaluate performance based on task type
def evaluate_model(task_type, y_true, y_pred):
    if task_type == "binclass":
        return roc_auc_score(y_true, y_pred)
    elif task_type == "multiclass":
        return accuracy_score(y_true, y_pred)
    elif task_type == "regression":
        return root_mean_squared_error(y_true, y_pred)
    else:
        raise ValueError("Unsupported task type.")

# Function to save detailed ranking results to file
def save_detailed_results(ranking_stats, output_file="detailed_ranking_results.txt"):
    with open(output_file, "w") as f:
        for split_ratio, dataset_data in ranking_stats.items():
            f.write(f"\nSplit Ratio: {split_ratio}\n")
            for dataset_size, task_data in dataset_data.items():
                for task_type, feature_data in task_data.items():
                    for feature_type, model_data in feature_data.items():
                        f.write(f"{dataset_size}+{task_type}+{feature_type}:\n")
                        for model, ranks in model_data.items():
                            avg_rank = np.mean(ranks) if ranks else float('nan')
                            f.write(f"  {model}: {avg_rank:.2f}\n")
    print(f"Detailed results saved to {output_file}")

# Function to generate visualization for each split ratio
def plot_rankings(ranking_stats, split_ratio, output_file=None):
    dataset_categories = []
    avg_ranks_per_model = {model: [] for model in ["KNN", "Decision Tree", "Random Forest", "XGBoost", "CatBoost"]}
    
    for dataset_size, task_data in ranking_stats[split_ratio].items():
        for task_type, feature_data in task_data.items():
            for feature_type, model_data in feature_data.items():
                category = f"{dataset_size}+{task_type}+{feature_type}"
                dataset_categories.append(category)
                
                for model in avg_ranks_per_model.keys():
                    avg_rank = np.mean(model_data[model]) if model_data[model] else float('nan')
                    avg_ranks_per_model[model].append(avg_rank)

    # Plot settings
    plt.figure(figsize=(15, 8))
    for model, avg_ranks in avg_ranks_per_model.items():
        plt.plot(dataset_categories, avg_ranks, marker='o', label=model)
    
    plt.title(f"Average Rankings Across Dataset Types (Split Ratio: {split_ratio})")
    plt.xlabel("Dataset Categories")
    plt.ylabel("Average Rank")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.grid(True)

    if output_file:
        output_path = f"{output_file}_split_{split_ratio}.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    plt.show()
def get_model(model_name,task_type):
    models = {
        "KNN": KNeighborsClassifier() if task_type != "regression" else KNeighborsRegressor(),
        "Decision Tree": DecisionTreeClassifier() if task_type != "regression" else DecisionTreeRegressor(),
        "Random Forest": RandomForestClassifier() if task_type != "regression" else RandomForestRegressor(),
        "XGBoost": XGBClassifier(eval_metric="logloss") if task_type != "regression" else XGBRegressor(),
        "CatBoost": CatBoostClassifier(verbose=0) if task_type != "regression" else CatBoostRegressor(verbose=0),
    }
    return models[model_name]
if __name__ == "__main__":
    dataset_sizes = ['small', 'large']
    task_types = ['binclass', 'multiclass', 'regression']
    feature_types = ['numerical', 'categorical', 'balanced']
    split_ratios = [(0.8, 0.2), (0.05, 0.95)]
    
    ranking_stats = {f"{train_ratio}:{test_ratio}": {} for train_ratio, test_ratio in split_ratios}
    # models = {
    #         "KNN": KNeighborsClassifier() if task_type != "regression" else KNeighborsRegressor(),
    #         "Decision Tree": DecisionTreeClassifier() if task_type != "regression" else DecisionTreeRegressor(),
    #         "Random Forest": RandomForestClassifier() if task_type != "regression" else RandomForestRegressor(),
    #         "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric="logloss") if task_type != "regression" else XGBRegressor(),
    #         "CatBoost": CatBoostClassifier(verbose=0) if task_type != "regression" else CatBoostRegressor(verbose=0),
    #     }
    models=["KNN","Decision Tree","Random Forest","XGBoost","CatBoost"]

    for split_ratio in split_ratios:
        train_ratio, test_ratio = split_ratio
        split_key = f"{train_ratio}:{test_ratio}"

        for dataset_size in tqdm(dataset_sizes, desc="Processing dataset sizes"):
            ranking_stats[split_key][dataset_size] = {}
            for task_type in tqdm(task_types, desc=f"Task types ({dataset_size})"):
                ranking_stats[split_key][dataset_size][task_type] = {}
                for feature_type in tqdm(feature_types, desc=f"Feature types ({task_type})"):
                    ranking_stats[split_key][dataset_size][task_type][feature_type] = {
                        model: [] for model in models
                    }

                    dataset_dir = f'./datasets/{dataset_size}_datasets/{task_type}/{feature_type}'
                    if not os.path.exists(dataset_dir):
                        continue

                    # Add tqdm to iterate over datasets
                    for dataset_name in tqdm(os.listdir(dataset_dir), desc=f"Datasets ({feature_type})"):
                        dataset_path = f'{dataset_dir}/{dataset_name}/{dataset_name}.csv'
                        if not os.path.exists(dataset_path):
                            continue
                        
                        data = pd.read_csv(dataset_path)
                        X = data.iloc[:, :-1].values
                        y = data.iloc[:, -1].values
                        
                        # Train-test split
                        X_train, X_test, y_train, y_test = train_test_split(
                            X, y, train_size=train_ratio, test_size=test_ratio, random_state=42
                        )

                        # Model evaluation
                        model_scores = {}
                        for model_name in models:
                            model=get_model(model_name,task_type)
                            try:
                                model.fit(X_train, y_train)
                                y_pred = model.predict(X_test)
                                score = evaluate_model(task_type, y_test, y_pred)
                                model_scores[model_name] = score
                            except Exception as e:
                                print(f"Error with {model_name} on {dataset_name}: {e}")
                                model_scores[model_name] = float('nan')
                        
                        # Ranking
                        sorted_models = sorted(model_scores, key=model_scores.get, reverse=(task_type != "regression"))
                        for rank, model_name in enumerate(sorted_models, 1):
                            ranking_stats[split_key][dataset_size][task_type][feature_type][model_name].append(rank)

    # Save detailed results
    save_detailed_results(ranking_stats, output_file="detailed_ranking_results.txt")

    # Generate plots
    for split_ratio in ranking_stats.keys():
        plot_rankings(ranking_stats, split_ratio, output_file="ranking_plot")