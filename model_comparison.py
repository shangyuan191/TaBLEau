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
from sklearn.svm import LinearSVC, LinearSVR  # Import SVM models
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



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
    avg_ranks_per_model = {model: [] for model in ["Linear Model", "KNN", "Decision Tree", "Random Forest", "XGBoost", "CatBoost", "SVM", "DNN","CatDNN"]}
    
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
        output_path = f"{output_file}_split_{split_ratio}_small_only.png"
        plt.savefig(output_path)
        print(f"Plot saved to {output_path}")
    plt.show()

# Function to get model based on task type
def get_model(model_name, task_type, split_ratio,input_dim,X_train=None,y_train=None,X_test=None,y_test=None):
    # Check if it's a few-shot scenario (split ratio is 0.05:0.95)
    if split_ratio == (0.05, 0.95):
        knn_neighbors = 3
    else:
        knn_neighbors = 5  # Default KNN neighbors

    if model_name == "Linear Model":
        if task_type != "regression":
            return LogisticRegression()
        else:
            return LinearRegression()
    elif model_name == "DNN":
        # 建立神經網絡模型
        if task_type=="binclass":
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')  # 二分類任務使用sigmoid
            ])
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
            return model
        elif task_type=="multiclass":
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # 多分類任務使用softmax
            ])
            model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        elif task_type=="regression":
            model = Sequential([
                Dense(64, activation='relu', input_dim=input_dim),
                Dense(32, activation='relu'),
                Dense(1)  # 回歸任務不使用激活函數
            ])
            model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
        return model
    
    elif model_name == "CatDNN":
        if task_type=="binclass":
            catboost_model = CatBoostClassifier(verbose=0)
            catboost_model.fit(X_train, y_train)
            catboost_preds_train = catboost_model.predict_proba(X_train)[:, 1].reshape(-1, 1)
            catboost_preds_test = catboost_model.predict_proba(X_test)[:, 1].reshape(-1, 1)
            # Concatenate CatBoost predictions with original features
            X_train_combined = np.hstack([X_train, catboost_preds_train])
            X_test_combined = np.hstack([X_test, catboost_preds_test])

            # Build the DNN model
            dnn_model = Sequential([
                Dense(64, activation='relu', input_dim=X_train_combined.shape[1]),
                Dense(32, activation='relu'),
                Dense(1, activation='sigmoid')  # Binary classification uses sigmoid
            ])
            dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['AUC'])
            # Train DNN on the combined features
            dnn_model.fit(X_train_combined, y_train, epochs=10, batch_size=32)
            loss, auc = dnn_model.evaluate(X_test_combined, y_test)
            return auc
        elif task_type=="multiclass":
            catboost_model = CatBoostClassifier(verbose=0)
            catboost_model.fit(X_train, y_train)
            catboost_preds_train = catboost_model.predict_proba(X_train)
            catboost_preds_test = catboost_model.predict_proba(X_test)
            # Concatenate CatBoost predictions with original features
            X_train_combined = np.hstack([X_train, catboost_preds_train])
            X_test_combined = np.hstack([X_test, catboost_preds_test])

            # Build the DNN model
            dnn_model = Sequential([
                Dense(64, activation='relu', input_dim=X_train_combined.shape[1]),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # Multi-class classification uses softmax
            ])
            dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            dnn_model.fit(X_train_combined, y_train, epochs=10, batch_size=32)
            loss, accuracy = dnn_model.evaluate(X_test_combined, y_test)
            return accuracy
        elif task_type=="regression":
            catboost_model = CatBoostRegressor(verbose=0)
            catboost_model.fit(X_train, y_train)
            catboost_preds_train = catboost_model.predict(X_train).reshape(-1, 1)
            catboost_preds_test = catboost_model.predict(X_test).reshape(-1, 1)
            # Concatenate CatBoost predictions with original features
            X_train_combined = np.hstack([X_train, catboost_preds_train])
            X_test_combined = np.hstack([X_test, catboost_preds_test])

            # Build the DNN model
            dnn_model = Sequential([
                Dense(64, activation='relu', input_dim=X_train_combined.shape[1]),
                Dense(32, activation='relu'),
                Dense(1)  # Regression task does not use activation function
            ])
            dnn_model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])
            dnn_model.fit(X_train_combined, y_train, epochs=10, batch_size=32)
            y_pred = dnn_model.predict(X_test_combined)
            return root_mean_squared_error(y_test, y_pred)
        elif task_type=="multiclass":
            catboost_model = CatBoostClassifier(verbose=0)
            catboost_model.fit(X_train, y_train)
            catboost_preds_train = catboost_model.predict_proba(X_train)
            catboost_preds_test = catboost_model.predict_proba(X_test)
            # Concatenate CatBoost predictions with original features
            X_train_combined = np.hstack([X_train, catboost_preds_train])
            X_test_combined = np.hstack([X_test, catboost_preds_test])

            # Build the DNN model
            dnn_model = Sequential([
                Dense(64, activation='relu', input_dim=X_train_combined.shape[1]),
                Dense(32, activation='relu'),
                Dense(3, activation='softmax')  # Multi-class classification uses softmax
            ])
            dnn_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return dnn_model
    else:
        models = {
            "KNN": KNeighborsClassifier(n_neighbors=knn_neighbors) if task_type != "regression" else KNeighborsRegressor(n_neighbors=knn_neighbors),
            "Decision Tree": DecisionTreeClassifier() if task_type != "regression" else DecisionTreeRegressor(),
            "Random Forest": RandomForestClassifier() if task_type != "regression" else RandomForestRegressor(),
            "XGBoost": XGBClassifier(eval_metric="logloss") if task_type != "regression" else XGBRegressor(),
            "CatBoost": CatBoostClassifier(verbose=0) if task_type != "regression" else CatBoostRegressor(verbose=0),
            "SVM": LinearSVC() if task_type != "regression" else LinearSVR()
        }
        return models[model_name]
if __name__ == "__main__":
    dataset_sizes = ['small']
    task_types = ['binclass', 'multiclass', 'regression']
    feature_types = ['numerical', 'categorical', 'balanced']
    split_ratios = [(0.8, 0.2), (0.05, 0.95)]
    
    ranking_stats = {f"{train_ratio}:{test_ratio}": {} for train_ratio, test_ratio in split_ratios}
    models = ["Linear Model", "KNN", "Decision Tree", "Random Forest", "XGBoost", "CatBoost", "SVM", "DNN","CatDNN"]

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
                        print(f"Processing {dataset_name} ({dataset_size}+{task_type}+{feature_type})")
                        

                        dataset_path = f'{dataset_dir}/{dataset_name}/{dataset_name}.csv'
                        if not os.path.exists(dataset_path):
                            continue
                        
                        data = pd.read_csv(dataset_path)
                        X = data.loc[:, data.columns != 'target'].values
                        y = data.loc[:, 'target'].values


                        # Perform stratified split for few-shot scenario (0.05:0.95)
                        if  task_type != "regression":
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, train_size=train_ratio, test_size=test_ratio, random_state=42, stratify=y
                            )
                        else:
                            X_train, X_test, y_train, y_test = train_test_split(
                                X, y, train_size=train_ratio, test_size=test_ratio, random_state=42
                            )
                        input_dim = X_train.shape[1]
                        # Model evaluation
                        model_scores = {}
                        for model_name in models:
                            if model_name=="CatDNN":
                                model_scores[model_name] = get_model(model_name, task_type, split_ratio,input_dim,X_train,y_train,X_test,y_test)
                                continue
                            model = get_model(model_name, task_type, split_ratio,input_dim)
                            if model_name == "DNN":
                                model=get_model(model_name, task_type, split_ratio,input_dim)
                                model.fit(X_train, y_train, epochs=10, batch_size=32)
                                if task_type =="binclass":
                                    loss, auc = model.evaluate(X_test, y_test)
                                    model_scores[model_name] = auc
                                elif task_type =="multiclass":
                                    loss, accuracy = model.evaluate(X_test, y_test)
                                    model_scores[model_name] = accuracy
                                elif task_type =="regression":
                                    y_pred=model.predict(X_test)
                                    rmse=root_mean_squared_error(y_test, y_pred)
                                    model_scores[model_name] = rmse
                            else:
                                try:
                                    model.fit(X_train, y_train)
                                    if task_type =="binclass" and hasattr(model, "predict_proba"):
                                        y_pred = model.predict_proba(X_test)[:, 1]
                                    else:
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
    save_detailed_results(ranking_stats, output_file="detailed_ranking_results_small_only.txt")

    # Generate plots
    for split_ratio in ranking_stats.keys():
        plot_rankings(ranking_stats, split_ratio, output_file="ranking_plot")



