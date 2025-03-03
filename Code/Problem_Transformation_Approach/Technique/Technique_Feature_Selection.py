import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, recall_score, hamming_loss, jaccard_score,
                             multilabel_confusion_matrix, f1_score, precision_score, accuracy_score,
                             zero_one_loss)
from xgboost import XGBClassifier
from skmultilearn.problem_transform import LabelPowerset
from functools import reduce
import time

# Store execution times for each value of k
execution_data = []
# Load dataset
file_path = 'Technique_Dataset.csv'
df = pd.read_csv(file_path)

# Drop the column 'Lateral Movement'
df = df.drop(columns=['T1458','T1660','T1456','T1631','T1664','T1663','T1461','T1661','T1639','T1641','T1474','T1603','T1638'])
# Separate features and labels
X = df.iloc[:, 1:-48]
y = df.iloc[:, -48:]

# Get column names for X
column_names = df.columns[1:-48].tolist()
label_names = y.columns.tolist()

# Define different random seeds to evaluate
random_seeds = [42, 1433, 2396, 451, 995, 98, 262, 354, 560, 1600]

# List of k values
k_values = list(range(13100, 13400, 100))

# Function to evaluate the model
def evaluate_model(X_train, X_test, y_train, y_test, selected_features, model_name, file, random_seed, k):
    X_train_selected = X_train.iloc[:, selected_features]
    X_test_selected = X_test.iloc[:, selected_features]
    
    lb_model = LabelPowerset(XGBClassifier(gamma=0,learning_rate=0.3,max_depth=6,n_estimators=100,random_state=42))
    lb_model.fit(X_train_selected, y_train)
    lb_predictions = lb_model.predict(X_test_selected)
    
    file.write(f"\n\n********** K: {k} **********\n\n")
    file.write(f"\n\n********** Random Seed: {random_seed} **********\n\n")
    file.write(f"\nEvaluating {model_name}")
    file.write(f"\nAccuracy: {accuracy_score(y_test, lb_predictions)}")
    file.write(f"\nMacro F1 Score: {f1_score(y_test, lb_predictions, average='macro')}")
    file.write(f"\nWeighted F1 score: {f1_score(y_test, lb_predictions, average='weighted')}")
    file.write(f"\nMicro F1 score: {f1_score(y_test, lb_predictions, average='micro')}")
    file.write(f"\nMacro Precision: {precision_score(y_test, lb_predictions, average='macro')}")
    file.write(f"\nWeighted Precision: {precision_score(y_test, lb_predictions, average='weighted')}")
    file.write(f"\nMicro Precision: {precision_score(y_test, lb_predictions, average='micro')}")
    file.write(f"\nMacro Recall: {recall_score(y_test, lb_predictions, average='macro')}")
    file.write(f"\nWeighted Recall: {recall_score(y_test, lb_predictions, average='weighted')}")
    file.write(f"\nMicro Recall: {recall_score(y_test, lb_predictions, average='micro')}")
    file.write(f"\nHamming Loss: {hamming_loss(y_test, lb_predictions)}")
    file.write(f"\nZero One Loss: {zero_one_loss(y_test, lb_predictions)}")
    file.write(f"\nJaccard Similarity: {jaccard_score(y_test, lb_predictions, average='samples')}")
    file.write(f"\nClassification Report:\n{classification_report(y_test, lb_predictions, target_names=label_names)}")
    file.write(f"\nMultilabel Confusion Matrix:\n{multilabel_confusion_matrix(y_test, lb_predictions)}")
    file.write("\n")

    print(f"\n\n********** K: {k} **********\n\n")
    print(f"\n\n********** Random Seed: {random_seed} **********\n\n")
    print(f"Evaluating {model_name}")
    print(f"Accuracy: {accuracy_score(y_test, lb_predictions)}")
    print(f"Macro F1 Score: {f1_score(y_test, lb_predictions, average='macro')}")
    print(f"Weighted F1 score: {f1_score(y_test, lb_predictions, average='weighted')}")
    print(f"Micro F1 score: {f1_score(y_test, lb_predictions, average='micro')}")
    print(f"Macro Precision: {precision_score(y_test, lb_predictions, average='macro')}")
    print(f"Weighted Precision: {precision_score(y_test, lb_predictions, average='weighted')}")
    print(f"Micro Precision: {precision_score(y_test, lb_predictions, average='micro')}")
    print(f"Macro Recall: {recall_score(y_test, lb_predictions, average='macro')}")
    print(f"Weighted Recall: {recall_score(y_test, lb_predictions, average='weighted')}")
    print(f"Micro Recall: {recall_score(y_test, lb_predictions, average='micro')}")
    print(f"Hamming Loss: {hamming_loss(y_test, lb_predictions)}")
    print(f"Zero One Loss: {zero_one_loss(y_test, lb_predictions)}")
    print(f"Jaccard Similarity: {jaccard_score(y_test, lb_predictions, average='samples')}")
    print(f"Classification Report:\n{classification_report(y_test, lb_predictions, target_names=label_names)}")
    print(f"Multilabel Confusion Matrix:\n{multilabel_confusion_matrix(y_test, lb_predictions)}")
    print("\n")

# Store combined features across all random seeds
all_combined_features_set = set()

# Loop through each value of k
for k in k_values:
    start_time = time.time()  # Start measuring execution time
    selected_features_indices_per_seed = {seed: [] for seed in random_seeds}
    
    # Loop through each random seed
    for seed in random_seeds:
        selected_features_indices = []
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
        
        print(f"\nRandom Seed: {seed}\n")
        with open(f"top_k_features_k_{k}.txt", "a", encoding='utf-8') as seed_file:
            seed_file.write(f"\nRandom Seed: {seed}\n")
            combined_features_set = set()
            for i in range(y.shape[1]):
                selector = SelectKBest(score_func=chi2, k=k)
                selector.fit(X_train, y_train.iloc[:, i])
                selected_features_indices.append(selector.get_support(indices=True))
                
                selected_features = [column_names[j] for j in selector.get_support(indices=True)]
                combined_features_set.update(selected_features)
                print(f"Label {i+1} Selected Features: {selected_features}")
                seed_file.write(f"Label {i + 1} Selected Features:\n")
                seed_file.write(", ".join(selected_features))
                seed_file.write("\n")
            
            # Print and write the combined features for this seed
            combined_features = list(combined_features_set)
            num_combined_features = len(combined_features)
            print(f"\nNumber of Combined Features for Seed {seed}: {num_combined_features}")
            print(f"\nCombined Features for Seed {seed}\n: {combined_features}")
            seed_file.write(f"\nNumber of Combined Features for Seed {seed}: {num_combined_features}\n")
            seed_file.write(f"\nCombined Features:\n")
            seed_file.write(", ".join(combined_features))
            seed_file.write("\n")
        
        selected_features_indices_per_seed[seed] = selected_features_indices
        all_combined_features_set.update(combined_features)
    
    # Compute the union of top k features across all random seeds
    union_selected_features_per_label = {label: reduce(np.union1d, [selected_features_indices_per_seed[seed][label] for seed in random_seeds]) for label in range(y.shape[1])}
    combined_features_all_seeds_set = set()
    
    for label, selected_indices in union_selected_features_per_label.items():
        combined_features_all_seeds_set.update(selected_indices)
    
    final_selected_features = list(combined_features_all_seeds_set)
    selected_features_names = [column_names[i] for i in final_selected_features]
    
    print(f"\n\nUnion of top k features across all random seeds for k={k}:")
    for label, selected_indices in union_selected_features_per_label.items():
        selected_features = [column_names[i] for i in selected_indices]
        print(f"Label {label + 1} Combined Selected Features: {selected_features}")

    # Print the number of combined features
    print(f"Number of combined features: {len(final_selected_features)}")   
    print(f"\nFinal Selected Features across all random seeds: {selected_features_names}")

    # Open a file to write evaluations for the current k value
    with open(f"Tactic_KBest_k_{k}_XGBoost_model_evaluations.txt", "w", encoding='utf-8') as file:
        for seed in random_seeds:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
            evaluate_model(X_train, X_test, y_train, y_test, final_selected_features, "Label Powerset", file, seed, k)
    
    print(f"Evaluations and combined features saved for k={k}")

    # Save all combined features from all random seeds to a file
    all_combined_features = list(all_combined_features_set)
    with open(f"top_k_features_k_{k}.txt", "a", encoding='utf-8') as file:
        num_all_combined_features = len(all_combined_features)
        file.write(f"\nNumber of Combined Features: {num_all_combined_features}\n")
        file.write("Combined Features from All Random Seeds:\n")
        file.write(", ".join(all_combined_features))

    print(f"\nAll combined features from all random seeds saved in all_combined_features_all_random_seeds.txt")
    end_time = time.time()  # End measuring execution time
    execution_time = end_time - start_time
    # Append execution time and number of combined features to execution_data
    execution_data.append({"k": k, "execution_time": execution_time, "num_selected_features": len(final_selected_features)})

# Convert execution_data to DataFrame
execution_df = pd.DataFrame(execution_data)

# Save DataFrame to CSV file
execution_df.to_csv("Feature_Selection_execution_data.csv", index=False)