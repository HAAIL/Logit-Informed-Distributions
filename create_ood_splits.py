import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

def get_dataset_info(dataset_name):
    """Get dataset type and target information"""
    dict_df = pd.read_csv(f'Data/{dataset_name}/{dataset_name}_dict.csv')
    df = pd.read_csv(f'Data/{dataset_name}/{dataset_name}_prep.csv')

    target_cols = dict_df[dict_df['target'] == 'T']['name'].tolist()
    feature_cols = dict_df[dict_df['target'] == 'F']['name'].tolist()

    # Determine dataset type
    if len(target_cols) > 1:
        # Multi-class one-hot encoded
        n_classes = len(target_cols)
        if n_classes == 2:
            dataset_type = 'binary classification'
        else:
            dataset_type = 'multiclass classification'
        # Convert one-hot to single column
        y = df[target_cols].idxmax(axis=1)
    else:
        y = df[target_cols[0]]
        n_unique = y.nunique()
        if n_unique == 2:
            dataset_type = 'binary classification'
        elif n_unique <= 20:
            dataset_type = 'multiclass classification'
        else:
            dataset_type = 'regression'

    return feature_cols, target_cols, y, dataset_type

def evaluate_model(X_train, y_train, X_test, y_test, dataset_type):
    """Train model and evaluate performance"""
    if dataset_type == 'regression':
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        performance = mean_squared_error(y_test, y_pred)
    elif dataset_type == 'binary classification':
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        performance = roc_auc_score(y_test, y_pred_proba)
    else:  # multiclass
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)
        # Binarize the labels for OVR AUC
        classes = model.classes_
        y_test_bin = label_binarize(y_test, classes=classes)
        if y_test_bin.shape[1] == 1:
            # Binary case
            performance = roc_auc_score(y_test, y_pred_proba[:, 1])
        else:
            performance = roc_auc_score(y_test_bin, y_pred_proba, average='macro')

    return performance

def create_ood_split_numerical(X_train_iid, y_train_iid, feature, strategy):
    """Create OOD split for numerical feature"""
    # Sort by feature
    sorted_indices = X_train_iid[feature].argsort()
    n = len(sorted_indices)

    if strategy == 'half':
        # Keep bottom 50%
        keep_indices = sorted_indices[:n//2]
        split_info = f"keep_lower_50%"
    else:  # '75%'
        # Keep bottom 75%
        keep_indices = sorted_indices[:int(n*0.75)]
        split_info = f"keep_lower_75%"

    X_train_ood = X_train_iid.iloc[keep_indices].reset_index(drop=True)
    y_train_ood = y_train_iid.iloc[keep_indices].reset_index(drop=True)
    return X_train_ood, y_train_ood, split_info

def create_ood_split_categorical(X_train_iid, y_train_iid, feature):
    """Create OOD split for categorical feature - exclude each unique value"""
    unique_values = X_train_iid[feature].unique()

    ood_splits = []
    for value in unique_values:
        # Exclude this value
        mask = X_train_iid[feature] != value
        X_train_ood = X_train_iid[mask].reset_index(drop=True)
        y_train_ood = y_train_iid[mask].reset_index(drop=True)
        split_info = f"exclude_{feature}={value}"
        ood_splits.append((X_train_ood, y_train_ood, split_info))

    return ood_splits

def process_dataset(dataset_name):
    """Process a single dataset to create OOD splits"""
    print(f"\nProcessing {dataset_name}...")

    # Load data
    df = pd.read_csv(f'Data/{dataset_name}/{dataset_name}_prep.csv')
    feature_cols, target_cols, y, dataset_type = get_dataset_info(dataset_name)

    X = df[feature_cols]

    # Create IID train/test split (70-30)
    X_train_iid, X_test, y_train_iid, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y if dataset_type != 'regression' else None
    )

    # Evaluate IID performance
    P_IID = evaluate_model(X_train_iid, y_train_iid, X_test, y_test, dataset_type)
    print(f"  IID Performance: {P_IID:.4f}")

    # Try different OOD strategies
    # For regression: want highest MSE (worst), start at -inf
    # For classification: want lowest AUC (worst), start at inf
    best_ood_perf = float('-inf') if dataset_type == 'regression' else float('inf')
    best_ood_train = None
    best_feature = None
    best_split_info = None

    for feature in feature_cols:
        # Check if numerical or categorical
        n_unique = X_train_iid[feature].nunique()

        if n_unique > 10:  # Numerical
            # Try both strategies: 50% and 75%
            for strategy in ['half', '75%']:
                X_train_ood, y_train_ood, split_info = create_ood_split_numerical(
                    X_train_iid, y_train_iid, feature, strategy)

                # Skip if too few samples
                if len(X_train_ood) < 10:
                    continue

                # Evaluate OOD performance
                try:
                    P_OOD = evaluate_model(X_train_ood, y_train_ood, X_test, y_test, dataset_type)

                    # Check if this is worse (higher MSE or lower AUC)
                    if dataset_type == 'regression':
                        is_worse = P_OOD > best_ood_perf
                    else:
                        is_worse = P_OOD < best_ood_perf

                    if is_worse:
                        best_ood_perf = P_OOD
                        best_ood_train = X_train_ood.copy()
                        best_y_train = y_train_ood.copy()
                        best_feature = feature
                        best_split_info = split_info
                except Exception as e:
                    continue

        else:  # Categorical
            ood_splits = create_ood_split_categorical(X_train_iid, y_train_iid, feature)

            for X_train_ood, y_train_ood, split_info in ood_splits:
                # Skip if too few samples
                if len(X_train_ood) < 10:
                    continue

                # Evaluate OOD performance
                try:
                    P_OOD = evaluate_model(X_train_ood, y_train_ood, X_test, y_test, dataset_type)

                    # Check if this is worse
                    if dataset_type == 'regression':
                        is_worse = P_OOD > best_ood_perf
                    else:
                        is_worse = P_OOD < best_ood_perf

                    if is_worse:
                        best_ood_perf = P_OOD
                        best_ood_train = X_train_ood.copy()
                        best_y_train = y_train_ood.copy()
                        best_feature = feature
                        best_split_info = split_info
                except Exception as e:
                    continue

    # Save best OOD train set
    if best_ood_train is not None:
        # Combine features and target
        if len(target_cols) > 1:
            # Need to reconstruct one-hot encoding from labels
            # best_y_train contains the label names, need to convert back to one-hot
            train_target_df = pd.DataFrame(0, index=range(len(best_y_train)), columns=target_cols)
            for idx, label in enumerate(best_y_train):
                train_target_df.loc[idx, label] = 1
            train_data = pd.concat([best_ood_train, train_target_df], axis=1)
        else:
            train_data = pd.concat([best_ood_train, best_y_train], axis=1)

        train_data.to_csv(f'Data/{dataset_name}/{dataset_name}_train.csv', index=False)

        # Save test set
        if len(target_cols) > 1:
            # Reconstruct one-hot encoding for test set
            test_target_df = pd.DataFrame(0, index=range(len(y_test)), columns=target_cols)
            for idx, label in enumerate(y_test):
                test_target_df.loc[idx, label] = 1
            test_data = pd.concat([X_test.reset_index(drop=True), test_target_df], axis=1)
        else:
            test_data = pd.concat([X_test.reset_index(drop=True), y_test.reset_index(drop=True)], axis=1)

        test_data.to_csv(f'Data/{dataset_name}/{dataset_name}_test.csv', index=False)

        print(f"  Best OOD Performance: {best_ood_perf:.4f}")
        print(f"  Feature: {best_feature}")
        print(f"  Split: {best_split_info}")

        return {
            'dname': dataset_name,
            'type': dataset_type,
            'feature_split_on': best_feature,
            'split_info': best_split_info,
            'P_IID': P_IID,
            'P_OOD': best_ood_perf
        }
    else:
        print(f"  Warning: Could not find valid OOD split, using IID split")
        # Just save IID splits
        train_data = pd.concat([X_train_iid.reset_index(drop=True),
                               y_train_iid.reset_index(drop=True)], axis=1)
        train_data.to_csv(f'Data/{dataset_name}/{dataset_name}_train.csv', index=False)

        test_data = pd.concat([X_test.reset_index(drop=True),
                              y_test.reset_index(drop=True)], axis=1)
        test_data.to_csv(f'Data/{dataset_name}/{dataset_name}_test.csv', index=False)

        return {
            'dname': dataset_name,
            'type': dataset_type,
            'feature_split_on': 'IID',
            'split_info': 'IID_split',
            'P_IID': P_IID,
            'P_OOD': P_IID
        }

# Process all datasets
datasets = [
    'abalone', 'adult', 'airfoil_noise', 'bank_marketing', 'bike_sharing',
    'breast_cancer_diagnostic', 'breast_cancer_original', 'cervical_cancer',
    'chronic_kidney_disease', 'combined_cycle_power', 'communities_crime',
    'concrete_strength', 'dry_bean', 'ecoli', 'energy_efficiency',
    'forest_fires', 'german_credit', 'glass', 'heart_disease', 'hepatitis',
    'mammographic_mass', 'openml_diabetes', 'rice', 'yeast'
]

results = []
for dataset in datasets:
    try:
        result = process_dataset(dataset)
        results.append(result)
    except Exception as e:
        print(f"  ERROR: {e}")
        import traceback
        traceback.print_exc()

# Save results to split_info.csv
results_df = pd.DataFrame(results)
results_df.to_csv('split_info.csv', index=False)

print("\n" + "="*60)
print("All datasets processed!")
print(f"Results saved to split_info.csv")
print("\nSummary:")
print(results_df.to_string(index=False))
