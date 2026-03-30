import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
import os
from tqdm import tqdm
from prediction_utility.LLM_requests import llm_single_token_logprob, LLM_load_model
from prediction_utility.ML_models import compute_posterior, predict_using_posterior, \
                                        compute_posterior_multiclass, predict_using_posterior_multiclass
from prediction_utility.ML_models_regression import compute_posterior_regression, \
                                                     predict_using_posterior_regression
from prediction_utility.KL_divergence import calc_KL
from dataset_specific_templates import get_templates_and_tokens

np.random.seed(42)


def generate_feature_impact_sentences(feature, task):
    """Generate sentence templates for probing LLM beliefs about feature impact."""
    templates = [
        "The impact of {} on {} is",
        "The effect of {} on {} is",
        "The influence of {} on {} is",
        "The relationship between {} and {} is",
        "When considering {}, the effect on {} is",
        "In terms of {}, the impact on {} is",
        "Regarding {}, the influence on {} is",
        "For {}, the relationship with {} is",
        "The correlation between {} and {} is",
        "The role of {} in {} is",
        "The association between {} and {} is",
        "The contribution of {} to {} is",
        "How {} affects {} is",
        "The connection between {} and {} is",
        "The significance of {} for {} is",
        "The relevance of {} to {} is",
        "Concerning {}, the impact on {} is",
        "With respect to {}, the influence on {} is",
        "The predictive power of {} for {} is",
        "The degree to which {} influences {} is"
    ]
    return [template.format(feature, task) for template in templates]


def calculate_entropy(prob_pos, prob_neg):
    """Calculate Shannon entropy for binary probability distribution."""
    total_prob = prob_pos + prob_neg
    p_pos = prob_pos / total_prob
    p_neg = prob_neg / total_prob

    epsilon = 1e-12
    p_pos = max(p_pos, epsilon)
    p_neg = max(p_neg, epsilon)

    entropy = -p_pos * np.log(p_pos) - p_neg * np.log(p_neg)
    return entropy


def std_method_variance_of_differences(differences, alpha, beta):
    """Calculate std using variance of logit differences - NO SCALING."""
    std_of_diffs = np.std(differences)
    return std_of_diffs  # Raw std, no alpha/beta scaling


def std_method_entropy(prob_pairs, beta):
    """Calculate std using average entropy across sentences - NO SCALING."""
    entropies = [calculate_entropy(prob_pos, prob_neg) for prob_pos, prob_neg in prob_pairs]
    mean_entropy = np.mean(entropies)
    return mean_entropy  # Raw entropy, no beta scaling


def std_method_logprob_difference(log_prob_pairs, alpha, beta):
    """Calculate std using variance of log probability differences - NO SCALING."""
    log_prob_diffs = [log_pos - log_neg for log_pos, log_neg in log_prob_pairs]
    std_of_log_diffs = np.std(log_prob_diffs)
    return std_of_log_diffs  # Raw std, no alpha/beta scaling


def load_data_with_target(dataset_name, data_dictionary, use_ood_split=True):
    """
    Load dataset and create proper target column based on dictionary.

    Returns:
        train_df, test_df: DataFrames with 'target' column extracted
        task_type: 'binary', 'multiclass', or 'regression'
    """
    # Load data
    if use_ood_split:
        train = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_train.csv")
        test = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_test.csv")
    else:
        full_data = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_prep.csv")
        train, test = train_test_split(full_data, test_size=0.3, random_state=42)

    # Find target columns from dictionary
    target_cols = data_dictionary[data_dictionary['target'] == 'T']['name'].tolist()

    if len(target_cols) == 0:
        raise ValueError(f"No target columns found in dictionary for {dataset_name}")

    # Determine task type and create target column
    if len(target_cols) == 1:
        # Regression or single-column target
        target_col = target_cols[0]
        n_unique = train[target_col].nunique()

        # Use data type and number of unique values to determine task
        # Regression: continuous or count data with many values
        # Classification: categorical with few unique values
        if n_unique > 20:  # More than 20 unique values -> regression
            task_type = 'regression'
        elif n_unique <= 2:
            task_type = 'binary'
        else:
            task_type = 'multiclass'

        # Only rename and drop if target column is not already named 'target'
        if target_col != 'target':
            train['target'] = train[target_col]
            test['target'] = test[target_col]
            train = train.drop(columns=[target_col])
            test = test.drop(columns=[target_col])

    elif len(target_cols) == 2:
        # Check if binary one-hot encoding or multi-output regression
        # Binary one-hot: values are 0/1 and sum to 1 per row
        # Multi-output regression: continuous values
        first_col_unique = train[target_cols[0]].nunique()

        if first_col_unique > 50:
            # Multi-output regression - use first target only
            task_type = 'regression'
            train['target'] = train[target_cols[0]]
            test['target'] = test[target_cols[0]]
            train = train.drop(columns=target_cols)
            test = test.drop(columns=target_cols)
        else:
            # Binary classification with one-hot encoding
            task_type = 'binary'
            train['target'] = train[target_cols[0]].astype(int)
            test['target'] = test[target_cols[0]].astype(int)
            train = train.drop(columns=target_cols)
            test = test.drop(columns=target_cols)

    else:
        # Multiclass with one-hot encoding
        task_type = 'multiclass'
        # Find which class each row belongs to
        train['target'] = train[target_cols].idxmax(axis=1)
        test['target'] = test[target_cols].idxmax(axis=1)
        # Convert class names to integers
        class_mapping = {cls: idx for idx, cls in enumerate(target_cols)}
        train['target'] = train['target'].map(class_mapping)
        test['target'] = test['target'].map(class_mapping)
        # Drop one-hot columns
        train = train.drop(columns=target_cols)
        test = test.drop(columns=target_cols)

    return train, test, task_type


def get_logit_priors_binary(X_test, data_dictionary, task_description, num_sentences, alpha, beta, std_method, prior_save_path, dataset_name, mean_scale=1.0, ignore_previous_run=False):
    """
    Extract priors for binary classification using positive/negative token probabilities.

    Compatible with new dict format:
    - name: feature name
    - f_nl: natural language description
    - actual_value: actual value for one-hot encoded features
    - target: T/F indicating if feature is target
    """
    if os.path.exists(prior_save_path) and not ignore_previous_run:
        with open(prior_save_path, 'r') as f:
            coeffs_prior = f.readlines()
        coeffs_prior = {line.split(": ")[0].strip(): (
            line.split(": ")[1].strip().split(",")[0],
            float(line.split(": ")[1].strip().split(",")[1]),
            float(line.split(": ")[1].strip().split(",")[2])
        ) for line in coeffs_prior if ":" in line}
        return coeffs_prior

    coeffs_prior = {}

    # Get dataset-specific templates and tokens
    templates, (token_pos, token_neg) = get_templates_and_tokens(dataset_name, task_description)

    for col in tqdm(X_test.columns, desc="Extracting binary priors"):
        # Find feature in dictionary
        dict_row = data_dictionary[data_dictionary['name'] == col]

        if dict_row.empty:
            print(f"WARNING: Feature {col} not found in dictionary, skipping")
            continue

        feat_desc = dict_row.iloc[0]['f_nl']

        # Generate sentences using dataset-specific templates
        sentences = [template.format(feat_desc, task_description) for template in templates]
        np.random.shuffle(sentences)
        sentences = sentences[:num_sentences]

        differences = []
        prob_pairs = []
        log_prob_pairs = []

        for sentence in sentences:
            # Use dataset-specific tokens
            prob_pos, log_prob_pos = llm_single_token_logprob(sentence, token_pos)
            prob_neg, log_prob_neg = llm_single_token_logprob(sentence, token_neg)

            prob_pairs.append((prob_pos, prob_neg))
            log_prob_pairs.append((log_prob_pos, log_prob_neg))

            # Calculate logit difference
            total_prob = prob_pos + prob_neg
            p_pos_norm = prob_pos / total_prob
            p_pos_norm = np.clip(p_pos_norm, 1e-10, 1 - 1e-10)  # Avoid log(0) or log(inf)
            mean_of_diffs = np.log(p_pos_norm / (1 - p_pos_norm))
            differences.append(mean_of_diffs)

        mean_of_diffs = np.mean(differences)

        # Scale the mean
        mean_of_diffs_scaled = mean_of_diffs * mean_scale

        # Calculate std based on method
        if std_method == 'variance':
            confidence_std = std_method_variance_of_differences(differences, alpha, beta)
        elif std_method == 'entropy':
            confidence_std = std_method_entropy(prob_pairs, beta)
        elif std_method == 'logprob_difference':
            confidence_std = std_method_logprob_difference(log_prob_pairs, alpha, beta)
        else:
            raise ValueError(f"Unknown std_method: {std_method}")

        coeffs_prior[col] = ('Normal', mean_of_diffs_scaled, confidence_std)

    # Save priors
    os.makedirs(os.path.dirname(prior_save_path), exist_ok=True)
    with open(prior_save_path, 'w+') as f:
        for feat, coeff in coeffs_prior.items():
            f.write(f"{feat}: {coeff[0]},{coeff[1]},{coeff[2]}\n")

    return coeffs_prior


def get_logit_priors_regression(X_test, data_dictionary, task_description, num_sentences, alpha, beta, std_method, prior_save_path, dataset_name, ignore_previous_run=False):
    """
    Extract priors for regression using increases/decreases token probabilities.

    For regression, we probe whether a feature "increases" or "decreases" the target,
    then use logit differences as coefficient priors.
    """
    if os.path.exists(prior_save_path) and not ignore_previous_run:
        with open(prior_save_path, 'r') as f:
            coeffs_prior = f.readlines()
        coeffs_prior = {line.split(": ")[0].strip(): (
            line.split(": ")[1].strip().split(",")[0],
            float(line.split(": ")[1].strip().split(",")[1]),
            float(line.split(": ")[1].strip().split(",")[2])
        ) for line in coeffs_prior if ":" in line}
        return coeffs_prior

    coeffs_prior = {}

    # Get dataset-specific templates and tokens
    templates, (token_pos, token_neg) = get_templates_and_tokens(dataset_name, task_description)

    for col in tqdm(X_test.columns, desc="Extracting regression priors"):
        dict_row = data_dictionary[data_dictionary['name'] == col]

        if dict_row.empty:
            print(f"WARNING: Feature {col} not found in dictionary, skipping")
            continue

        feat_desc = dict_row.iloc[0]['f_nl']

        # Generate sentences using dataset-specific templates
        sentences = [template.format(feat_desc, task_description) for template in templates]
        np.random.shuffle(sentences)
        sentences = sentences[:num_sentences]

        differences = []
        prob_pairs = []
        log_prob_pairs = []

        for sentence in sentences:
            # Use dataset-specific tokens (e.g., "increases"/"decreases")
            prob_pos, log_prob_pos = llm_single_token_logprob(sentence, token_pos)
            prob_neg, log_prob_neg = llm_single_token_logprob(sentence, token_neg)

            prob_pairs.append((prob_pos, prob_neg))
            log_prob_pairs.append((log_prob_pos, log_prob_neg))

            # Calculate logit difference
            total_prob = prob_pos + prob_neg
            p_pos_norm = prob_pos / total_prob
            p_pos_norm = np.clip(p_pos_norm, 1e-10, 1 - 1e-10)  # Avoid log(0) or log(inf)
            mean_of_diffs = np.log(p_pos_norm / (1 - p_pos_norm))
            differences.append(mean_of_diffs)

        mean_of_diffs = np.mean(differences)

        # Calculate std based on method
        if std_method == 'variance':
            confidence_std = std_method_variance_of_differences(differences, alpha, beta)
        elif std_method == 'entropy':
            confidence_std = std_method_entropy(prob_pairs, beta)
        elif std_method == 'logprob_difference':
            confidence_std = std_method_logprob_difference(log_prob_pairs, alpha, beta)
        else:
            raise ValueError(f"Unknown std_method: {std_method}")

        coeffs_prior[col] = ('Normal', mean_of_diffs, confidence_std)

    # Save priors
    os.makedirs(os.path.dirname(prior_save_path), exist_ok=True)
    with open(prior_save_path, 'w+') as f:
        for feat, coeff in coeffs_prior.items():
            f.write(f"{feat}: {coeff[0]},{coeff[1]},{coeff[2]}\n")

    return coeffs_prior


def get_logit_priors_multiclass(X_test, y_test, data_dictionary, task_description, num_sentences, alpha, beta, std_method, prior_save_path, ignore_previous_run=False):
    """
    Extract priors for multiclass classification.

    For each feature, create separate priors for each class (one-vs-rest).
    Returns: {feature: [(distro, mean, std) for each class]}
    """
    # Calculate n_classes first (needed for loading cached priors)
    n_classes = len(np.unique(y_test))

    # Get class labels and descriptions from dictionary
    target_rows = data_dictionary[data_dictionary['target'] == 'T']
    class_labels = []
    class_descriptions = []

    if len(target_rows) > 0:
        for _, row in target_rows.iterrows():
            class_labels.append(row['actual_value'] if pd.notna(row['actual_value']) else f"class_{len(class_labels)}")
            class_descriptions.append(row['f_nl'])  # Natural language description for each class

        if len(class_labels) != n_classes:
            class_labels = [f"class_{k}" for k in range(n_classes)]
            class_descriptions = [task_description] * n_classes  # Fallback
    else:
        class_labels = [f"class_{k}" for k in range(n_classes)]
        class_descriptions = [task_description] * n_classes

    # Check for cached priors (n_classes is now defined)
    if os.path.exists(prior_save_path) and not ignore_previous_run:
        with open(prior_save_path, 'r') as f:
            lines = f.readlines()
        coeffs_prior_multiclass = {}
        for line in lines:
            if ":" not in line:
                continue
            feat = line.split(": ")[0].strip()
            values = line.split(": ")[1].strip().split(",")
            distro = values[0].strip()
            class_priors = []
            for k in range(n_classes):
                mean = float(values[1 + 2*k])
                std = float(values[2 + 2*k])
                class_priors.append((distro, mean, std))
            coeffs_prior_multiclass[feat] = class_priors
        return coeffs_prior_multiclass

    coeffs_prior_multiclass = {}

    for col in tqdm(X_test.columns, desc="Extracting multiclass priors"):
        dict_row = data_dictionary[data_dictionary['name'] == col]

        if dict_row.empty:
            print(f"WARNING: Feature {col} not found in dictionary, skipping")
            continue

        feat_desc = dict_row.iloc[0]['f_nl']

        # For each class, compute class-specific prior (TRUE OVR)
        class_priors = []
        for k in range(n_classes):
            # Use class-specific task description for this class
            class_task_desc = class_descriptions[k]

            # Generate sentences with THIS class's description
            sentences = generate_feature_impact_sentences(feat_desc, class_task_desc)
            np.random.shuffle(sentences)
            sentences = sentences[:num_sentences]

            differences = []
            prob_pairs = []

            for sentence in sentences:
                # Query LLM for this specific class
                prob_pos, log_prob_pos = llm_single_token_logprob(sentence, "positive")
                prob_neg, log_prob_neg = llm_single_token_logprob(sentence, "negative")

                prob_pairs.append((prob_pos, prob_neg))

                # Calculate logit for class k vs rest
                total_prob = prob_pos + prob_neg
                p_pos_norm = prob_pos / total_prob
                p_pos_norm = np.clip(p_pos_norm, 1e-10, 1 - 1e-10)
                logit_k = np.log(p_pos_norm / (1 - p_pos_norm))
                differences.append(logit_k)

            mean_logit = np.mean(differences)

            if std_method == 'variance':
                confidence_std = std_method_variance_of_differences(differences, alpha, beta)
            elif std_method == 'entropy':
                confidence_std = std_method_entropy(prob_pairs, beta)
            else:
                confidence_std = std_method_variance_of_differences(differences, alpha, beta)

            class_priors.append(('Normal', mean_logit, confidence_std))

        coeffs_prior_multiclass[col] = class_priors

    # Save priors
    os.makedirs(os.path.dirname(prior_save_path), exist_ok=True)
    with open(prior_save_path, 'w+') as f:
        for feat, class_priors in coeffs_prior_multiclass.items():
            values = [class_priors[0][0]]  # Distribution type
            for distro, mean, std in class_priors:
                values.extend([str(mean), str(std)])
            f.write(f"{feat}: {','.join(values)}\n")

    return coeffs_prior_multiclass


def construct_prior_save_path(model_name, dataset_name, task_type, seed, num_sentences=None):
    """Construct path for saving priors."""
    if num_sentences is not None:
        file_name = f"{model_name}_{dataset_name}_{task_type}_{seed}_sent{num_sentences}.txt"
    else:
        file_name = f"{model_name}_{dataset_name}_{task_type}_{seed}.txt"
    return f'LLM_predictions/Priors/{file_name}'


def run_binary_classification(dataset_name, model_name, task_description, alpha, beta, num_sentences, std_method, use_ood_split, ignore_previous_run=False):
    """Run prior extraction and Bayesian inference for binary classification."""

    print(f"\n{'='*60}")
    print(f"Running Binary Classification: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    data_dictionary = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_dict.csv")
    train, test, _ = load_data_with_target(dataset_name, data_dictionary, use_ood_split)

    X_train = train.drop(columns=['target'])
    y_train = train['target'].values
    X_test = test.drop(columns=['target'])
    y_test = test['target'].values

    # Fill missing values
    X_train = X_train.fillna(X_train.mean())
    X_test = X_test.fillna(X_test.mean())

    # PROPER OOD STANDARDIZATION: Each set standardized independently
    # Train: standardize with train statistics
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = (X_train - train_mean) / train_std.replace(0, 1)  # Avoid division by zero

    # Test: standardize with TEST statistics (no leakage!)
    test_mean = X_test.mean()
    test_std = X_test.std()
    X_test = (X_test - test_mean) / test_std.replace(0, 1)

    # Fill any remaining NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Adjust labels to 0/1
    y_train = y_train - 1 if min(y_train) == 1 else y_train
    y_test = y_test - 1 if min(y_test) == 1 else y_test

    # Extract priors
    seed = 42
    prior_save_path = construct_prior_save_path(model_name, dataset_name, 'binary', seed, num_sentences)

    print(f"Extracting priors...")
    coeffs_prior = get_logit_priors_binary(
        X_test, data_dictionary, task_description,
        num_sentences, alpha, beta, std_method, prior_save_path, dataset_name, mean_scale=1.0, ignore_previous_run=ignore_previous_run
    )

    print(f"Extracted priors for {len(coeffs_prior)} features")

    if len(coeffs_prior) == 0:
        print("ERROR: No priors extracted. Check that features in data match dictionary.")
        return 0.0, 0.0

    # Compute posterior
    print(f"Computing posterior...")
    X_new = X_train[list(coeffs_prior.keys())]
    trace = compute_posterior(X_new, y_train, coeffs_prior, dependencies=None,
                             target_accept=0.95, draws=1000, tune=1000)

    if trace is None:
        print("ERROR: Failed to compute posterior")
        return 0.0, 0.0

    # Predict
    print(f"Making predictions...")
    probas_train = predict_using_posterior(trace, X_train, list(coeffs_prior.keys()))
    probas_test = predict_using_posterior(trace, X_test, list(coeffs_prior.keys()))

    if probas_train is None or probas_test is None:
        print("ERROR: Failed to make predictions")
        return 0.0, 0.0

    # Evaluate
    auc_train = roc_auc_score(y_train, probas_train)
    auc_test = roc_auc_score(y_test, probas_test)

    print(f"\nResults:")
    print(f"  Train AUC: {auc_train:.4f}")
    print(f"  Test AUC:  {auc_test:.4f}")

    # Save posteriors
    posterior_samples = {}
    for feature in coeffs_prior.keys():
        posterior_samples[feature] = trace.posterior[feature].stack(sample=("chain", "draw")).values
    posterior_samples['intercept'] = trace.posterior["intercept"].stack(sample=("chain", "draw")).values

    os.makedirs("LLM_predictions/Posteriors", exist_ok=True)
    np.savez(f"LLM_predictions/Posteriors/{model_name}_{dataset_name}_binary.npz", **posterior_samples)

    # Calculate KL divergence
    try:
        mean_kl = calc_KL(f"LLM_predictions/Posteriors/{model_name}_{dataset_name}_binary.npz", dataset=dataset_name)
        print(f"  Mean KL:   {mean_kl:.4f}")
    except (FileNotFoundError, KeyError, ValueError, Exception) as e:
        mean_kl = 0.0
        print(f"  Mean KL:   N/A (skipped)")

    return auc_test, mean_kl


def run_regression(dataset_name, model_name, task_description, alpha, beta, num_sentences, std_method, use_ood_split, ignore_previous_run=False):
    """Run prior extraction and Bayesian inference for regression."""

    print(f"\n{'='*60}")
    print(f"Running Regression: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    data_dictionary = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_dict.csv")
    train, test, _ = load_data_with_target(dataset_name, data_dictionary, use_ood_split)

    X_train = train.drop(columns=['target'])
    y_train = train['target'].values
    X_test = test.drop(columns=['target'])
    y_test = test['target'].values

    # PROPER OOD STANDARDIZATION: Each set standardized independently
    # Train: standardize with train statistics
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = (X_train - train_mean) / train_std.replace(0, 1)  # Avoid division by zero

    # Test: standardize with TEST statistics (no leakage!)
    test_mean = X_test.mean()
    test_std = X_test.std()
    X_test = (X_test - test_mean) / test_std.replace(0, 1)

    # Fill any remaining NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Extract priors
    seed = 42
    prior_save_path = construct_prior_save_path(model_name, dataset_name, 'regression', seed, num_sentences)

    print(f"Extracting priors...")
    coeffs_prior = get_logit_priors_regression(
        X_test, data_dictionary, task_description,
        num_sentences, alpha, beta, std_method, prior_save_path, dataset_name, ignore_previous_run
    )

    print(f"Extracted priors for {len(coeffs_prior)} features")

    if len(coeffs_prior) == 0:
        print("ERROR: No priors extracted. Check that features in data match dictionary.")
        return 0.0, 0.0

    # Compute posterior
    print(f"Computing posterior...")
    X_new = X_train[list(coeffs_prior.keys())]
    trace = compute_posterior_regression(X_new, y_train, coeffs_prior, dependencies=None,
                                        target_accept=0.95, draws=1000, tune=1000)

    if trace is None:
        print("ERROR: Failed to compute posterior")
        return 0.0, 0.0

    # Predict
    print(f"Making predictions...")
    preds_train = predict_using_posterior_regression(trace, X_train, list(coeffs_prior.keys()))
    preds_test = predict_using_posterior_regression(trace, X_test, list(coeffs_prior.keys()))

    if preds_train is None or preds_test is None:
        print("ERROR: Failed to make predictions")
        return 0.0, 0.0

    # Evaluate
    mse_train = mean_squared_error(y_train, preds_train)
    mse_test = mean_squared_error(y_test, preds_test)

    print(f"\nResults:")
    print(f"  Train MSE: {mse_train:.4f}")
    print(f"  Test MSE:  {mse_test:.4f}")

    # Save posteriors
    posterior_samples = {}
    for feature in coeffs_prior.keys():
        posterior_samples[feature] = trace.posterior[feature].stack(sample=("chain", "draw")).values
    posterior_samples['intercept'] = trace.posterior["intercept"].stack(sample=("chain", "draw")).values

    os.makedirs("LLM_predictions/Posteriors", exist_ok=True)
    np.savez(f"LLM_predictions/Posteriors/{model_name}_{dataset_name}_regression.npz", **posterior_samples)

    return mse_test, 0.0


def run_multiclass(dataset_name, model_name, task_description, alpha, beta, num_sentences, std_method, use_ood_split, ignore_previous_run=False):
    """Run prior extraction and Bayesian inference for multiclass classification."""

    print(f"\n{'='*60}")
    print(f"Running Multiclass Classification: {dataset_name}")
    print(f"{'='*60}")

    # Load data
    data_dictionary = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_dict.csv")
    train, test, _ = load_data_with_target(dataset_name, data_dictionary, use_ood_split)

    X_train = train.drop(columns=['target'])
    y_train = train['target'].values
    X_test = test.drop(columns=['target'])
    y_test = test['target'].values

    # PROPER OOD STANDARDIZATION: Each set standardized independently
    # Train: standardize with train statistics
    train_mean = X_train.mean()
    train_std = X_train.std()
    X_train = (X_train - train_mean) / train_std.replace(0, 1)  # Avoid division by zero

    # Test: standardize with TEST statistics (no leakage!)
    test_mean = X_test.mean()
    test_std = X_test.std()
    X_test = (X_test - test_mean) / test_std.replace(0, 1)

    # Fill any remaining NaNs
    X_train = X_train.fillna(0)
    X_test = X_test.fillna(0)

    # Adjust labels to start from 0
    y_train = y_train - 1 if min(y_train) == 1 else y_train
    y_test = y_test - 1 if min(y_test) == 1 else y_test

    n_classes = len(np.unique(y_test))
    print(f"Number of classes: {n_classes}")

    # Extract priors
    seed = 42
    prior_save_path = construct_prior_save_path(model_name, dataset_name, 'multiclass', seed, num_sentences)

    print(f"Extracting priors...")
    coeffs_prior_multiclass = get_logit_priors_multiclass(
        X_test, y_test, data_dictionary, task_description,
        num_sentences, alpha, beta, std_method, prior_save_path, ignore_previous_run
    )

    print(f"Extracted priors for {len(coeffs_prior_multiclass)} features")

    if len(coeffs_prior_multiclass) == 0:
        print("ERROR: No priors extracted. Check that features in data match dictionary.")
        return 0.0, 0.0

    # Convert to format expected by compute_posterior_multiclass
    base_priors = {feat: class_priors[0] for feat, class_priors in coeffs_prior_multiclass.items()}

    # Compute posterior
    print(f"Computing posterior...")
    X_new = X_train[list(base_priors.keys())]
    trace = compute_posterior_multiclass(X_new, y_train, base_priors, n_classes,
                                        dependencies=None, target_accept=0.95, draws=1000, tune=1000)

    if trace is None:
        print("ERROR: Failed to compute posterior")
        return 0.0, 0.0

    # Predict
    print(f"Making predictions...")
    probas_test = predict_using_posterior_multiclass(trace, X_test, list(base_priors.keys()), n_classes)

    if probas_test is None:
        print("ERROR: Failed to make predictions")
        return 0.0, 0.0

    # Evaluate
    y_test_bin = label_binarize(y_test, classes=np.unique(y_test))
    auc_test = roc_auc_score(y_test_bin, probas_test, average='macro', multi_class='ovr')

    print(f"\nResults:")
    print(f"  Test AUC:  {auc_test:.4f}")

    # Save posteriors
    posterior_samples = {}
    for feature in base_priors.keys():
        for k in range(n_classes - 1):
            posterior_samples[f"{feature}_class{k}"] = trace.posterior[f"{feature}_class{k}"].stack(sample=("chain", "draw")).values
    for k in range(n_classes - 1):
        posterior_samples[f"intercept_class{k}"] = trace.posterior[f"intercept_class{k}"].stack(sample=("chain", "draw")).values

    os.makedirs("LLM_predictions/Posteriors", exist_ok=True)
    np.savez(f"LLM_predictions/Posteriors/{model_name}_{dataset_name}_multiclass.npz", **posterior_samples)

    return auc_test, 0.0


def main(args):
    """Main entry point."""

    # Load model
    LLM_load_model(args.model)

    # Read split_info to determine task types
    split_info = pd.read_csv('split_info.csv')

    results = []

    for dataset in args.datasets:
        # Find dataset in split_info
        dataset_info = split_info[split_info['dname'] == dataset]

        if dataset_info.empty:
            print(f"WARNING: Dataset {dataset} not found in split_info.csv, skipping")
            continue

        task_type = dataset_info.iloc[0]['type']

        # Get task description from dict
        try:
            data_dict = pd.read_csv(f"Data/{dataset}/{dataset}_dict.csv")
            target_row = data_dict[data_dict['target'] == 'T'].iloc[0]
            task_description = target_row['f_nl']
        except:
            task_description = "target"

        print(f"\nTask Description: {task_description}")

        # Run appropriate pipeline
        if task_type == 'binary classification':
            auc, kl = run_binary_classification(
                dataset, args.model, task_description,
                args.alpha, args.beta, args.num_sentences, args.std_method,
                args.use_ood_split, args.ignore_previous_run
            )
            results.append({
                'dataset': dataset,
                'type': task_type,
                'metric': 'AUC',
                'value': auc,
                'kl': kl
            })
        elif task_type == 'regression':
            mse, _ = run_regression(
                dataset, args.model, task_description,
                args.alpha, args.beta, args.num_sentences, args.std_method,
                args.use_ood_split, args.ignore_previous_run
            )
            results.append({
                'dataset': dataset,
                'type': task_type,
                'metric': 'MSE',
                'value': mse,
                'kl': 0.0
            })
        elif task_type == 'multiclass classification':
            auc, _ = run_multiclass(
                dataset, args.model, task_description,
                args.alpha, args.beta, args.num_sentences, args.std_method,
                args.use_ood_split, args.ignore_previous_run
            )
            results.append({
                'dataset': dataset,
                'type': task_type,
                'metric': 'AUC',
                'value': auc,
                'kl': 0.0
            })
        else:
            print(f"WARNING: Unknown task type '{task_type}' for dataset {dataset}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for result in results:
        print(f"{result['dataset']:30s} | {result['type']:25s} | {result['metric']}: {result['value']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract LLM priors and run Bayesian inference")
    parser.add_argument('--datasets', nargs='+', required=True, help="List of dataset names")
    parser.add_argument('--model', type=str, required=True, help="LLM model name")
    parser.add_argument('--alpha', type=float, default=0.2, help="Alpha for std calculation")
    parser.add_argument('--beta', type=float, default=2.0, help="Beta for std calculation")
    parser.add_argument('--num_sentences', type=int, default=10, help="Number of sentence templates")
    parser.add_argument('--std_method', type=str, default='variance',
                       choices=['variance', 'entropy', 'logprob_difference'],
                       help="Method for calculating standard deviation")
    parser.add_argument('--use_ood_split', action='store_true',
                       help="Use OOD train/test splits instead of random split")
    parser.add_argument('--ignore_previous_run', action='store_true',
                       help="Force re-extraction of priors even if cached files exist")

    args = parser.parse_args()
    main(args)
