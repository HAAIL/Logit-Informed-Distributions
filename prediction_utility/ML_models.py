import numpy as np
import pandas as pd
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
# from pgmpy.estimators import HillClimbSearch, BicScore
import pymc as pm
from scipy.special import expit


def topological_sort_features(features, dependencies):
    """
    Return a topological ordering of features given parent dependencies.
    """
    G = nx.DiGraph()
    G.add_nodes_from(features)
    for child, parent in dependencies.items():
        G.add_edge(parent, child)  # edge from parent to child
    return list(nx.topological_sort(G))

def read_train_test_split(model_name, dataset_name, for_prior = False):
    # mifs = []
    # if dataset_name in ['hnc', 'income', 'bank', 'creditg']:
    #     with open(f'LLM_predictions/MIFs/{model_name}_{dataset_name}.txt') as f:
    #         mifs = [line.strip() for line in f.readlines()]
    #     mifs.append('target')
    
    train = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_train_v2.csv')
    y_train = train['target'].values
    X_train = train.drop('target', axis=1)
    test = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_test_v2.csv')
    y_test = test['target'].values
    X_test = test.drop('target', axis=1)

    if for_prior:
        scaler = StandardScaler()
        X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
        X_test = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)
        y_train = y_train - 1 if min(y_train) != 0 else y_train
        y_test = y_test - 1 if min(y_test) != 0 else y_test

    return X_train, X_test, y_train, y_test

def train_LR_custom_weight(data, model_pseudonym, dataset_name, seed, \
                            coeffs = None, post_train = False, test_provided = False):
    X = data.drop(columns=["target"])
    y = data["target"].values
    scaler = StandardScaler()
    if dataset_name == 'calhousing':
        X = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    mifs = []
    if test_provided:
        if dataset_name in ['hnc', 'income', 'bank', 'creditg']:
            with open(f'LLM_predictions/MIFs/{model_pseudonym}_{dataset_name}.txt') as f:
                mifs = [line.strip() for line in f.readlines()]
            mifs.append('target')
        train = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_train.csv')
        if len(mifs) > 0: train = train[mifs]
        y_train = train['target'] 
        X_train = train.drop('target', axis=1)
        test = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_test.csv')
        if len(mifs) > 0: test = test[mifs]
        y_test = test['target'] 
        X_test = test.drop('target', axis=1)

    # Train the model with data
    model = LogisticRegression(max_iter=1000) if len(np.unique(y_test)) < 3 else\
            LogisticRegression(max_iter=1000, multi_class='multinomial', solver='lbfgs')
    model.fit(X_train, y_train)
    
    # Set custom weights
    if coeffs:
        model.coef_ = np.array(list(coeffs.values())).T  
        model.intercept_ = np.array([0.5])

    if post_train:
        model.fit(X_train, y_train)

    probas = model.predict_proba(X_test)[:, 1] if len(np.unique(y_test)) < 3 else model.predict_proba(X_test)
    auc_score = 0
    if len(np.unique(y_test)) > 2:
        y_bin = label_binarize(y_test, classes=np.unique(y_test))
        auc_score = roc_auc_score(y_bin, probas, average="macro", multi_class="ovr")
    else:
        auc_score = roc_auc_score(y_test, probas)

    if not coeffs:
        print(f'AUC score for Logistic Regression for {dataset_name}: {auc_score}') 
    else:  
        print(f'AUC score for {model_pseudonym} for {dataset_name}: {auc_score}')

    coeffs = model.coef_[0]
    return coeffs, auc_score


# def learn_structure_gbn(X):
#     """
#     Learn structure of a Bayesian Network from numerical data using BIC and Hill Climbing.

#     Returns:
#         dependencies: dict mapping each feature to its parent (only 1 parent max per node).
#     """
#     hc = HillClimbSearch(X)
#     model = hc.estimate(scoring_method=BicScore(X))

#     # Extract dependencies as a dict: child -> parent
#     dependencies = {}
#     for parent, child in model.edges():
#         if child not in dependencies:
#             dependencies[child] = parent  # Only one parent (choose first if multiple)
#     return dependencies


def compute_posterior(X, y, priors, dependencies=None, target_accept=0.95, draws=100, tune=100, intercept_prior=None):
    """
    Build Bayesian logistic regression with structured priors using PyMC.

    IMPORTANT: This function only supports BINARY classification. Multi-class is NOT supported.
    The implementation uses pm.Bernoulli which requires binary labels (0 or 1).

    Parameters:
        X: pd.DataFrame - Feature data
        y: pd.Series or np.ndarray - Labels (0 or 1 for binary classification)
        priors: dict - {'feature': (mean, std)}
        dependencies: dict - {'feature': [parent1, parent2]}, optional
        target_accept: float - Target acceptance rate for NUTS
        draws: int - MCMC samples
        tune: int - Tuning steps

    Returns:
        trace: PyMC InferenceData object with posterior samples
    """
    # Check if multi-class
    n_classes = len(np.unique(y))
    if n_classes > 2:
        print(f"ERROR: compute_posterior only supports binary classification. Found {n_classes} classes.")
        print("Multi-class classification requires pm.Categorical instead of pm.Bernoulli.")
        return None
    features = list(priors.keys())
    dependencies = dependencies or {}
    # dependencies = learn_structure_gbn(X)
    
    # if dependencies:
    #     features = topological_sort_features(features, dependencies)
    with pm.Model() as model:
        betas = {}
        for feat in features:
            distro, mu, sigma = priors[feat]
            if feat in dependencies:
                parent_beta = betas[dependencies[feat]]
                betas[feat] = pm.Normal(feat, mu=parent_beta, sigma=sigma)
            else:
                # betas[feat] = pm.Normal(feat, mu=mu, sigma=sigma)
                if distro == 'Normal' or distro == 'normal':
                    betas[feat] = pm.Normal(feat, mu=mu, sigma=sigma)
                elif distro == 'Uniform' or distro == 'uniform':
                    betas[feat] = pm.Uniform(feat, lower=mu - sigma, upper=mu + sigma)
                else:
                    betas[feat] = pm.Laplace(feat, mu=mu, b=sigma) 

        if intercept_prior is None:
            intercept = pm.Normal("intercept", mu=0, sigma=1)
        else:
            intercept = pm.Normal("intercept", mu=intercept_prior[1], sigma=intercept_prior[2])
        logits = intercept + sum(X[feat].values * betas[feat] for feat in features)
        y_obs = pm.Bernoulli("y_obs", logit_p=logits, observed=y)
        try:
            trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, return_inferencedata=True, progressbar=False)
        except Exception as e:
            print(f'ERROR: {e}')
            return None

    return trace

def compute_posterior_multiclass(X, y, priors, n_classes, dependencies=None, target_accept=0.95, draws=100, tune=100, intercept_prior=None):
    features = list(priors.keys())
    dependencies = dependencies or {}

    with pm.Model() as model:
        # Create coefficient matrices: (n_features, n_classes-1)
        # We use n_classes-1 because the last class is the reference category
        betas = {}

        for feat in features:
            distro, mu, sigma = priors[feat]

            # Create coefficients for each class (except reference class)
            feat_coeffs = []
            for k in range(n_classes - 1):
                if feat in dependencies:
                    # Not implementing dependencies for multi-class yet
                    parent_beta = betas[dependencies[feat]]
                    coeff = pm.Normal(f"{feat}_class{k}", mu=parent_beta, sigma=sigma)
                else:
                    if distro == 'Normal' or distro == 'normal':
                        coeff = pm.Normal(f"{feat}_class{k}", mu=mu, sigma=sigma)
                    elif distro == 'Uniform' or distro == 'uniform':
                        coeff = pm.Uniform(f"{feat}_class{k}", lower=mu - sigma, upper=mu + sigma)
                    else:
                        coeff = pm.Laplace(f"{feat}_class{k}", mu=mu, b=sigma)
                feat_coeffs.append(coeff)

            betas[feat] = feat_coeffs

        # Intercepts for each class (except reference)
        intercepts = []
        for k in range(n_classes - 1):
            if intercept_prior is None:
                intercept = pm.Normal(f"intercept_class{k}", mu=0, sigma=1)
            else:
                intercept = pm.Normal(f"intercept_class{k}", mu=intercept_prior[1], sigma=intercept_prior[2])
            intercepts.append(intercept)

        # Compute logits for each class
        logits = []
        for k in range(n_classes - 1):
            logit_k = intercepts[k] + sum(X[feat].values * betas[feat][k] for feat in features)
            logits.append(logit_k)

        # Stack logits and add reference class (all zeros)
        logits_stacked = pm.math.stack(logits, axis=1)  # Shape: (n_samples, n_classes-1)

        # Add reference class logits (zeros)
        reference_logits = np.zeros((X.shape[0], 1))
        logits_all = pm.math.concatenate([logits_stacked, reference_logits], axis=1)  # Shape: (n_samples, n_classes)

        # Categorical likelihood
        y_obs = pm.Categorical("y_obs", logit_p=logits_all, observed=y)

        try:
            trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, return_inferencedata=True, progressbar=False)
        except Exception as e:
            print(f'ERROR in multi-class posterior: {e}')
            return None

    return trace

def predict_using_posterior(trace, X_new, feature_names):
    X = X_new[feature_names].values
    posterior_samples = {}
    for fname in feature_names:
        posterior_samples[fname] = trace.posterior[fname].stack(sample=("chain", "draw")).values

    intercept = trace.posterior["intercept"].stack(sample=("chain", "draw")).values
    intercept = intercept.reshape(1, -1)  

    coefs = np.array([posterior_samples[f] for f in feature_names]) 
    logits = X @ coefs + intercept
    probs = expit(logits)  
    predicted_probs = probs.mean(axis=1)

    return predicted_probs

def predict_using_posterior_multiclass(trace, X_new, feature_names, n_classes):
    """
    Multi-class version of predict_using_posterior.

    Args:
        trace: PyMC InferenceData object from compute_posterior_multiclass
        X_new: pandas DataFrame of new examples
        feature_names: list of feature names used in model
        n_classes: int - Number of classes

    Returns:
        predicted_probs: numpy array of shape (n_samples, n_classes) with class probabilities
    """
    X = X_new[feature_names].values  # Shape: (n_samples, n_features)

    # Extract posterior samples for each feature and class
    n_samples = X.shape[0]

    # Get number of MCMC samples
    sample_intercept = trace.posterior[f"intercept_class0"].stack(sample=("chain", "draw")).values
    n_mcmc_samples = len(sample_intercept)

    # Initialize logits array: (n_samples, n_classes, n_mcmc_samples)
    logits_all_classes = np.zeros((n_samples, n_classes, n_mcmc_samples))

    # Compute logits for each class (except reference)
    for k in range(n_classes - 1):
        # Get intercept for this class
        intercept_k = trace.posterior[f"intercept_class{k}"].stack(sample=("chain", "draw")).values
        intercept_k = intercept_k.reshape(1, -1)  # Shape: (1, n_mcmc_samples)

        # Get coefficients for this class
        coefs_k = []
        for fname in feature_names:
            coef = trace.posterior[f"{fname}_class{k}"].stack(sample=("chain", "draw")).values
            coefs_k.append(coef)
        coefs_k = np.array(coefs_k)  # Shape: (n_features, n_mcmc_samples)

        # Compute logits: X @ coefs + intercept
        logits_k = X @ coefs_k + intercept_k  # Shape: (n_samples, n_mcmc_samples)
        logits_all_classes[:, k, :] = logits_k

    # Reference class has logits = 0
    # logits_all_classes[:, n_classes-1, :] is already zeros

    # Apply softmax to get probabilities
    # Shape: (n_samples, n_classes, n_mcmc_samples)
    exp_logits = np.exp(logits_all_classes)
    probs_all = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    # Average over MCMC samples
    predicted_probs = probs_all.mean(axis=2)  # Shape: (n_samples, n_classes)

    return predicted_probs