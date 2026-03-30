import pandas as pd
import numpy as np
import pymc as pm
from sklearn.preprocessing import StandardScaler

def read_train_test_split_regression(model_name, dataset_name, use_full_train=False):
    if use_full_train:
        full_data = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_prep.csv')
        test = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_test_v2.csv')
        full_data['_row_id'] = range(len(full_data))
        merged = full_data.merge(test, on=list(test.columns), how='left', indicator=True)
        train = full_data[merged['_merge'] == 'left_only'].drop('_row_id', axis=1)
        test = test.copy()
    else:
        train = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_train_v2.csv')
        test = pd.read_csv(f'../Data/datasets/{dataset_name}/{dataset_name}_test_v2.csv')

    X_train = train.drop(columns=['target'])
    y_train = train['target'].values
    X_test = test.drop(columns=['target'])
    y_test = test['target'].values

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

    X_train_scaled = X_train_scaled.fillna(X_train_scaled.mean())
    X_test_scaled = X_test_scaled.fillna(X_test_scaled.mean())

    return X_train_scaled, X_test_scaled, y_train, y_test

def compute_posterior_regression(X, y, priors, dependencies=None, target_accept=0.95, draws=100, tune=100, intercept_prior=None, noise_prior=('HalfNormal', 1.0)):
    features = list(priors.keys())
    dependencies = dependencies or {}

    with pm.Model() as model:
        betas = {}
        for feat in features:
            distro, mu, sigma = priors[feat]
            if feat in dependencies:
                parent_beta = betas[dependencies[feat]]
                betas[feat] = pm.Normal(feat, mu=parent_beta, sigma=sigma)
            else:
                if distro == 'Normal' or distro == 'normal':
                    betas[feat] = pm.Normal(feat, mu=mu, sigma=sigma)
                elif distro == 'Uniform' or distro == 'uniform':
                    betas[feat] = pm.Uniform(feat, lower=mu - sigma, upper=mu + sigma)
                else:
                    betas[feat] = pm.Laplace(feat, mu=mu, b=sigma)

        if intercept_prior is None:
            intercept = pm.Normal("intercept", mu=0, sigma=10)
        else:
            intercept = pm.Normal("intercept", mu=intercept_prior[1], sigma=intercept_prior[2])

        if noise_prior[0] == 'HalfNormal':
            sigma_noise = pm.HalfNormal("sigma", sigma=noise_prior[1])
        elif noise_prior[0] == 'HalfCauchy':
            sigma_noise = pm.HalfCauchy("sigma", beta=noise_prior[1])
        else:
            sigma_noise = pm.HalfNormal("sigma", sigma=1.0)

        mu_pred = intercept + sum(X[feat].values * betas[feat] for feat in features)

        y_obs = pm.Normal("y_obs", mu=mu_pred, sigma=sigma_noise, observed=y)

        try:
            trace = pm.sample(draws=draws, tune=tune, target_accept=target_accept, return_inferencedata=True, progressbar=False)
        except Exception as e:
            print(f'ERROR: {e}')
            return None

    return trace

def predict_using_posterior_regression(trace, X_new, feature_names):
    X = X_new[feature_names].values
    posterior_samples = {}
    for fname in feature_names:
        posterior_samples[fname] = trace.posterior[fname].stack(sample=("chain", "draw")).values

    intercept = trace.posterior["intercept"].stack(sample=("chain", "draw")).values
    intercept = intercept.reshape(1, -1)

    coefs = np.array([posterior_samples[f] for f in feature_names])
    predictions = X @ coefs + intercept

    predicted_mean = predictions.mean(axis=1)

    return predicted_mean
