"""
Modified version of prior_prediction.py with preamble support.
"""
import numpy as np
import pandas as pd
import pymc as pm
from tqdm import tqdm
import argparse

# Import from original
from prior_prediction import (
    llm_single_token_logprob, load_data_with_target,
    std_method_variance_of_differences, std_method_entropy, std_method_logprob_difference
)
from prediction_utility.LLM_requests import LLM_load_model
from contextual_preambles import get_preamble
from dataset_specific_templates_with_preamble import get_templates_and_tokens_with_preamble


def extract_binary_priors_with_preamble(X_test, data_dictionary, dataset_name, task_description, 
                                        num_sentences=10, use_preamble=False):
    """Extract priors with optional preambles."""
    coeffs_prior = {}

    for col in tqdm(X_test.columns, desc="Extracting binary priors"):
        dict_row = data_dictionary[data_dictionary['name'] == col]
        if dict_row.empty:
            continue

        feat_desc = dict_row.iloc[0]['f_nl']
        preamble = get_preamble(dataset_name, col) if use_preamble else ""

        templates, (token_pos, token_neg) = get_templates_and_tokens_with_preamble(
            dataset_name, task_description, preamble=preamble
        )
        
        sentences = [template.format(feat_desc, task_description) for template in templates]
        np.random.shuffle(sentences)
        sentences = sentences[:num_sentences]

        differences = []
        for sentence in sentences:
            prob_pos, _ = llm_single_token_logprob(sentence, token_pos)
            prob_neg, _ = llm_single_token_logprob(sentence, token_neg)

            total_prob = prob_pos + prob_neg
            p_pos_norm = prob_pos / total_prob
            p_pos_norm = np.clip(p_pos_norm, 1e-10, 1 - 1e-10)
            differences.append(np.log(p_pos_norm / (1 - p_pos_norm)))

        mean_val = np.mean(differences)
        std_val = std_method_variance_of_differences(differences, 1.0, 1.0)
        coeffs_prior[col] = ('Normal', mean_val, std_val)

    return coeffs_prior


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--model', type=str, default='gemma-2')
    parser.add_argument('--num_sentences', type=int, default=10)
    parser.add_argument('--use_ood_split', action='store_true')
    parser.add_argument('--use_preamble', action='store_true')
    args = parser.parse_args()

    # Load LLM model once
    print(f"Loading model: {args.model}")
    LLM_load_model(args.model)

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Running {dataset_name} with preamble={args.use_preamble}")
        print(f"{'='*60}")

        # Load data
        data_dictionary = pd.read_csv(f"Data/{dataset_name}/{dataset_name}_dict.csv")
        train, test, _ = load_data_with_target(dataset_name, data_dictionary, args.use_ood_split)

        X_train = train.drop(columns=['target']).fillna(0)
        y_train = train['target'].values
        X_test = test.drop(columns=['target']).fillna(0)
        y_test = test['target'].values

        try:
            target_row = data_dictionary[data_dictionary['target'] == 'T'].iloc[0]
            task_description = target_row['f_nl']
        except:
            task_description = "target"

        # Standardization
        X_train = (X_train - X_train.mean()) / X_train.std().replace(0, 1)
        X_test = (X_test - X_test.mean()) / X_test.std().replace(0, 1)
        X_train, X_test = X_train.fillna(0), X_test.fillna(0)

        # Adjust labels
        y_train = y_train - 1 if y_train.min() == 1 else y_train
        y_test = y_test - 1 if y_test.min() == 1 else y_test

        # Extract priors
        print("Extracting priors...")
        coeffs_prior = extract_binary_priors_with_preamble(
            X_test, data_dictionary, dataset_name, task_description,
            num_sentences=args.num_sentences, use_preamble=args.use_preamble
        )

        # Bayesian logistic regression
        print("Computing posterior...")
        with pm.Model() as model:
            coeffs = {col: pm.Normal(col, mu=m, sigma=s) 
                     for col, (_, m, s) in coeffs_prior.items()}
            intercept = pm.Normal('intercept', mu=0, sigma=1)

            logit_p = intercept
            for i, col in enumerate(X_train.columns):
                if col in coeffs:
                    logit_p += coeffs[col] * X_train.values[:, i]

            p = pm.math.sigmoid(logit_p)
            y_obs = pm.Bernoulli('y_obs', p=p, observed=y_train)

            trace = pm.sample(1000, tune=1000, chains=4, target_accept=0.95, 
                            random_seed=42, progressbar=False)

        # Predictions
        print("Making predictions...")
        # Get all posterior samples flattened (from chains and draws)
        intercept_samples = trace.posterior['intercept'].values.flatten()
        n_samples = len(intercept_samples)
        n_test = len(X_test)

        # Initialize logits for all test points and all samples
        logit_test = np.tile(intercept_samples.reshape(-1, 1), (1, n_test))

        for i, col in enumerate(X_test.columns):
            if col in coeffs:
                coeff_samples = trace.posterior[col].values.flatten().reshape(-1, 1)
                logit_test += coeff_samples * X_test.values[:, i].reshape(1, -1)

        y_pred_prob = (1 / (1 + np.exp(-logit_test))).mean(axis=0)

        # Evaluate
        from sklearn.metrics import roc_auc_score
        test_auc = roc_auc_score(y_test, y_pred_prob)
        print(f"\nResults:")
        print(f"  Test AUC:  {test_auc:.4f}")
