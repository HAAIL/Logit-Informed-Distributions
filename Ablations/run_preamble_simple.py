"""
Simple preamble ablation - modifies templates directly.
"""
import sys
import pandas as pd
from prior_prediction import *
from contextual_preambles import get_preamble

# Monkey-patch the template function to add preambles
original_get_templates = get_templates_and_tokens

def get_templates_and_tokens_with_preamble(dataset_name, task_description, feature_name=None):
    """Modified to add preambles if feature_name provided."""
    templates, tokens = original_get_templates(dataset_name, task_description)
    
    # Add preamble if we have a feature name
    if feature_name:
        preamble = get_preamble(dataset_name, feature_name)
        if preamble:
            templates = [f"{preamble} {template}" for template in templates]
    
    return templates, tokens

# Monkey-patch globally
import dataset_specific_templates
dataset_specific_templates.get_templates_and_tokens = get_templates_and_tokens_with_preamble

# Now just run the normal script with preambles enabled
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', required=True)
    parser.add_argument('--model', type=str, default='gemma-2')
    parser.add_argument('--num_sentences', type=int, default=10)
    parser.add_argument('--use_ood_split', action='store_true')
    args = parser.parse_args()
    
    # This won't work directly, we need to modify extract_binary_priors
    print("ERROR: This approach requires modifying prior_prediction.py")
    print("Please run preamble experiments manually by modifying dataset_specific_templates.py")
