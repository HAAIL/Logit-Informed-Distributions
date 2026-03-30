"""
Ablation study 1: Test different feature description styles.

Tests three grammatical variants:
- Variant 1: Concise/terse (baseline)
- Variant 2: Descriptive/verbose  
- Variant 3: Clinical/formal

Datasets: stroke, blood, heart_failure, adult
"""
import subprocess
import sys
import pandas as pd
from feature_synonym_descriptions import FEATURE_DESCRIPTIONS

datasets = ['stroke', 'blood', 'heart_failure', 'adult']
variants = ['variant1', 'variant2', 'variant3']
model = 'gemma-2'
num_sentences = 10
seed = 42

results = []

for dataset in datasets:
    for variant in variants:
        print(f"\n{'='*80}")
        print(f"Running {dataset} with {variant}")
        print(f"{'='*80}\n")
        
        # Create temporary dictionary file with variant descriptions
        dict_file = f'Data/{dataset}/{dataset}_dict.csv'
        df_dict = pd.read_csv(dict_file)
        
        # Update f_nl column with variant descriptions
        variant_descs = FEATURE_DESCRIPTIONS[dataset][variant]
        for feature, desc in variant_descs.items():
            if feature in df_dict['name'].values:
                df_dict.loc[df_dict['name'] == feature, 'f_nl'] = desc
        
        # Save temporary variant dictionary
        temp_dict_file = f'Data/{dataset}/{dataset}_dict_{variant}.csv'
        df_dict.to_csv(temp_dict_file, index=False)
        
        # Modify prior_prediction.py temporarily to use variant dict
        # Run experiment
        cmd = [
            'python', 'prior_prediction.py',
            '--datasets', dataset,
            '--model', model,
            '--num_sentences', str(num_sentences),
            '--use_ood_split',
            '--dict_suffix', f'_{variant}'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            # Parse results from output
            lines = result.stdout.split('\n')
            for line in lines:
                if 'Test AUC:' in line or 'Test MSE:' in line:
                    metric_value = float(line.split(':')[-1].strip())
                    results.append({
                        'dataset': dataset,
                        'variant': variant,
                        'metric': 'AUC' if 'AUC' in line else 'MSE',
                        'value': metric_value
                    })
                    print(f"Result: {metric_value}")
            
        except subprocess.TimeoutExpired:
            print(f"TIMEOUT for {dataset} {variant}")
            results.append({
                'dataset': dataset,
                'variant': variant,
                'metric': 'TIMEOUT',
                'value': None
            })
        except Exception as e:
            print(f"ERROR for {dataset} {variant}: {e}")
            results.append({
                'dataset': dataset,
                'variant': variant,
                'metric': 'ERROR',
                'value': str(e)
            })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv('feature_description_ablation_results.csv', index=False)
print(f"\n{'='*80}")
print("RESULTS SAVED TO: feature_description_ablation_results.csv")
print(f"{'='*80}\n")
print(results_df)
