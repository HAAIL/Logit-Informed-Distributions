import numpy as np
from scipy.stats import entropy
import os

def calc_KL(out_path, dataset = None):
    print(f"Processing dataset: {dataset}")
    # Get the directory of the script that's calling this function (Model-Sara)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    indistrib_path = os.path.join(base_dir, f"LLM_predictions/Posteriors/{dataset}_true_posterior.npz")

    if not os.path.exists(indistrib_path):
        print(f"Reference file {indistrib_path} not found. Returning KL=0.")
        return 0

    post_indistrib = np.load(indistrib_path)
    post_ood = np.load(out_path)

    features = list(post_indistrib.keys())
    kls = []
    for var in features:
        if var == 'intercept':
            continue
        trace_p = post_indistrib[var].flatten()
        try:
            trace_q = post_ood[var].flatten()
        except:
            print(f"Variable {var} not found in OOD posteriors.")
            continue
        
        all_samples = np.concatenate([trace_p, trace_q])
        bins = np.histogram_bin_edges(all_samples, bins=100)
        p_hist, _ = np.histogram(trace_p, bins=bins, density=True)
        q_hist, _ = np.histogram(trace_q, bins=bins, density=True)
        epsilon = 1e-12
        p_hist += epsilon
        q_hist += epsilon
        p_hist /= p_hist.sum()
        q_hist /= q_hist.sum()
        # Fixed: Calculate KL(OOD || true) instead of KL(true || OOD)
        # This measures how much the OOD posterior diverges from truth
        kls.append(entropy(q_hist, p_hist))

    mean_KL = np.mean(kls) if kls else 0
    print(f"Mean KL Divergence for {dataset}: {mean_KL}")
    return mean_KL