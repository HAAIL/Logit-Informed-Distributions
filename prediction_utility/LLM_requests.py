from openai import OpenAI
from tqdm import tqdm
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import torch.nn.functional as F

API_KEY = "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"
DeepSeek_API_KEY = 'sk-e02df72af50c4171b7b6d5a36eba0491'
models = {'gemma-3': 'google/gemma-3-27b-it', \
          'gemma-2': 'google/gemma-2-27b-it',
          'claude': 'anthropic/claude-3.7-sonnet',\
          'gpt-4o': 'openai/gpt-4o-2024-11-20',
          'qwen': 'qwen/qwen-2.5-72b-instruct',
          'llama-3-8b': 'meta-llama/Meta-Llama-3.1-8B',
          'gpt-2': 'openai-community/gpt2',
          'gpt-3.5': 'openai/gpt-3.5-turbo'}


def LLM_request(model, prompt, patient_record, examples = [], history = []):
    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=API_KEY,
    )
    #adding examples as user-model messages
    messages = []
    for record in history:
        messages.extend(
           [{"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": record[0]
                }
            ]
            }, {"role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": record[1]
                }
            ]
        }])
    for example in examples:
        # risk = "high" if example["pediction"] == 1 else "low"
        messages.extend(
           [{"role": "user",
            "content": [
                {
                    "type": "text",
                    "text": prompt + example["record"]
                }
            ]
            }, {"role": "assistant",
            "content": [
                {
                    "type": "text",
                    "text": f"### Prediction: {example['prediction']}. ###"
                }
            ]
        }])
    messages.append({
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": prompt + patient_record
            }
        ]
    })
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000
    )
    try:
        return completion.choices[0].message.content
    except Exception as e:
        print("completion error: ", completion)  
        return None
    
def default_label_maker(response):
    try:
        return response.split("### Prediction: ")[1].split(". ###")[0].strip()
    except:
        print("Error in extractrion for ", response)
        return None

def class_label_maker(response):
    res = default_label_maker(response)
    if res == "Recurrence":
        return 1
    elif res == "No Recurrence":
        return 0
    
def save_and_extract(read_path, save_path, prompt, examples, data_serialized, model, label_maker = default_label_maker):
    with open(f"LLM_predictions/{read_path}.txt", "w+") as f:
        for patient in tqdm(data_serialized):
            # rejection sampling
            res = ""
            while "### Prediction: " not in res:
                res = LLM_request(models[model], prompt, patient, examples)
            f.write(f'\"{res}\"\n,')

    with open(f"LLM_predictions/{read_path}.txt", "r") as f, open(f"LLM_predictions/{save_path}.txt", "w+") as f_out:
        for line in f.readlines():
            prediction = None
            if "###" in line:
                prediction = label_maker(line)
            if "\"None\"" in line:
                prediction = 0.0
            if prediction != None:
                f_out.write(str(prediction) + "\n")

def LLM_label(model, dataset_name, data_serialized, prediction_target):
    examples = [{"record": data_serialized[0], "prediction": "0"}, {"record": data_serialized[12], "prediction": "1"}]
    prompt = f"Predict the {prediction_target} for the following patient record. \
            Based on your reasoning, provide a final answer in the form of a sentence betweeen ### and ###, \
            filling the blank in this sentence with either \"0\" for negative or \"1\" for positive: Prediction: ______.\n Petient record: "

    save_and_extract(f"Labels/dummy", f"Labels/{model}_{dataset_name}", prompt, examples, data_serialized, model, default_label_maker)

def LLM_probability(model, dataset_name, data_serialized, prediction_target):
    examples = [{"record": data_serialized[0], "prediction": 0.2}, {"record":  data_serialized[12], "prediction": 0.8}]
    prompt = f"Predict the {prediction_target} for the following patient record. \
            Based on your reasoning, provide a final answer in the form of a sentence betweeen ### and ###, \
            filling the blanks in this sentence: Prediction: _____.\n Patient record: "

    save_and_extract(f'Probabilities/{model}_{dataset_name}', f'Probabilities/{model}_{dataset_name}', prompt, examples)

def pattern_extreactor(pattern, lines, expected_extraction_size):
    patterns_extracted = [re.search(pattern, line).group() for line in lines if re.search(pattern, line)]
    if len(patterns_extracted) != expected_extraction_size:
        return -1
    return patterns_extracted

def coeff_extractor(coeffs, feature_size):
    coeffs = [line.strip() for line in coeffs.split('\n')]
    coeff_matcher = r"[a-zA-Z_][a-zA-Z_0-9]*(?:_[a-zA-Z_]*\d*\.?\d*)*:\s*-?\d+\.?\d*"
    return pattern_extreactor(coeff_matcher, coeffs, feature_size)

def prior_extractor(priors, feature_size):
    priors = [line.strip() for line in priors.split('\n')]
    prior_matcher = r"[a-zA-Z_][a-zA-Z_0-9]*(?:_[a-zA-Z_]*\d*\.?\d*)*:\s*(Normal|Laplace),\s*-?\d+\.\d+,\s*-?\d+\.\d+"
    return pattern_extreactor(prior_matcher, priors, feature_size)

def important_feature_extractor(mifs, features, mif_size):
    mifs = [line.strip() for line in mifs.split('\n')]
    mifs_extracted = []
    for feature in features:
        for mif in mifs:
            if feature in mif:
                mifs_extracted.append(mif)
    if len(mifs_extracted) < mif_size:
        return -1
    return mifs_extracted


def LLM_feature_selection(model, out_path, features_serialized, prediction_task):
    prompt = f"The following are the features and description of a classification problem. \
          The features are provided in the following format: <feature name>: <feature description>. \
          The task is to {prediction_task}. \
          Identify the most important features for this task and provide a list of at most 15 most important features."
    prompt += features_serialized
    prompt += "Output the features in the following format: \
            feature_name_1\nfeature_name_2\n...."
    important_features = LLM_request(models[model], prompt, "")
    if important_features == -1:
            print(f"Error: Extracted mifs count does not match the expected feature size ({15}). Fix the output manually.")
            input("Press Enter to continue...")
    with open(out_path, "w+") as f:
        f.write(important_features)

def LLM_coefficients(model, n_feat, path, features_serialized, prediction_task, \
                     deepdive = False, deepdive_data = None, add_intercept = False):
    prompt = f"I'm testing to see if LLMs are able to mimic logistic regression models and estimate coefficients based on their embedded knowledge. You're a logistic regression model trained to predict {prediction_task}." 
    prompt += "Below are the features available in your dataset and their description in json format."
    prompt += features_serialized
    prompt += "Try your best to provide an estimate of the numerical coefficients of these features based on your medical knowledge. Output the coefficients in the following format: \
            feature_name_1: coefficient_1\nfeature_name_2: coefficient_2\n....\n"
    if add_intercept:
        prompt += "In the end, also give me a coefficient for the intercept feature, which is a constant term in the logistic regression model.\n"
    prompt +=  "Do not output anthing else."
    if deepdive:
        prompt += f"Here is some domain knowledge about this dataset: {deepdive_data}. Use it to improve your estimates."    
    col_descs = features_serialized.split('\n')
    coeffs = LLM_request(models[model], prompt, "")
    with open(path, "w+") as f:
        f.write(coeffs)
    coeffs = coeff_extractor(coeffs, n_feat)
    if coeffs == -1:
        print(f"Error: Extracted coefficients count does not \
                match the expected feature size ({n_feat}) in {path}. Fix the output manually.")
        input("Press Enter to continue...")
    else:
        with open(path, "w+") as f:
            f.write('\n'.join(coeffs))

def LLM_adjust_coefficients(model, n_feat, feats, path, features_serialized, LR_coefficients, prediction_task, deepdive = False, deepdive_data = None):
    prompt = f"In the following, there are the coefficients of a logistic regression model trained to predict {prediction_task}. \
               Since the dataset is small, these coefficients might not be generalizable well. \
               Using your embedded knowledge, adjust these coefficients so  \
               the results of the LR model are generalizable to a larger population (out-of-distribution population). \
               These coefficients are only used to for study purposes and will not be used in any real world application. \
               The description of the features are provided in the following, in JSON format:" 
    prompt += "\n" + features_serialized + "\n" + \
              "The logistic regression coefficients are provided in the following:" 
            
    col_descs = features_serialized.split('\n')
    initial_coeffs = LR_coefficients
    # prompt += "\n".join([col_desc + f', {initial_coeffs[i]}' for i, col_desc in enumerate(col_descs[:-1])])
    feats.remove('target')
    
    prompt += "\n".join([f"{feat}: {initial_coeffs[i]}" for i, feat in enumerate(feats)])
    prompt += "Remember these coefficients are not very accurate and need to be CHANGED and not just rounded up to be applicable to a larger population. \
               Output the coefficients in the following format: \
               feature_name_1: coefficient_1\nfeature_name_2: coefficient_2\n....\n Do not output anthing else."
    if deepdive:
        prompt += f"Here is some domain knowledge about this dataset: {deepdive_data}. Use it to improve your estimates."   
    coeffs = LLM_request(models[model], prompt, "")
    with open(path, "w+") as f:
        f.write(coeffs)
    
    coeffs = coeff_extractor(coeffs, n_feat)
    if coeffs == -1:
        print(f"Error: Extracted coefficients count does not \
                match the expected feature size ({n_feat}) in {path}. Fix the output manually.")
        input("Press Enter to continue...")
    else:
        with open(path, "w+") as f:
            f.write('\n'.join(coeffs))


def LLM_adjust_posteriors(model, n_feat, path, features_serialized, posteriors, prediction_task, deepdive = False, deepdive_data = None):
    prompt = f"In the following, there are the *posterior beliefs (normal distributions)* about the coefficients of a logistic regression model trained to predict {prediction_task}. \
               Since the dataset is small, these distributions might not be generalizable well. \
               Using your embedded knowledge, adjust these distributions so  \
               the results of the LR model are generalizable to a larger population (out-of-distribution population). \
               These coefficients are only used to for study purposes and will not be used in any real world application. \
               The description of the features are provided in the following, in JSON format:" 
    prompt += "\n" + features_serialized + "\n" + \
              "The logistic regression coefficient distributions are provided in the following:" 
            
    initial_coeffs = posteriors
    # prompt += "\n".join([col_desc + f', {initial_coeffs[i]}' for i, col_desc in enumerate(col_descs[:-1])])
    prompt += "Remember these distributions are not very accurate and need to be CHANGED and not just rounded up to be applicable to a larger population. \
               Output the coefficients in the following format: \
               <feature_name>: <Normal>,mean,scale\n \
               <feature_name>: <Normal>,mean,scale \
               example: age: Normal,0.5,0.1\n\
               Do not output anything else."
    if deepdive:
        prompt += f"Here is some domain knowledge about this dataset: {deepdive_data}. Use it to improve your estimates."   
    coeffs = LLM_request(models[model], prompt, "")
    with open(path, "w+") as f:
        f.write(coeffs)
    coeffs = coeff_extractor(coeffs, n_feat)
    if coeffs == -1:
        print(f"Error: Extracted coefficients count does not \
                match the expected feature size ({n_feat}). Fix the output manually.")
        input("Press Enter to continue...")
    else:
        with open(path, "w+") as f:
            f.write('\n'.join(coeffs))

# def LLM_priors_feat(model, n_feat, path, features_serialized, prediction_task, deepdive = False, deepdive_data = None):
#     prompt = f"I want to use LLMs to extract informative priors for predicting {prediction_task} to perform bayesian inference. \
#                Below is one of the features available in my dataset and its description in json format. The feature values are z-scored. So your prior mean, scale/std should be in the range of -1 to 1.\n"
#     prompt += features_serialized
#     # prompt += f"Based on your embedded knowledge about the topic, output a prior distribution demonstrating the feature's coefficient in a logistic regression model that was trained on a dataset for predicting {prediction_task}.\
#     #            Use \"Normal\" distribution for numerical and \"Laplace\" distribution for categorical features.\
#     #            Do not use typical distributions like mean=0,std=1. However, your priors shouldn't reflect strong beliefs. \
#     #            Use the following format for outputs: \
#     #            <feature_name>: <Normal/Laplace>,mean,scale/std\n \
#     #            example: age: Normal,0.5,0.1\n age: Laplace,50,10\n\
#     #            Do not output anything else."
#     prompt += f"Based on your embedded knowledge about the topic, output a prior distribution demonstrating the feature's coefficient in a logistic regression model that was trained on a dataset for predicting {prediction_task}.\
#                Use \"Normal\" distribution for all features.\
#                Do not use typical distributions like mean=0,std=1. However, your priors shouldn't reflect strong beliefs. \
#                Use the following format for outputs: \
#                <feature_name>: <Normal>,mean,scale\n \
#                example: age: Normal,0.5,0.1\n\
#                Do not output anything else."
#     if deepdive:
#         prompt += f"Here is some domain knowledge about this dataset: {deepdive_data}. Use it to improve your estimates."    
#     coeffs = LLM_request(models[model], prompt, "")
#     with open(path, "a+") as f:
#         f.write(coeffs)
#     coeffs = prior_extractor(coeffs, n_feat)
#     if coeffs == -1:
#         print(f"Error: Extracted coefficients count does not \
#                 match the expected feature size ({n_feat}). Fix the output manually.")
#         input("Press Enter to continue...")
#     else: 
#         with open(path, "w+") as f:
#             f.write('\n'.join(coeffs))

def LLM_priors(path, features_serialized, prediction_task, experiment_setup, n_feat, history):
    distrib_shift = experiment_setup['additional_info']['shift_explained'][1]
    deepdive_data = experiment_setup['additional_info']['deep_dive'][1]
    deepdive = experiment_setup['additional_info']['deep_dive'][0]
    add_intercept = experiment_setup['additional_info']['LLM_intercept']
    model = experiment_setup['model_name']

    prompt = f"I want to use LLMs to extract informative priors for predicting {prediction_task} to perform bayesian inference. \
               Below are the features available in my dataset and some description about the train set in json format. The feature values are z-scored.\n"
    prompt += features_serialized
    prompt += f"The test has a distributions shift: {distrib_shift} Based on your embedded knowledge about the topic, for each feature, \
               output a prior distribution demonstrating the feature's coefficient in a logistic regression model that was trained on a dataset for predicting {prediction_task}.\
               Use \"Normal\" distribution for all features.\
               Do not use typical distributions like mean=0,std=1. However, your priors shouldn't reflect strong beliefs. \
               Use the following format for outputs: \
               <feature_name>: <Normal>,mean,std\n \
               <feature_name>: <Normal>,mean,std \
               example: age: Normal,0.5,0.1\n"
    if add_intercept:
        prompt += "In the end, also give me a prior for the intercept feature, which is a constant term in the logistic regression model. "     
    prompt += "Do not output anything else."
    if deepdive:
        prompt += f"Here is some domain knowledge about this dataset: {deepdive_data}. Use it to improve your estimates."    
    if history != '':
        prompt_history = [(
            prompt, history
        )]
        prompt = 'The coefficients you provided do not result in a high auc score. Please adjust them to improve the auc score.'
        coeffs = LLM_request(models[model], prompt, "", history=prompt_history)
    else:
        coeffs = LLM_request(models[model], prompt, "")
    with open(path, "w+") as f:
        f.write(coeffs)
    coeffs = prior_extractor(coeffs, n_feat)
    if coeffs == -1:
        print(f"Error: Extracted coefficients count does not \
                match the expected feature size ({n_feat}). Fix the output manually.")
        input("Press Enter to continue...")
    else: 
        with open(path, "w+") as f:
            f.write('\n'.join(coeffs))

def LLM_priors_regression(path, features_serialized, prediction_task, experiment_setup, n_feat, history, alpha=0.2, beta=2.0, num_sentences=10, std_method='variance'):
    model = experiment_setup['model_name']

    prompt = f"I want to use LLMs to extract informative priors for predicting {prediction_task} using linear regression. \
               Below are the features available in my dataset in json format. The feature values are z-scored.\n"
    prompt += features_serialized
    prompt += f"Based on your embedded knowledge about the topic, for each feature, \
               output a prior distribution demonstrating the feature's coefficient in a linear regression model for predicting {prediction_task}.\
               Use \"Normal\" distribution for all features.\
               Do not use typical distributions like mean=0,std=1. However, your priors shouldn't reflect strong beliefs. \
               Use the following format for outputs: \
               <feature_name>: <Normal>,mean,std\n \
               <feature_name>: <Normal>,mean,std \
               example: age: Normal,0.5,0.1\n"
    prompt += "In the end, also give me a prior for the intercept feature, which is a constant term in the linear regression model. "
    prompt += "Do not output anything else."

    if history != '':
        prompt_history = [(
            prompt, history
        )]
        prompt = 'The coefficients you provided do not result in a low MSE. Please adjust them to improve the MSE.'
        coeffs = LLM_request(models[model], prompt, "", history=prompt_history)
    else:
        coeffs = LLM_request(models[model], prompt, "")
    with open(path, "w+") as f:
        f.write(coeffs)
    coeffs = prior_extractor(coeffs, n_feat)
    if coeffs == -1:
        print(f"Error: Extracted coefficients count does not \
                match the expected feature size ({n_feat}). Fix the output manually.")
        input("Press Enter to continue...")
    else:
        with open(path, "w+") as f:
            f.write('\n'.join(coeffs))

def LLM_dependencies(model, dataset_name, features_serialized, prediction_task):
    prompt = f'We have a set of features for predicting {prediction_task}. We want to construct a Bayesian network for prediction.\
            However, our domain knowlege is limited. Using your knowledge about this field, suggest a list of \
            dependencies that can be used for the Byesian network. Below are the features and their descriptions:\n'

    prompt += features_serialized
    prompt += 'Your output should be in the following formant: \
                <feature_1>: <parent_feature1>,<parent_feature2>...\nThere should be no circular dependencies.'
    dependencies = LLM_request(models[model], prompt, "")
    with open(f"LLM_predictions/Dependencies/{model}_{dataset_name}_fs.txt", "w+") as f:
        f.write(dependencies)
    return dependencies

tokenizer, model = None, None
def LLM_load_model(model_name):
    global tokenizer, model
    model_id = models[model_name]
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

def LLM_coeff_logits(text_prefix, target_str):
    global tokenizer, model

    full_text = text_prefix + target_str
    full_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    prefix_inputs = tokenizer(text_prefix, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**full_inputs)
        logits = outputs.logits  # [1, seq_len, vocab_size]

    input_ids = full_inputs["input_ids"][0]
    prefix_len = prefix_inputs["input_ids"].shape[1]

    log_probs = []
    for i in range(prefix_len, len(input_ids)):
        token_id = input_ids[i]
        logits_prev = logits[0, i - 1]
        log_prob = F.log_softmax(logits_prev, dim=-1)[token_id].item()
        log_probs.append(log_prob)

    total_log_prob = sum(log_probs)
    return torch.exp(torch.tensor(total_log_prob)).item(), total_log_prob

def llm_single_token_logprob(sentence, target_token, info = None):
    global tokenizer, model

    # Tokenize the input and move to model's device
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)

    # Run model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # [vocab_size] at last position

    # Tokenize the target token and move to same device
    target_token_id = tokenizer.encode(target_token, add_special_tokens=False)[0]

    # Compute probability and log probability
    probs = F.softmax(logits, dim=-1)
    prob = probs[target_token_id].item()
    log_prob = torch.log(probs[target_token_id]).item()

    return prob, log_prob

def llm_top_k_tokens(sentence, k=100):
    """
    Get the top-k most likely next tokens and their log probabilities.

    Args:
        sentence: The input prompt
        k: Number of top tokens to return (default 100)

    Returns:
        List of tuples: [(token_string, log_prob), ...]
    """
    global tokenizer, model

    # Tokenize the input and move to model's device
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)

    # Run model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # [vocab_size] at last position

    # Compute probabilities
    probs = F.softmax(logits, dim=-1)
    log_probs = torch.log(probs)

    # Get top-k
    top_k_probs, top_k_indices = torch.topk(probs, k)
    top_k_log_probs = torch.log(top_k_probs)

    # Decode tokens
    results = []
    for idx, log_prob in zip(top_k_indices, top_k_log_probs):
        token_str = tokenizer.decode([idx.item()]).strip()
        results.append((token_str, log_prob.item()))

    return results

def llm_multiclass_token_logprob(sentence, target_tokens):
    """
    Multi-class version of llm_single_token_logprob.

    Args:
        sentence: The input prompt
        target_tokens: List of class label tokens (e.g., ["class_0", "class_1", "class_2"])

    Returns:
        probs: List of probabilities for each class token
        log_probs: List of log probabilities for each class token
    """
    global tokenizer, model

    # Tokenize the input and move to model's device
    input_ids = tokenizer.encode(sentence, return_tensors="pt").to(model.device)

    # Run model
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0, -1, :]  # [vocab_size] at last position

    # Get probabilities
    all_probs = F.softmax(logits, dim=-1)

    # Get probabilities for each target token
    probs = []
    log_probs = []
    for token in target_tokens:
        token_id = tokenizer.encode(token, add_special_tokens=False)[0]
        prob = all_probs[token_id].item()
        log_prob = torch.log(all_probs[token_id]).item()
        probs.append(prob)
        log_probs.append(log_prob)

    return probs, log_probs
