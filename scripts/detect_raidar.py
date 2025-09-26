import os
import json
import numpy as np

from metrics import get_roc_metrics, get_precision_recall_metrics
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
from fuzzywuzzy import fuzz
from ImBD.dataset import CustomDatasetRewrite
import torch
from torch.utils.data import DataLoader, Subset
import argparse
import random

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def tokenize_and_normalize(sentence):
    # Tokenization and normalization
    return [word.lower().strip() for word in sentence.split()]

def extract_ngrams(tokens, n):
    # Extract n-grams from the list of tokens
    return [' '.join(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]

def common_elements(list1, list2):
    # Find common elements between two lists
    return set(list1) & set(list2)

def calculate_sentence_common(sentence1, sentence2):
    tokens1 = tokenize_and_normalize(sentence1)
    tokens2 = tokenize_and_normalize(sentence2)

    # Find common words
    common_words = common_elements(tokens1, tokens2)

    # Find common n-grams (let's say up to 3-grams for this example)
    common_ngrams = set()

    number_common_hierarchy = [len(list(common_words))]

    for n in range(2, 5):  # 2-grams to 3-grams
        ngrams1 = extract_ngrams(tokens1, n)
        ngrams2 = extract_ngrams(tokens2, n)
        common_ngrams = common_elements(ngrams1, ngrams2) 
        number_common_hierarchy.append(len(list(common_ngrams)))

    return number_common_hierarchy

def sum_for_list(a,b):
    return [aa+bb for aa, bb in zip(a,b)]

def get_data_stat(data_loader, human=True, verbose=False):

    data_stats = []
    total_len = len(data_loader)
    for idxx, each in enumerate(data_loader):
        if human:
            text = each[0][0]
            text_rewrite = [x[0] for x in each[2]]
        else:
            text = each[1][0]
            text_rewrite = [x[0] for x in each[3]]
        raw = tokenize_and_normalize(text)
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue
        else:
            if verbose:
                print(idxx, total_len)

        all_text = [text]
        all_text.extend(text_rewrite)

        statistic_res = {}
        ratio_fzwz = {}
        all_statistic_res = [0 for i in range(ngram_num)]
        cnt = 0
        whole_combined = ''
        for pp in range(len(all_text)):
            whole_combined += (' ' + all_text[pp])
            
            res = calculate_sentence_common(text, all_text[pp])
            statistic_res[pp] = res
            all_statistic_res = sum_for_list(all_statistic_res, res)

            ratio_fzwz[pp] = [fuzz.ratio(text, all_text[pp]), fuzz.token_set_ratio(text, all_text[pp])]
            cnt += 1
        
        each_stat = {}
        each_stat['input'] = text
        each_stat['fzwz_features'] = ratio_fzwz
        each_stat['common_features'] = statistic_res
        each_stat['avg_common_features'] = [a / cnt for a in all_statistic_res]
        each_stat['common_features_ori_vs_allcombined'] = calculate_sentence_common(text, whole_combined)

        data_stats.append(each_stat)

        if idxx == 400:
            break

    return data_stats

def get_feature_vec(input_json):
    all_list = []
    for idxx, each in enumerate(input_json):   
        try:
            raw = tokenize_and_normalize(each['input'])
            r_len = len(raw)*1.0
        except:
            import pdb; pdb.set_trace()
        each_data_fea  = []

        if r_len == 0:
            continue
        if len(raw) < cutoff_start or len(raw) > cutoff_end:
            continue

        each_data_fea = [ind_d / r_len for ind_d in each['avg_common_features']]
        for ek in each['common_features'].keys():
            each_data_fea.extend([ind_d / r_len for ind_d in each['common_features'][ek]])
        
        each_data_fea.extend([ind_d / r_len for ind_d in each['common_features_ori_vs_allcombined']])
        for ek in each['fzwz_features'].keys():
            each_data_fea.extend(each['fzwz_features'][ek])

        all_list.append(np.array(each_data_fea))

    all_list = np.vstack(all_list)
    return all_list

def train_classifier(human_stat, llm_stat):
    llm_all = get_feature_vec(llm_stat)
    human_all = get_feature_vec(human_stat)

    X_train = np.concatenate((human_all, llm_all), axis=0)
    y_train = np.concatenate((np.zeros(human_all.shape[0]), np.ones(llm_all.shape[0])), axis=0)

    # Neural network
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    clf = MLPClassifier(hidden_layer_sizes=(10,), max_iter=1000, activation='relu', solver='adam', random_state=42)
    clf.fit(X_train, y_train)
    return scaler, clf

def classifier_eval(scaler, clf, texts_stats):
    X_test = get_feature_vec(texts_stats)

    X_test = scaler.transform(X_test)
    prob_predict = clf.predict_proba(X_test)[:, 1]
    return prob_predict   
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dataset', type=str, default='exp_prompt/data/squad_claude-3-5-haiku_expand&exp_prompt/data/squad_claude-3-5-haiku_rewrite&exp_prompt/data/squad_claude-3-5-haiku_polish&exp_prompt/data/writing_claude-3-5-haiku_expand&exp_prompt/data/writing_claude-3-5-haiku_rewrite&exp_prompt/data/writing_claude-3-5-haiku_polish')
    parser.add_argument('--eval_dataset', type=str, default="./exp_prompt/data/xsum_claude-3-5-haiku_rewrite")
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_claude-3-5-haiku_rewrite")
    parser.add_argument('--cache_dir', type=str, default='../cache')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    name = 'raidar'
    ngram_num = 4
    cutoff_start = 0
    cutoff_end = 6000000

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    train_data = CustomDatasetRewrite(data_json_dir=args.train_dataset)   ### here, we simply use our rewrite dataset (of course, we can use rewrite-texts generated by RAIDAR)
    val_data = CustomDatasetRewrite(data_json_dir=args.eval_dataset)

    subset_indices = torch.randperm(len(train_data))
    train_subset = Subset(train_data, subset_indices)
    n_samples = len(val_data)
    val_data = Subset(train_data, torch.randperm(len(val_data)))

    train_loader = DataLoader(train_subset, batch_size=1, shuffle=False)
    eval_loader = DataLoader(val_data, batch_size=1, shuffle=False)

    human_stat = get_data_stat(train_loader, human=True)
    llm_stat = get_data_stat(train_loader, human=False)
    scaler, clf = train_classifier(human_stat, llm_stat)

    eval_human_stat = get_data_stat(eval_loader, human=True)
    eval_llm_stat = get_data_stat(eval_loader, human=False)
    human_pred = classifier_eval(scaler, clf, eval_human_stat)
    llm_pred = classifier_eval(scaler, clf, eval_llm_stat)
    
    print("-----------------------------------------------------------------")
    predictions = {'real': human_pred.tolist(), 'samples': llm_pred.tolist()}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name} ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')


