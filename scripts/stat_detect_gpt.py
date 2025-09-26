import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import tqdm
import argparse
import json
from data_builder import load_data
from model import load_tokenizer, load_model
from metrics import get_roc_metrics, get_precision_recall_metrics, get_roc_metrics_multi
from nuisance_func import BSplineTwoSample, ShiftLearner
from nuisance_func_human import BSplineTheory
from utils import load_training_data, separated_string
from statistics import get_martingale_stat, get_classification_stat, get_cusum_supstat, get_cusum_infstat, get_variance_stat, get_meanlb_stat, get_appro_variance_stat, get_distr_stat
import json
import time
from utils import GpuMem

def experiment(args):
    # load model
    scoring_tokenizer = load_tokenizer(args.scoring_model_name, args.cache_dir)
    scoring_model = load_model(args.scoring_model_name, args.device, args.cache_dir)
    scoring_model.eval()
    if args.sampling_model_name != args.scoring_model_name:
        sampling_tokenizer = load_tokenizer(args.sampling_model_name, args.cache_dir)
        sampling_model = load_model(args.sampling_model_name, args.device, args.cache_dir)
        sampling_model.eval()
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # evaluate criterion
    if args.discrepancy == 'martingale':
        name = "martingale"
        criterion_fn = get_martingale_stat
    elif args.discrepancy == 'classification':
        name = "classification"
        criterion_fn = get_classification_stat
    elif args.discrepancy == 'variance':
        name = "variance"
        criterion_fn = get_variance_stat
    elif args.discrepancy == 'bivariance':
        name = "bivariance"
        criterion_fn = get_variance_stat
    elif args.discrepancy == 'meanlb':
        name = "meanlb"
        criterion_fn = get_meanlb_stat
    elif args.discrepancy == 'appro_variance':
        name = "appro_variance"
        criterion_fn = get_appro_variance_stat
    elif args.discrepancy == 'distr':
        name = "distr"
        criterion_fn = get_distr_stat
    elif args.discrepancy == 'cusumsup':
        name = "cusumsup"
        criterion_fn = get_cusum_supstat
    elif args.discrepancy == 'cusuminf':
        name = "cusuminf"
        criterion_fn = get_cusum_infstat

    # w function
    start = time.perf_counter()
    tracker = GpuMem()
    if args.w_func == 'identity':
        w_func = nn.Identity()
        beta = None
    else:
        bspline_args = args.config
        ## load training data
        print(f"Datasets for learning BSpline: {args.train_dataset}")
        with tracker:
            train_data = load_training_data(args.train_dataset)
            if args.num_subsample > 0:
                args.num_subsample = min(args.num_subsample, len(train_data['original']))
                train_data['original'] = train_data['original'][:args.num_subsample]
                train_data['sampled'] = train_data['sampled'][:args.num_subsample]
            human_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['original']]
            if args.w_func == 'bspline' or args.w_func == 'bspline_theory_constrained':
                machine_token_list = [scoring_tokenizer(x, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device) for x in train_data['sampled']]
            
            if args.w_func == 'bspline':
                w_func = BSplineTwoSample(bspline_args, args.device)
                w_func.fit(human_token_list, machine_token_list, scoring_model, args)
            elif args.w_func == 'bspline_theory':
                w_func = BSplineTheory(bspline_args, machine_text=False)
                w_func.fit(human_token_list, None, scoring_model, args)
            elif args.w_func == 'bspline_theory_constrained':
                w_func = BSplineTheory(bspline_args, machine_text=True)
                w_func.fit(human_token_list, machine_token_list, scoring_model, args)
        beta = w_func.beta_hat.detach().cpu().tolist()
    pre_time = time.perf_counter() - start
    pre_memory = tracker.memory_usage()
    
    if args.debias:
        print(f"Datasets for learning bias: {args.train_dataset}")
        train_data = load_training_data(args.train_dataset)
        shift_learner = ShiftLearner()
        shift_learner.fit(train_data, scoring_tokenizer, scoring_model, w_func, args)
        shift_value = shift_learner.c_hat
    else:
        shift_value = torch.zeros(1).to(args.device)
    
    if args.discrepancy == 'distr':
        tokenized = scoring_tokenizer(data["sampled"][1], return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        if args.burn_in < 1.0:
            burn_in_num = int(args.burn_in * labels.size(-1))
        else:
            burn_in_num = int(args.burn_in)
        logits_score = scoring_model(**tokenized).logits[:, :-1]
        logits_ref = logits_score
        ref_distr = criterion_fn(logits_ref, logits_score, labels, burn_in_num, w_func, None)
        shift_value = ref_distr
    
    include_revised_text = 'revised' in list(data.keys())
    
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    eval_time_list = []
    eval_memory_list = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        # original text
        start = time.perf_counter()
        with tracker:
            tokenized = scoring_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            if args.burn_in < 1.0:
                burn_in_num = int(args.burn_in * labels.size(-1))
            else:
                burn_in_num = int(args.burn_in)
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = sampling_tokenizer(original_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = sampling_model(**tokenized).logits[:, :-1]
                original_crit = criterion_fn(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value)
        eval_time_list.append(time.perf_counter() - start)
        eval_memory_list.append(tracker.memory_usage())
        # sampled text
        tokenized = scoring_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
        labels = tokenized.input_ids[:, 1:]
        if args.burn_in < 1.0:
            burn_in_num = int(args.burn_in * labels.size(-1))
        else:
            burn_in_num = int(args.burn_in)
        with torch.no_grad():
            logits_score = scoring_model(**tokenized).logits[:, :-1]
            if args.sampling_model_name == args.scoring_model_name:
                logits_ref = logits_score
            else:
                tokenized = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                logits_ref = sampling_model(**tokenized).logits[:, :-1]
            sampled_crit = criterion_fn(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
        # print(f"Real mean: {original_crit:.2f}, Samples mean/std: {sampled_crit:.2f}")
        # refined text
        if include_revised_text:
            revised_text = data["revised"][idx]
            tokenized = scoring_tokenizer(revised_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
            labels = tokenized.input_ids[:, 1:]
            with torch.no_grad():
                logits_score = scoring_model(**tokenized).logits[:, :-1]
                if args.sampling_model_name == args.scoring_model_name:
                    logits_ref = logits_score
                else:
                    tokenized = sampling_tokenizer(sampled_text, return_tensors="pt", padding=True, return_token_type_ids=False).to(args.device)
                    assert torch.all(tokenized.input_ids[:, 1:] == labels), "Tokenizer is mismatch."
                    logits_ref = sampling_model(**tokenized).logits[:, :-1]
                revised_crit = criterion_fn(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value)
            results[-1]["revised"] = revised_text
            results[-1]["revised_crit"] = revised_crit
    eval_time = np.mean(np.array([eval_time_list]))
    eval_memory = np.mean(np.array([eval_memory_list]))
    
    # results
    if args.debias:
        results_file = f'{args.output_file}.{name}.{args.w_func}.debias.json'
    else:
        results_file = f'{args.output_file}.{name}.{args.w_func}.json'
    if not include_revised_text:
        # compute prediction scores for real/sampled passages
        predictions = {'real': [x["original_crit"] for x in results],
                    'samples': [x["sampled_crit"] for x in results]}
        print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
        fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
        p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")

        results = { 'name': f'{name}_threshold',
                    'info': {'n_samples': n_samples},
                    'predictions': predictions,
                    'raw_results': results,
                    'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                    'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                    'loss': 1 - pr_auc, 
                    'beta': beta, 
                    'bias': shift_value.detach().cpu().tolist(),
                    'compute_info': {'pre_time': pre_time, 'eval_time': eval_time, 
                                     'pre_memory': pre_memory, 'eval_memory': eval_memory,}
                    }
    else:
        predictions = {'real': [x["original_crit"] for x in results],
                       'revised': [x["revised_crit"] for x in results],
                       'samples': [x["sampled_crit"] for x in results]}
        print(f"Real mean/std: {np.mean(predictions['real'], axis=0):.2f}/{np.std(predictions['real'], axis=0):.2f}, Revised mean/std: {np.mean(predictions['revised'], axis=0):.2f}/{np.std(predictions['revised'], axis=0):.2f}, Samples mean/std: {np.mean(predictions['samples'], axis=0):.2f}/{np.std(predictions['samples'], axis=0):.2f}")
        roc_auc = get_roc_metrics_multi(predictions['real'], predictions['revised'], predictions['samples'])
        print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}")
        results = { 'name': f'{name}_threshold',
                    'info': {'n_samples': n_samples},
                    'predictions': predictions,
                    'raw_results': results,
                    'metrics': {'roc_auc': roc_auc},
                    'loss': 1 - roc_auc,
                    'compute_info': {'pre_time': pre_time, 'eval_time': eval_time, 
                                     'pre_memory': pre_memory, 'eval_memory': eval_memory,}
                  }
        
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_main/results/writing_qwen-7b")  # output file prefix
    parser.add_argument('--dataset', type=str, default="writing")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/writing_qwen-7b")
    parser.add_argument('--train_dataset', type=separated_string, default="./exp_main/data/xsum_qwen-7b_0&./exp_main/data/writing_qwen-7b")
    parser.add_argument('--num_subsample', type=int, default=-1)
    parser.add_argument('--sampling_model_name', type=str, default="qwen-7b")
    parser.add_argument('--scoring_model_name', type=str, default="qwen-7b")
    parser.add_argument('--discrepancy', type=str, default='classification', choices=['classification', 'martingale', 'cusumsup', 'cusuminf', 'variance', 'bivariance', 'appro_variance', 'meanlb', 'distr'])
    parser.add_argument('--burn_in', type=float, default=0.0)
    parser.add_argument('--w_func', type=str, default='bspline', choices=['identity', 'bspline', 'bspline_theory', 'bspline_theory_constrained'])
    parser.add_argument("--config", type=json.loads, default='{"start": -32, "end": 0, "n_bases": 7, "spline_order": 2, "intercept": 1}', help='A JSON dict')
    parser.add_argument('--debias', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
