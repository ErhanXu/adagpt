import random
import numpy as np
import tqdm
import argparse
import json
from data_builder import load_data
from metrics import get_roc_metrics, get_precision_recall_metrics

def experiment(args):
    # load data
    sampling_model_list = args.sampling_model_name.split("_")
    scoring_model_list = args.scoring_model_name.split("_")

    pre_classifier_res = []
    task = args.dataset_file.split('/')[-1]
    model_pair_list = []
    for sampling_model, scoring_model in zip(sampling_model_list, scoring_model_list):
        res_file_name = f'{args.pre_classified_dir}/{task}.{sampling_model}_{scoring_model}.classification.bspline.json'
        with open(res_file_name, 'r') as fout:
            res = json.load(fout)
        pre_classifier_res.append(res)
        model_pair_list.append(f'{sampling_model}_{scoring_model}')

    real_pred_all = np.array([x['predictions']['real'] for x in pre_classifier_res])
    llm_pred_all = np.array([x['predictions']['samples'] for x in pre_classifier_res])

    real_pred = np.max(real_pred_all, axis=0)
    llm_pred = np.max(llm_pred_all, axis=0)

    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    name = 'classification'
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        original_text = data["original"][idx]
        sampled_text = data["sampled"][idx]
        original_crit = real_pred[idx]
        sampled_crit = llm_pred[idx]
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

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
                'compute_info': {'pre_time': 0.0, 'eval_time': 0.0, 
                                    'pre_memory': 0.0, 'eval_memory': 0.0,}
                }

    results_file = f'{args.output_file}.classification.bspline.json'
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_main/results/writing_qwen-7b.falcon-7b_falcon-7b-instruct_qwen-7b_qwen-7b-instruct")  # output file prefix
    parser.add_argument('--pre_classified_dir', type=str, default="./exp_gpt3to4/results/")  # output file prefix
    parser.add_argument('--dataset', type=str, default="writing")
    parser.add_argument('--dataset_file', type=str, default="./exp_main/data/writing_qwen-7b")
    parser.add_argument('--sampling_model_name', type=str, default="falcon-7b_llama2-13b")
    parser.add_argument('--scoring_model_name', type=str, default="falcon-7b-instruct_llama2-13b-chat")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
