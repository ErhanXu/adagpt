import numpy as np
import re
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from RevisedDetect.bart_score import BARTScorer

import argparse
import json
from data_builder import load_data, save_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from model import load_tokenizer, load_model
import custom_datasets
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

PROMPT = "Revise the following text: \"{}\""
REGEN_NUMBER = 4

class PrefixSampler:
    def __init__(self, args):
        self.args = args
        self.base_tokenizer = load_tokenizer(args.base_model_name, args.cache_dir)
        self.base_model = load_model(args.base_model_name, args.device, args.cache_dir)
        # self.pipe = pipeline("text-generation", model=self.base_model, tokenizer=self.base_tokenizer, device=torch.cuda.current_device())

    def _sample_rewrite_text_from_model(self, texts):
        texts_num_tokens = self.base_tokenizer(texts, return_tensors="pt", padding=True, return_token_type_ids=False)['input_ids'].shape[1]
        prompt_texts = [PROMPT.format(o) for o in texts] 

        self.base_model.eval()
        decoded = ['' for _ in range(len(texts))]

        sampling_kwargs = {'temperature': self.args.temperature}
        if self.args.do_top_p:
            sampling_kwargs['top_p'] = self.args.top_p
        elif self.args.do_top_k:
            sampling_kwargs['top_k'] = self.args.top_k

        sampling_kwargs['min_new_tokens'] = int(0.5*texts_num_tokens)
        sampling_kwargs['max_new_tokens'] = int(1.5*texts_num_tokens)
        sampling_kwargs["eos_token_id"] = self.base_tokenizer.eos_token_id
        sampling_kwargs['pad_token_id'] = self.base_tokenizer.eos_token_id

        all_encoded = self.base_tokenizer(prompt_texts, return_tensors="pt", padding=True, return_token_type_ids=False).to(self.args.device)
        prompt_lens = all_encoded['input_ids'].shape[1]
        outputs = self.base_model.generate(**all_encoded, do_sample=True, **sampling_kwargs)
        gen_ids = outputs[:, prompt_lens:]
        decoded = self.base_tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        return decoded

    def generate_samples(self, raw_data, batch_size):
        def _truncate_to_substring(text, substring, idx_occurrence):
            # truncate everything after the idx_occurrence occurrence of substring
            assert idx_occurrence > 0, 'idx_occurrence must be > 0'
            idx = -1
            for _ in range(idx_occurrence):
                idx = text.find(substring, idx + 1)
                if idx == -1:
                    return text
            return text[:idx]

        data = {
            "original": [],
            "sampled": [],
        }

        assert len(raw_data) % batch_size == 0
        for batch in range(len(raw_data) // batch_size):
            print('Generating samples for batch', batch, 'of', len(raw_data) // batch_size)
            original_text = raw_data[batch * batch_size:(batch + 1) * batch_size]
            sampled_text = self._sample_rewrite_text_from_model(original_text)

            for o, s in zip(original_text, sampled_text):
                if self.args.dataset == 'pubmed':
                    s = _truncate_to_substring(s, 'Question:', 2)
                    o = o.replace(custom_datasets.SEPARATOR, ' ')

                # add to the data
                data["original"].append(o)
                data["sampled"].append(s)

        return data

def get_regen_samples(sampler, text):
    data = sampler.generate_samples([text], batch_size=1)
    return data['sampled']

def get_revised_gpt_simple_statistic(sampler, text, model, regens=None):
    if regens is None:
        regens = get_regen_samples(sampler, text)  # list of length K

    sim_score = model.score(regens, [text], batch_size=1)

    return sim_score.item()

def experiment(args):
    sampler = PrefixSampler(args)
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # n_samples = 2
    # evaluate criterion
    name = "revised_gpt2" ## existing methods (just resampling one time)
    model = BARTScorer(device=args.device, model_name="facebook/bart-large-cnn", cache_dir=args.cache_dir)
    criterion_fn = get_revised_gpt_simple_statistic

    rewrite_texts_file = f'{args.output_file}.rewrite_{REGEN_NUMBER}.json'
    if os.path.exists(rewrite_texts_file):
        with open(rewrite_texts_file, "r") as fin:
            rewrite_texts = json.load(fin)
            print(f"Load rewrite texts from file: {rewrite_texts_file}")
    else:
        rewrite_texts = {'rewrite_original': [None]*n_samples, 'rewrite_sampled': [None]*n_samples}
    
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        # original text
        original_text = data["original"][idx]
        rewrite_original_text = [rewrite_texts[idx]['rewrite_original'][0]]
        original_crit = criterion_fn(sampler, original_text, model, rewrite_original_text)
        # sampled text
        sampled_text = data["sampled"][idx]
        rewrite_sampled_text = [rewrite_texts[idx]['rewrite_sampled'][0]]
        sampled_crit = criterion_fn(sampler, sampled_text, model, rewrite_sampled_text)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})

    # compute prediction scores for real/sampled passages
    predictions = {'real': [x["original_crit"] for x in results],
                   'samples': [x["sampled_crit"] for x in results]}
    print(f"Real mean/std: {np.mean(predictions['real']):.2f}/{np.std(predictions['real']):.2f}, Samples mean/std: {np.mean(predictions['samples']):.2f}/{np.std(predictions['samples']):.2f}")
    fpr, tpr, roc_auc = get_roc_metrics(predictions['real'], predictions['samples'])
    p, r, pr_auc = get_precision_recall_metrics(predictions['real'], predictions['samples'])
    print(f"Criterion {name}_threshold ROC AUC: {roc_auc:.4f}, PR AUC: {pr_auc:.4f}")
    # results
    results_file = f'{args.output_file}.{name}.json'
    results = { 'name': f'{name}_threshold',
                'info': {'n_samples': n_samples},
                'predictions': predictions,
                'raw_results': results,
                'metrics': {'roc_auc': roc_auc, 'fpr': fpr, 'tpr': tpr},
                'pr_metrics': {'pr_auc': pr_auc, 'precision': p, 'recall': r},
                'loss': 1 - pr_auc}
    with open(results_file, 'w') as fout:
        json.dump(results, fout)
        print(f'Results written into {results_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_gemma-9b-instruct_expand")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_prompt/data/xsum_gemma-9b-instruct_expand")
    parser.add_argument('--base_model_name', type=str, default="gemma-9b-instruct")
    parser.add_argument('--do_top_k', action='store_true')
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
