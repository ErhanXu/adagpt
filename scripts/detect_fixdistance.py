import numpy as np
import re
import torch
import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

import argparse
import json
from data_builder import load_data, save_data
from metrics import get_roc_metrics, get_precision_recall_metrics
from model import load_tokenizer, load_model
import custom_datasets
import os
os.environ["TORCH_COMPILE_DISABLE"] = "1"
torch._dynamo.config.disable = True

PROMPT = "You are a rewriting expert and you would rewrite the text without missing the original details. Original text: \"{}\" Here is the rewritten version: \n\n"

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
    data = [text] * sampler.args.regen_number
    data = sampler.generate_samples(data, batch_size=sampler.args.batch_size)
    return data['sampled']

def compute_total_logprob_from_logits(logits, labels, pad_index):
    """
    返回未归一的总 log-prob（sum over non-pad tokens）。
    logits: [B, T, V], labels: [B, T]
    """
    lprobs = torch.log_softmax(logits, dim=-1)  # [B, T, V]
    # gather true-token logprobs
    labels_expanded = labels.unsqueeze(-1)  # [B, T, 1]
    token_logprobs = lprobs.gather(dim=-1, index=labels_expanded).squeeze(-1)  # [B, T]
    mask = (labels != pad_index).float()  # [B, T]
    total = (token_logprobs * mask).sum(dim=1)  # [B]
    return total  # summed log-prob per example


def sequence_total_logprob(sampler, input_ids, attention_mask):
    """
    计算整个序列 input_ids 的 total log-prob
    """
    with torch.no_grad():
        outputs = sampler.base_model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits  # [B, T, V]
    # shift for causal language modeling: predict input_ids[:,1:] from logits[:,:-1]
    shifted_logits = logits[:, :-1, :]  # [B, T-1, V]
    shifted_labels = input_ids[:, 1:]  # [B, T-1]
    pad_id = sampler.base_tokenizer.pad_token_id
    total_lp = compute_total_logprob_from_logits(shifted_logits, shifted_labels, pad_id)  # [B]
    return total_lp  # [B]


def compute_sequence_logprob(sampler, texts):
    """
    计算一批文本的 total log-prob（sum over tokens） under the base model.
    返回 tensor shape [len(texts)]，每个是 log p(text).
    """
    device = sampler.args.device
    pad_id = sampler.base_tokenizer.pad_token_id

    # Tokenize full texts
    encoded = sampler.base_tokenizer(texts, return_tensors="pt", padding=True).to(device)

    input_ids = encoded["input_ids"]  # [B, T]
    attention_mask = encoded["attention_mask"]

    # Use your existing helper: sequence_total_logprob expects full input_ids and mask
    total_logp = sequence_total_logprob(sampler, input_ids, attention_mask)  # [B]
    lengths = attention_mask.sum(dim=1).clamp(min=1).float()  # [B]

    return total_logp / lengths  # summed log-prob per example


def get_rewrite_gpt_simple_statistic(sampler, text, regen_number=None):
    """
    用新的统计量 score = z - (1/K) sum_i tilde_z_i，
    其中 z = log p(X)， tilde_z_i = log p(rewrite_i)
    """
    if regen_number is None:
        regen_number = sampler.args.regen_number

    # 1. 生成 K 个 rewrite \tilde X_i
    regens = get_regen_samples(sampler, text)  # list of length K

    # 2. 计算 z = log p(X)  和  each \tilde z_i = log p(\tilde X_i)
    z_tensor = compute_sequence_logprob(sampler, [text])  # [1]
    tilde_zs = compute_sequence_logprob(sampler, regens)  # [K]

    # 3. 统计量： z - mean(tilde_zs)
    score = z_tensor.mean() - tilde_zs.mean()  # scalar
    return score.item(), regens

def experiment(args):
    sampler = PrefixSampler(args)
    # load data
    data = load_data(args.dataset_file)
    n_samples = len(data["sampled"])
    # n_samples = 2
    # evaluate criterion
    name = "rewrite_gpt"
    criterion_fn = get_rewrite_gpt_simple_statistic

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rewrite_text = []
    results = []
    for idx in tqdm.tqdm(range(n_samples), desc=f"Computing {name} criterion"):
        # original text
        original_text = data["original"][idx]
        original_crit, rewrite_original = criterion_fn(sampler, original_text)
        # sampled text
        sampled_text = data["sampled"][idx]
        sampled_crit, rewrite_sampled = criterion_fn(sampler, sampled_text)
        # result
        results.append({"original": original_text,
                        "original_crit": original_crit,
                        "sampled": sampled_text,
                        "sampled_crit": sampled_crit})
        rewrite_text.append({'rewrite_original': rewrite_original, 'rewrite_sampled': rewrite_sampled})

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
    
    rewrite_texts_file = f'{args.output_file}.rewrite_{args.regen_number}.json'
    with open(rewrite_texts_file, 'w') as fout:
        json.dump(rewrite_text, fout)
        print(f'Rewritten texts saved into {rewrite_texts_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_file', type=str, default="./exp_prompt/results/xsum_gemma-9b-instruct_expand")
    parser.add_argument('--dataset', type=str, default="xsum")
    parser.add_argument('--dataset_file', type=str, default="./exp_prompt/data/xsum_gemma-9b-instruct_expand")
    parser.add_argument('--regen_number', type=int, default=4)
    parser.add_argument('--base_model_name', type=str, default="gemma-9b-instruct")
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--do_top_k', action='store_true')
    # parser.add_argument('--do_top_k', type=bool, default=True)
    parser.add_argument('--top_k', type=int, default=40)
    parser.add_argument('--do_top_p', action='store_true')
    parser.add_argument('--top_p', type=float, default=0.96)
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--cache_dir', type=str, default="../cache")
    args = parser.parse_args()

    experiment(args)
