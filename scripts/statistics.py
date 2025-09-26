import torch
import numpy as np

def get_token_likelihood(logits, labels):
    assert logits.shape[0] == 1
    if labels is not None:
        assert labels.shape[0] == 1
        labels = labels.unsqueeze(-1) if labels.ndim == logits.ndim - 1 else labels
    lprobs = torch.log_softmax(logits, dim=-1)
    if labels is not None:
        log_likelihood = lprobs.gather(dim=-1, index=labels)
    else:
        log_likelihood = lprobs
    return log_likelihood

def get_martingale_stat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    text_length = labels.size(-1)
    assert burn_in_num < text_length
    
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    log_likelihood_x = w_func(get_token_likelihood(logits_score, labels))
    log_likelihood_x_full = w_func(get_token_likelihood(logits_score, None))
    p_x_full = torch.softmax(logits_ref, dim=-1)
    mu_x = (log_likelihood_x_full * p_x_full).sum(dim=-1)
    sigma_x = (torch.square(log_likelihood_x_full - mu_x.unsqueeze(-1)) * p_x_full).sum(dim=-1).sqrt()
    discrepancy = (log_likelihood_x.squeeze(-1) - mu_x) / sigma_x
    discrepancy = discrepancy[0, burn_in_num:]
    stats = discrepancy.sum(dim=-1) / torch.sqrt(torch.tensor(text_length - burn_in_num))
    return stats.item()

def get_classification_stat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    # max_idx = probs_ref.argmax(dim=-1, keepdim=True)        
    # probs_ref = torch.zeros_like(probs_ref).scatter_(       
    #     dim=-1,
    #     index=max_idx,
    #     src=torch.ones_like(max_idx, dtype=probs_ref.dtype)  
    # )
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    # var_ref = torch.ones_like(mean_ref)
    log_likelihood = log_likelihood[0, burn_in_num:]
    mean_ref = mean_ref[0, burn_in_num:]
    var_ref = var_ref[0, burn_in_num:]
    L = torch.tensor(var_ref.shape[0])
    stat = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1) - shift_value * L) / var_ref.sum(dim=-1).sqrt()
    stat = stat.mean()
    return stat.item()

def get_meanlb_stat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    # var_ref = torch.ones_like(mean_ref)
    log_likelihood = log_likelihood[0, burn_in_num:]
    mean_ref = mean_ref[0, burn_in_num:]
    var_ref = var_ref[0, burn_in_num:]
    L = torch.tensor(var_ref.shape[0])
    constant = 1.96
    std = (torch.var(log_likelihood, unbiased=False) - torch.var(mean_ref, unbiased=False)).std()

    stat = (log_likelihood.mean(dim=-1) - mean_ref.mean(dim=-1) - shift_value * L) - constant * std
    stat = stat.mean()
    return stat.item()

def get_variance_stat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)

    log_likelihood = log_likelihood[0, burn_in_num:]
    mean_ref = mean_ref[0, burn_in_num:]
    term1 = torch.var(log_likelihood, unbiased=False)
    term2 = torch.var(mean_ref, unbiased=False)
    stat = term1 - term2
    return stat.item()

def get_appro_variance_stat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)

    log_likelihood = log_likelihood[0, burn_in_num:]
    mean_ref = mean_ref[0, burn_in_num:]
    term1 = torch.square(log_likelihood).mean(dim=-1)
    term2 = torch.square(mean_ref).mean(dim=-1)
    stat = term1 - term2
    return stat.item()

def get_cusum_supstat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None, 
                      n_scales = 4, alpha_min: float = 0.6, alpha_max: float = 1.0):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    mu_ref = (log_likelihood - mean_ref)[0, burn_in_num:]
    var_ref = var_ref[0, burn_in_num:]
    
    device = mu_ref.device
    prefix_mu = torch.cat([torch.zeros(1, device=device), mu_ref.cumsum(dim=0)])
    prefix_var = torch.cat([torch.zeros(1, device=device), var_ref.cumsum(dim=0)])
    min_std = torch.sqrt(prefix_var[-1] * torch.finfo(prefix_var.dtype).eps)  
    text_length = prefix_mu.size(-1)

    alphas = torch.linspace(alpha_min, alpha_max, n_scales, device=device)
    G_list = torch.floor((text_length ** alphas)).to(torch.int64)
    MIN_G = 30
    if (G_list < MIN_G).any():
        kept = G_list[G_list >= MIN_G]
        G_list = torch.cat([torch.tensor([MIN_G], dtype=G_list.dtype, device=G_list.device),
                            kept], dim=0)

    max_stat = -torch.inf
    for G in G_list:
        for i in range(text_length - G):
            mu_sum = prefix_mu[i+G] - prefix_mu[i]  # O(1)
            var_sum = prefix_var[i+G] - prefix_var[i]  # O(1)
            stat = mu_sum / torch.sqrt(var_sum).clamp_min(min_std)
            
            if stat > max_stat:
                max_stat = stat
                best_length = G
                best_j = i + G
                best_i = i
    print(f"sup stat with length: {best_length}, i={best_i}, j={best_j}")
    return max_stat.item()

def get_cusum_infstat(logits_ref, logits_score, labels, burn_in_num, w_func, shift_value=None, 
                      n_scales = 4, alpha_min: float = 0.6, alpha_max: float = 1.0):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        # print(f"WARNING: vocabulary size mismatch {logits_ref.size(-1)} vs {logits_score.size(-1)}.")
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    mu_ref = (log_likelihood - mean_ref)[0, burn_in_num:]
    var_ref = var_ref[0, burn_in_num:]
    
    device = mu_ref.device
    prefix_mu = torch.cat([torch.zeros(1, device=device), mu_ref.cumsum(dim=0)])
    prefix_var = torch.cat([torch.zeros(1, device=device), var_ref.cumsum(dim=0)])
    min_std = torch.sqrt(prefix_var[-1] * torch.finfo(prefix_var.dtype).eps)  
    text_length = prefix_mu.size(-1)

    alphas = torch.linspace(alpha_min, alpha_max, n_scales, device=device)
    G_list = torch.floor((text_length ** alphas)).to(torch.int64)
    MIN_G = 30
    if (G_list < MIN_G).any():
        kept = G_list[G_list >= MIN_G]
        G_list = torch.cat([torch.tensor([MIN_G], dtype=G_list.dtype, device=G_list.device),
                            kept], dim=0)

    min_stat = torch.inf
    for G in G_list:
        for i in range(text_length - G):
            mu_sum = prefix_mu[i+G] - prefix_mu[i]  # O(1)
            var_sum = prefix_var[i+G] - prefix_var[i]  # O(1)
            stat = mu_sum / torch.sqrt(var_sum).clamp_min(min_std)
            
            if stat <= min_stat:
                min_stat = stat
                best_length = G
                best_j = i + G
                best_i = i
    print(f"inf stat with length: {best_length}, i={best_i}, j={best_j}")
    return min_stat.item()



def cramer_von_mises_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    assert x.dim() == 1, "x must be a 1D tensor"
    assert y.dim() == 1, "y must be a 1D tensor"
    
    n = x.shape[0]
    m = y.shape[0]
    
    x_sorted, _ = torch.sort(x)
    y_sorted, _ = torch.sort(y)
    
    cdf_distance = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    
    combined = torch.cat([x_sorted, y_sorted])
    combined_sorted, _ = torch.sort(combined)
    unique_values = torch.unique(combined_sorted)
    
    for value in unique_values:
        cdf_x = torch.sum(x_sorted <= value) / n
        cdf_y = torch.sum(y_sorted <= value) / m
        cdf_distance += (cdf_x - cdf_y) ** 2
    
    return cdf_distance


def get_distr_stat(logits_ref, logits_score, labels, burn_in_num, w_func, ref_distr=None):
    assert logits_ref.shape[0] == 1
    assert logits_score.shape[0] == 1
    assert labels.shape[0] == 1
    if logits_ref.size(-1) != logits_score.size(-1):
        vocab_size = min(logits_ref.size(-1), logits_score.size(-1))
        logits_ref = logits_ref[:, :, :vocab_size]
        logits_score = logits_score[:, :, :vocab_size]

    labels = labels.unsqueeze(-1) if labels.ndim == logits_score.ndim - 1 else labels
    lprobs_score = w_func(torch.log_softmax(logits_score, dim=-1))
    probs_ref = torch.softmax(logits_ref, dim=-1)
    log_likelihood = lprobs_score.gather(dim=-1, index=labels).squeeze(-1)
    mean_ref = (probs_ref * lprobs_score).sum(dim=-1)
    var_ref = (probs_ref * torch.square(lprobs_score)).sum(dim=-1) - torch.square(mean_ref)
    # var_ref = torch.ones_like(mean_ref)
    log_likelihood = log_likelihood[0, burn_in_num:]
    mean_ref = mean_ref[0, burn_in_num:]
    var_ref = var_ref[0, burn_in_num:]
    stat_distr = (log_likelihood - mean_ref) / var_ref.sqrt()

    if ref_distr is not None:
        stat = cramer_von_mises_distance(ref_distr, stat_distr)
        return stat.item()
    else:
        return stat_distr

