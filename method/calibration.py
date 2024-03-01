import torch
from torch import Tensor
from method.utils import (
    normalize_prob,
    get_normalized_probs_from_logprob,
)
from typing import Tuple, Dict


def calibrate_cbm(
    norm_prob: Tensor, norm_type: str, X_axis: int = 1
) -> Tuple[Tensor, Tensor]:
    cali_unnorm_prob = norm_prob / norm_prob.mean(dim=X_axis, keepdims=True)
    cali_predscore = cali_unnorm_prob
    cali_prob = normalize_prob(cali_unnorm_prob, norm_type, -1)
    return cali_prob, cali_predscore

def calibrate_cc(
    log_unnorm_prob: Tensor, tensor_dict: Dict[str, Tensor], norm_type: str
) -> Tuple[Tensor, Tensor]:
    # https://github.com/tonyzhaozh/few-shot-learning/blob/cf6e9202c714d7cfc1cf374db0a0b29acddbe845/run_classification.py#L228
    p_cf = (torch.exp(tensor_dict['log_unnorm_empty']) + torch.exp(tensor_dict['log_unnorm_na']) + torch.exp(tensor_dict['log_unnorm_mask'])) / 3

    # https://github.com/tonyzhaozh/few-shot-learning/blob/cf6e9202c714d7cfc1cf374db0a0b29acddbe845/run_classification.py#L229
    p_cf /= p_cf.sum(-1, keepdims=True)

    # https://github.com/tonyzhaozh/few-shot-learning/blob/cf6e9202c714d7cfc1cf374db0a0b29acddbe845/run_classification.py#L210
    # https://github.com/tonyzhaozh/few-shot-learning/blob/cf6e9202c714d7cfc1cf374db0a0b29acddbe845/run_classification.py#L132
    norm_prob, _ = get_normalized_probs_from_logprob(log_unnorm_prob, 'mean', -1)
    
    # https://github.com/tonyzhaozh/few-shot-learning/blob/cf6e9202c714d7cfc1cf374db0a0b29acddbe845/run_classification.py#L134
    cali_unnorm_prob = norm_prob / p_cf

    cali_predscore = cali_unnorm_prob  # original score
    cali_prob = normalize_prob(cali_unnorm_prob, norm_type, -1)
    return cali_prob, cali_predscore

def calibrate_pmi(
    log_unnorm_prob: Tensor, tensor_dict: Dict[str, Tensor], norm_type: str
) -> Tuple[Tensor, Tensor]:
    # https://github.com/peterwestuw/surface-form-competition/blob/ffc22c6352f41ae1519c370564c9f4cb993fed75/utils.py#L288 they used sum to aggregate the tokens, but let's ignore that

    # https://github.com/peterwestuw/surface-form-competition/blob/ffc22c6352f41ae1519c370564c9f4cb993fed75/utils.py#L287
    # https://github.com/peterwestuw/surface-form-competition/blob/ffc22c6352f41ae1519c370564c9f4cb993fed75/utils.py#L399
    log_cali_unnorm_prob = log_unnorm_prob - tensor_dict['log_unnorm_empty']

    cali_predscore = log_cali_unnorm_prob  # original score
    cali_prob, _ = get_normalized_probs_from_logprob(log_cali_unnorm_prob, norm_type, -1)
    return cali_prob, cali_predscore

def get_calibrated_prob_and_predscore(
    norm_prob: Tensor, log_unnorm_prob: Tensor, tensor_dict: Dict[str, Tensor], cali_type: str, norm_type: str, X_axis: int = 1
) -> Tuple[Tensor, Tensor]:
    assert cali_type in ['cbm', 'cc', 'pmi'], "cali_type must be one of 'cbm', 'cc', or 'pmi'."
    if cali_type == 'cbm':
        return calibrate_cbm(norm_prob, norm_type, X_axis)
    elif cali_type == 'cc':
        return calibrate_cc(log_unnorm_prob, tensor_dict, norm_type)
    elif cali_type == 'pmi':
        return calibrate_pmi(log_unnorm_prob, tensor_dict, norm_type)