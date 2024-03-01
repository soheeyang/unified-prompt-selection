import os
import json
import torch
from torch import Tensor
import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, accuracy_score
from typing import Tuple, List, Union, Dict


def safe_log(prob: Tensor) -> Tensor:
    return torch.log(prob + 1e-7)

def get_entropy(prob: Tensor, sum_axis: int, keepdims: bool = False) -> Tensor:
    return -(prob * safe_log(prob)).sum(axis=sum_axis, keepdims=keepdims)

def get_one_hot(prob: Tensor, Y_axis: int = -1) -> Tensor:    
    return torch.nn.functional.one_hot(prob.argmax(Y_axis), prob.size(Y_axis)).float()

def _mean_normalize_prob(prob: Tensor, Y_axis: int = -1) -> Tensor:
    normalized_prob = prob / prob.sum(dim=Y_axis, keepdim=True)
    return normalized_prob

def normalize_prob(prob: Tensor, norm_type: str, Y_axis: int = -1) -> Tensor:
    if norm_type == 'mean':
        normalized_prob = _mean_normalize_prob(prob, Y_axis)
    elif norm_type == 'softmax':
        normalized_prob = torch.softmax(prob, dim=Y_axis)
    else:
        raise ValueError(norm_type)

    assert not torch.isnan(normalized_prob).any()
    return normalized_prob

def get_normalized_probs_from_logprob(log_prob: Tensor, norm_type: str, Y_axis: int = -1) -> Tuple[Tensor, Tensor]:
    prob = torch.exp(log_prob)

    normalized_prob = normalize_prob(prob, norm_type, Y_axis)

    normalized_log_prob = safe_log(normalized_prob)
    assert not torch.isnan(normalized_log_prob).any()
    return normalized_prob, normalized_log_prob

def get_predictions(predscore: Tensor) -> List[int]:
    T, X, Y = predscore.size()
    return [predscore[t].argmax(-1).tolist() for t in range(T)]

def get_f1_acc(predictions: List[int], targets: Union[Tensor, List[int]]) -> Tuple[List[float], List[float]]:
    f1s = []
    accs = []

    T = len(predictions)
    for t in range(T):
        t_predictions = predictions[t]
        f1s.append(f1_score(targets, t_predictions, average='macro'))
        accs.append(accuracy_score(targets, t_predictions))
    return f1s, accs

def is_zps(method: str) -> bool:
    return True if method in ['ZLP', 'ZPM', 'ZMV'] else False

def is_ppl(method: str) -> bool:
    return True if method == 'PPL' else False

def scatter(df_result: dict, combn: str, method: str) -> None:
    df_result = df_result[(df_result.combn == combn)]
    x, y, z = df_result[method], df_result['acc'], df_result['f1']

    plt.title(combn)
    plt.xlabel(method)
    plt.scatter(x, y, color='blue', label="Accuracy", alpha=0.5)
    plt.scatter(x, z, color='orange', label="F1 Score", alpha=0.3)
    plt.legend()

    # line of best fit
    k = np.linspace(x.min(), x.max(), 100)
    m, b = np.polyfit(x, y, 1)
    plt.plot(k, m*k + b, '-', color='blue', alpha=0.5)
    m, b = np.polyfit(x, z, 1)
    plt.plot(k, m*k + b, '-', color='orange', alpha=0.3)
    
    plt.show()

def get_task_wise_summary_result(
    ps_result: Tuple[DataFrame, Dict[str, Dict[str, List[int]]], List[int]], 
    method: str, 
    cali_type: str, 
    cali_norm_type: str,
) -> dict:
    df_result, calibration_to_prompts_to_predictions, targets = ps_result
    summary_result = dict()
    for combn in df_result.combn.unique():
        result_per_combn = df_result[(df_result.combn == combn)].sort_values(by=method, ascending=False).iloc[0]
        prompt2predictions = calibration_to_prompts_to_predictions[combn]
        predictions = prompt2predictions[result_per_combn['prompt']]
        summary_result[combn] = {
            'method': method,
            'cali_type': cali_type,
            'cali_norm_type': cali_norm_type,
            'accuracy': result_per_combn['acc'],
            'macro_f1': result_per_combn['f1'],
            'model': result_per_combn['model'],
            'task': result_per_combn['task'],
            'token': result_per_combn['token'],
            'prompt': result_per_combn['prompt'],
            'prediction': predictions,
            'target': targets
        }
    return summary_result

def get_instance_wise_summary_result(
    ps_result: DataFrame, method: str, cali_type: str, cali_norm_type: str
) -> dict:
    summary_result = dict()
    for combn in ps_result.combn.unique():
        result_per_combn = ps_result[ps_result.combn == combn]
        summary_result[combn] = {
            'method': method,
            'cali_type': cali_type,
            'cali_norm_type': cali_norm_type,
            'accuracy': accuracy_score(result_per_combn['target'], result_per_combn['prediction']),
            'macro_f1': f1_score(result_per_combn['target'], result_per_combn['prediction'], average='macro'),
            'model': result_per_combn['model'].tolist(),
            'task': result_per_combn['task'].tolist(),
            'token': result_per_combn['token'].tolist(),
            'instance': result_per_combn['instance'].tolist(),
            'prompt': result_per_combn['prompt'].tolist(),
            'prediction': result_per_combn['prediction'].tolist(),
            'target': result_per_combn['target'].tolist()
        }
    return summary_result

def get_summary_result(
    ps_result: Union[
        Tuple[DataFrame, Dict[str, Dict[str, List[int]]], List[int]],
        DataFrame
    ], 
    method: str, 
    cali_type: str, 
    cali_norm_type: str, 
    select_for_each_x: bool,
) -> dict:
    if select_for_each_x:
        summaryFunc = get_instance_wise_summary_result
    else:
        summaryFunc = get_task_wise_summary_result
    return summaryFunc(ps_result, method, cali_type, cali_norm_type)

def save_summary_result(
    summary_result: dict, 
    output_dir: str, 
    method: str, 
    first_token: bool, 
    one_hot: bool, 
    select_for_each_x: bool,
    cali_type: str,
    cali_norm_type: str,
    filter: bool,
    unbalance: bool,
) -> str:
    
    result_dir = "./results"
    output_dir = os.path.join(result_dir, output_dir)
    os.makedirs(output_dir, exist_ok=True)

    output_filename = f"method={method}__all_tokens={not first_token}__one_hot={one_hot}__select_for_each_x={select_for_each_x}__cali_type={cali_type}__cali_norm_type={cali_norm_type}__filter={filter}__unbalance={unbalance}.json"
    ps_result_dir = os.path.join(output_dir, output_filename)
    with open(ps_result_dir, 'w') as f:
        json.dump(summary_result, f, indent=4)
    return ps_result_dir