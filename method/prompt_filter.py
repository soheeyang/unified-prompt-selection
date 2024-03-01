import torch
from torch import Tensor
import numpy as np
from numpy import ndarray
from sklearn.cluster import KMeans
from typing import List, Dict, Union, Tuple


def get_prompt_filter_indices(prob: Tensor) -> ndarray:
    T, X, Y = prob.size()

    top2_prob = prob.topk(k=2, dim=-1, sorted=True)[0]
    conf_scores = (top2_prob[:, :, 0] - top2_prob[:, :, 1]).sum(1, keepdims=True)
    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(conf_scores)
    except ValueError:  # only one cluster
        return np.arange(T)
    return np.arange(T)[(kmeans.labels_ == kmeans.cluster_centers_.argmax(axis=0))]

def apply_filter_on_prob(
    prob: Tensor, filter_indices: Union[ndarray, List[int]]
) -> Tensor:
    assert isinstance(prob, torch.Tensor), "Prob must be an instance of torch.Tensor"
    return prob[filter_indices, ...]

def apply_filter_on_tensor_dict(
    tensor_dict: Dict[str, Tensor], filter_indices: Union[ndarray, List[int]]
) -> Dict[str, Tensor]:
    assert isinstance(tensor_dict, dict), "tensor_dict must be a dict type"
    new_tensor_dict = dict()
    for key in tensor_dict:
        assert isinstance(tensor_dict[key], torch.Tensor), (
            "The value of tensor_dict must be an instance of torch.Tensor"
        )
        new_tensor_dict[key] = tensor_dict[key][filter_indices, ...]
    return new_tensor_dict

def apply_filter_on_info(
    info: List[str], filter_indices: Union[ndarray, List[int]]
) -> List[str]:
    assert isinstance(info, list), "info must be a list type"
    return np.asarray(info)[filter_indices].tolist()

def apply_filter(
    prob: Tensor, tensor_dict: Dict[str, Tensor], filter_indices: Union[ndarray, List[int]]
) -> Tuple[Tensor, Dict[str, Tensor]]:
    # apply filter to prob, tensor_dict
    prob = apply_filter_on_prob(prob, filter_indices)
    tensor_dict = apply_filter_on_tensor_dict(tensor_dict, filter_indices)
    return prob, tensor_dict
