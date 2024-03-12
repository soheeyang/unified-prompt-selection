from method.utils import safe_log, get_entropy, get_one_hot
from torch import Tensor
from typing import Tuple, List

'''
* [ Demension ]

T : The number of prompts
X : The number of instances
Y : The number of answer choices

tenser_dict_prob: [T, X, Y]
template_prob: [X, Y]
'''

def get_le(template_prob: Tensor) -> Tensor:
    return get_entropy(template_prob, sum_axis=-1).mean()

def get_mdl_m(template_prob: Tensor) -> Tensor:
    return - get_entropy(template_prob, sum_axis=-1).mean()

def get_ge_m(template_prob: Tensor, X_axis: int = 0, Y_axis: int = -1, keepdims: bool = False) -> Tensor:
    d_prob = template_prob.mean(axis=X_axis, keepdims=keepdims)
    return get_entropy(d_prob, sum_axis=Y_axis, keepdims=keepdims)

def get_ge(template_prob: Tensor, X_axis: int = 0, Y_axis: int = -1, keepdims: bool = False) -> Tensor:
    template_prob = get_one_hot(template_prob)
    d_prob = template_prob.mean(axis=X_axis, keepdims=keepdims)
    return get_entropy(d_prob, sum_axis=Y_axis, keepdims=keepdims)

def get_log_prob_mean(tensor_dict_prob: Tensor, axis: int = 0) -> Tensor:
    return safe_log(tensor_dict_prob).mean(axis)

def get_prob_mean(tensor_dict_prob: Tensor, axis: int = 0) -> Tensor:
    return tensor_dict_prob.mean(axis)

def get_binary_prob_sum(tensor_dict_prob: Tensor, axis: int = 0) -> Tensor:
    return get_one_hot(tensor_dict_prob, Y_axis=-1).sum(axis)

def get_zlp(template_prob: Tensor, tensor_dict_prob: Tensor, Y_axis: int = -1) -> Tensor:
    s_xy = get_log_prob_mean(tensor_dict_prob)
    return (template_prob.argmax(Y_axis) == s_xy.argmax(Y_axis)).float().mean()

def get_zpm(template_prob: Tensor, tensor_dict_prob: Tensor, Y_axis: int = -1) -> Tensor:
    s_xy = get_prob_mean(tensor_dict_prob)
    return (template_prob.argmax(Y_axis) == s_xy.argmax(Y_axis)).float().mean()

def get_zmv(template_prob: Tensor, tensor_dict_prob: Tensor, Y_axis: int = -1) -> Tensor:
    s_xy = get_binary_prob_sum(tensor_dict_prob)
    return (template_prob.argmax(Y_axis) == s_xy.argmax(Y_axis)).float().mean()

def get_mi(template_prob: Tensor) -> Tensor:
    ge_m = get_ge_m(template_prob)
    mdl_m = get_mdl_m(template_prob)
    return (ge_m + mdl_m)

def get_mi_g(template_prob: Tensor) -> Tensor:
    ge = get_ge(template_prob)
    mdl_m = get_mdl_m(template_prob)
    return (ge + mdl_m)

def get_ppl(template_ppl: Tensor) -> Tensor:
    return template_ppl.mean()

def get_mi_gl(tensor_dict_prob: Tensor) -> Tuple[List[float], List[int]]:
    tensor_dict_prob = tensor_dict_prob.transpose(0, 1)  # [X, T, Y]
    ge = get_ge(tensor_dict_prob, X_axis=0, Y_axis=-1)

    X = tensor_dict_prob.size(0)
    mi_agls, selected_prompt_indices = [], []
    for i in range(X):
        instance_prob = tensor_dict_prob[i]  # [T, Y]
        mdl = -get_entropy(instance_prob, -1)
        mi_agl = ge + mdl
        mi_agls.append(mi_agl.max().item())
        selected_prompt_indices.append(mi_agl.argmax().item())
    return mi_agls, selected_prompt_indices

def get_mi_l(tensor_dict_prob: Tensor) -> Tuple[List[float], List[int]]:
    tensor_dict_prob = tensor_dict_prob.transpose(0, 1)  # [X, T, Y]
    ge_m = get_ge_m(tensor_dict_prob, X_axis=0, Y_axis=-1)

    X = tensor_dict_prob.size(0)
    mi_als, selected_prompt_indices = [], []
    for i in range(X):
        instance_prob = tensor_dict_prob[i]  # [T, Y]
        mdl = -get_entropy(instance_prob, -1)
        mi_al = ge_m + mdl
        mi_als.append(mi_al.max().item())
        selected_prompt_indices.append(mi_al.argmax().item())
    return mi_als, selected_prompt_indices

def get_mdl(tensor_dict_prob: Tensor) -> Tuple[List[float], List[int]]:
    tensor_dict_prob = tensor_dict_prob.transpose(0, 1)  # [X, T, Y]

    X = tensor_dict_prob.size(0)
    mdls, selected_prompt_indices = [], []
    for i in range(X):
        instance_prob = tensor_dict_prob[i]  # [T, Y]
        mdl = -get_entropy(instance_prob, -1)
        mdls.append(mdl.max().item())
        selected_prompt_indices.append(mdl.argmax().item())
    return mdls, selected_prompt_indices

def get_i_ppl(tensor_dict_perplexity: Tensor) -> Tuple[List[float], List[int]]:
    tensor_dict_perplexity = tensor_dict_perplexity.transpose(0, 1)  # [X, T]
    
    X = tensor_dict_perplexity.size(0)
    ppls, selected_prompt_indices = [], []
    for i in range(X):
        ppl = tensor_dict_perplexity[i]
        ppls.append(ppl.min().item())
        selected_prompt_indices.append(ppl.argmin().item())
    return ppls, selected_prompt_indices