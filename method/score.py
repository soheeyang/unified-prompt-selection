import torch
from torch import Tensor
import pandas as pd
from pandas import DataFrame
from tqdm.auto import tqdm
from method.postprocessor import PostProcessor
from method.calibration import get_calibrated_prob_and_predscore
from method.utils import (
    is_zps,
    is_ppl,
    get_predictions,
    get_f1_acc,
)
from method.prompt_filter import (
    get_prompt_filter_indices,
    apply_filter,
    apply_filter_on_info
)
from method.methods import (
    get_le,
    get_mdl_m,
    get_ge,
    get_ge_m,
    get_mi,
    get_mi_g,
    get_mi_l,
    get_mi_gl,
    get_mdl,
    get_ppl,
    get_i_ppl,
    get_zlp,
    get_zpm,
    get_zmv,
)
from typing import Dict, Callable, List, Tuple, Union


def get_prob_for_scoring(
    tensor_dict: Dict[str, Tensor], cali_score: bool, cali_type: str, cali_norm_type: str
) -> Tensor:
    score_prob = tensor_dict['prob']

    if cali_score:
        score_prob, _ = get_calibrated_prob_and_predscore(score_prob, tensor_dict['log_unnorm_prob'], tensor_dict, cali_type, cali_norm_type, X_axis=1)
    
    return score_prob

def get_predscore(
    tensor_dict: Dict[str, Tensor], cali_eval: bool, cali_type: str, cali_norm_type: str
) -> Tensor:
    if cali_eval:
        _, predscore = get_calibrated_prob_and_predscore(tensor_dict['prob'], tensor_dict['log_unnorm_prob'], tensor_dict, cali_type, cali_norm_type, X_axis=1)
    else:
        predscore = tensor_dict['log_unnorm_prob']  # no need to normalize cuz we just argmax
    return predscore

def get_task_wise_ps_scores(
    methodFunc: Callable, tensor_dict_value: Tensor, zps: bool
) -> List[float]:
    T = tensor_dict_value.size(0)
    ps_scores = []
    for t in range(T):
        template_prob = tensor_dict_value[t]
        if zps:
            ps_score = methodFunc(template_prob, tensor_dict_value)
        else:
            ps_score = methodFunc(template_prob)

        if isinstance(ps_score, torch.Tensor):
            ps_score = ps_score.item()

        ps_scores.append(ps_score)
    return ps_scores

def get_instance_wise_ps_scores(
    methodFunc: Callable, tensor_dict_value: Tensor
) -> Tuple[List[float], List[int]]:
    return methodFunc(tensor_dict_value)

def get_task_wise_ps_result(
    method: str,
    post_processor: PostProcessor,
    one_hot: bool,
    cali_type: str, 
    cali_norm_type: str, 
    filter: bool, 
    unbalance: bool,
    is_dynamic: bool,
) -> Tuple[DataFrame, Dict[str, Dict[str, List[int]]], List[int]]:
    methodFuncMap = {
        'MI': get_mi_g if one_hot else get_mi,
        'GE': get_ge if one_hot else get_ge_m,
        'LE': get_le,
        'MDL': get_mdl_m,
        'PPL': get_ppl,
        'ZLP': get_zlp,
        'ZPM': get_zpm,
        'ZMV': get_zmv,
    }
    assert method in methodFuncMap, (
        f"you can only use the following methods: {methodFuncMap.keys()}",
        "Check the value of the method argument in ./conf/method.yaml",
    )
    methodFunc = methodFuncMap[method]
    zps = is_zps(method)
    ppl = is_ppl(method)

    dfs = []
    calibration_to_prompts_to_predictions = dict()
    for combn, cali_score, cali_eval in tqdm([
        ('X', False, False),
        ('A', False, True),
        ('P', True, False),
        ('PA', True, True),
    ]):
        tasks, models, prompts, tokens = post_processor.get_tasks_models_prompts_tokens()
        if unbalance and is_dynamic:
            tensor_dict, targets = post_processor.get_unbalanced_tensor_dict_and_targets()
            tasks = [f"U_{task}" for task in tasks]
        elif unbalance and not is_dynamic:
            raise ValueError(
                "The Unbalance argument is only available for dynamic tasks. Please adjust the configuration."
            )
        else:
            tensor_dict, targets = post_processor.get_tensor_dict_and_targets()

        # SCORE CALCULATION
        score_prob = get_prob_for_scoring(tensor_dict, cali_score, cali_type, cali_norm_type)

        # and then filter using the calibrated score
        if filter:
            filter_indices = get_prompt_filter_indices(score_prob)
            score_prob, tensor_dict = apply_filter(score_prob, tensor_dict, filter_indices)
            filtered_infos = [
                apply_filter_on_info(infos, filter_indices) 
                for infos in [tasks, models, prompts, tokens]
            ]
            tasks, models, prompts, tokens = filtered_infos

        # PREDICTION
        predscore = get_predscore(tensor_dict, cali_eval, cali_type, cali_norm_type)  # [T, X, Y]
        predictions = get_predictions(predscore)
        f1, acc = get_f1_acc(predictions, targets)
        
        calibration_to_prompts_to_predictions[combn] = {k: v for k, v in zip(prompts, predictions)}

        if ppl:
            score_prob = tensor_dict['perplexity']

        ps_scores = get_task_wise_ps_scores(
            methodFunc, score_prob, zps
        )

        dfs.append(pd.DataFrame({
            'model': models,
            'task': tasks,
            'token': tokens,
            'prompt': prompts,
            'combn': [combn] * len(acc),
            'acc': acc,
            'f1': f1,
            f'{method}': ps_scores, 
        }))
    return pd.concat(dfs, ignore_index=True), calibration_to_prompts_to_predictions, targets.tolist()

def get_instance_wise_ps_result(
    method: str,
    post_processor: PostProcessor,
    one_hot: bool,
    cali_type: str, 
    cali_norm_type: str, 
    filter: bool, 
    unbalance: bool,
    is_dynamic: bool,
) -> DataFrame:
    methodFuncMap = {
        'PPL': get_i_ppl,
        'MDL': get_mdl,
        'MI': get_mi_gl if one_hot else get_mi_l,
    }
    assert method in methodFuncMap, (
        f"If select_for_each_x == True, you can only use the following methods: {methodFuncMap.keys()}",
        "Check the value of the method argument in ./conf/method.yaml",
    )
    methodFunc = methodFuncMap[method]
    ppl = is_ppl(method)

    df_rows = []

    for combn, cali_score, cali_eval in tqdm([
        ('X', False, False),
        ('A', False, True),
        ('P', True, False),
        ('PA', True, True),
    ]):
        tasks, models, prompts, tokens = post_processor.get_tasks_models_prompts_tokens()
        if unbalance and is_dynamic:
            tensor_dict, targets = post_processor.get_unbalanced_tensor_dict_and_targets()
            tasks = [f"U_{task}" for task in tasks]
        elif unbalance and not is_dynamic:
            raise ValueError(
                "The Unbalance argument is only available for dynamic tasks. Please adjust the configuration."
            )
        else:
            tensor_dict, targets = post_processor.get_tensor_dict_and_targets()

        # SCORE CALCULATION
        score_prob = get_prob_for_scoring(tensor_dict, cali_score, cali_type, cali_norm_type)

        # and then filter using the calibrated score
        if filter:
            filter_indices = get_prompt_filter_indices(score_prob)
            score_prob, tensor_dict = apply_filter(score_prob, tensor_dict, filter_indices)
            filtered_infos = [
                apply_filter_on_info(infos, filter_indices) 
                for infos in [tasks, models, prompts, tokens]
            ]
            tasks, models, prompts, tokens = filtered_infos

        # PREDICTION
        predscore = get_predscore(tensor_dict, cali_eval, cali_type, cali_norm_type)  # [T, X, Y]

        if ppl:
            score_prob = tensor_dict['perplexity']

        ps_scores, selected_prompt_indices = get_instance_wise_ps_scores(
            methodFunc, score_prob
        )
        T, X, Y = predscore.size()
        for i in range(X):
            selected_prompt_idx = selected_prompt_indices[i]
            prediction = predscore[selected_prompt_idx][i].argmax().item()
            target = targets[i].item()
            df_rows.append({
                'model': models[selected_prompt_idx],
                'task': tasks[selected_prompt_idx],
                'token': tokens[selected_prompt_idx],
                'prompt': prompts[selected_prompt_idx],
                'combn': combn,
                'instance': i,
                'prediction': prediction,
                'target': target,
                'correct': prediction == target,
                f'{method}': ps_scores[i],
            })

    return pd.DataFrame(df_rows)

def get_ps_result(
    method: str,
    post_processor: PostProcessor,
    one_hot: bool,
    cali_type: str, 
    cali_norm_type: str, 
    filter: bool, 
    unbalance: bool,
    is_dynamic: bool,
    select_for_each_x: bool,
) -> Union[
        Tuple[DataFrame, Dict[str, Dict[str, List[int]]], List[int]],
        DataFrame
    ]:
    if select_for_each_x:
        scoreFunc = get_instance_wise_ps_result
    else:
        scoreFunc = get_task_wise_ps_result
    
    return scoreFunc(
        method=method,
        post_processor=post_processor,
        one_hot=one_hot,
        cali_type=cali_type, 
        cali_norm_type=cali_norm_type, 
        filter=filter, 
        unbalance=unbalance,
        is_dynamic=is_dynamic,
    )