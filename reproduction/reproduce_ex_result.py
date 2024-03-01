import os
import json
import glob
import torch
import pickle
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.metrics import f1_score, accuracy_score

logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)
torch.set_default_dtype(torch.float64)


CWD = Path.cwd() if "reproduction" == Path.cwd().name else Path.cwd().joinpath("reproduction")
EVAL_DIRS = CWD.joinpath("ex_eval_results")

with open(CWD.joinpath("task2label_indices.pkl"), "rb") as f:
  task2label_indices = pickle.load(f)


def get_prompt_name_from_filepath(filepath):
    filename = filepath.name
    prompt_name = '__'.join(filename.split('__')[3:-1])
    return prompt_name

def get_all_dirnames():
    all_json_files = {
        raw_json 
        for raw_json in EVAL_DIRS.iterdir()
        if EVAL_DIRS.joinpath(raw_json).is_dir()
    }
    return all_json_files

def get_metainfo_from_dirname(dirname):
    splits = dirname.split('_')
    task = splits[1]
    model = splits[2]

    setup = 'v1'
    if 'fixed12' in dirname:
        setup = 'v12'
    elif 'fixed2' in dirname:
        setup = 'v2'
    elif 'fewshot' in dirname:
        setup = 'few'

    if 'otr' not in dirname:
        token = 'all'
    else:
        if 'test' not in dirname:
            token = 'first'
        else:
            return None

    if 'sum' in dirname:
        return None
    elif 'mean' in dirname:
        return None
    elif task in ['piqa', 'copa', 'hellaswag', 'storycloze']:
        aggr = 'sum'
    else:
        aggr = 'mean'

    if any(x in dirname for x in ['P_x', 'tzero', 'gpt-neox20b']):
        return None

    Y = {
        'imdb': 2,
        'sst2': 2,
        'agnews': 4,
        'glue-rte': 2,
        'newspop': 4,
        'tweet-irony': 2,
        'tweet-emotion': 4,
        'cb': 3,
        'sst5': 5,
        'copa': 2,
        'piqa': 2,
        'storycloze': 2,
        'hellaswag': 4,
    }

    return {
        'dirname': dirname,
        'model': model,
        'task': task,
        'setup': setup,
        'token': token,
        'aggr': aggr,
        'Y': Y[task],
    }

def get_all_dirnames_df():
    dfs = []
    for dirname in get_all_dirnames():
        metainfo = get_metainfo_from_dirname(dirname)
        if metainfo:
            dfs.append(metainfo)

    df = pd.DataFrame(dfs)
    return df

def safe_log(prob):
    return torch.log(prob + 1e-7)

def _mean_normalize_prob(prob, Y_axis=-1):
    normalized_prob = prob / prob.sum(dim=Y_axis, keepdim=True)
    return normalized_prob

def normalize_prob(prob, norm_type, Y_axis=-1):
    if norm_type == 'mean':
        normalized_prob = _mean_normalize_prob(prob, Y_axis)
    elif norm_type == 'softmax':
        normalized_prob = torch.softmax(prob, dim=Y_axis)
    else:
        raise ValueError(norm_type)

    assert not torch.isnan(normalized_prob).any()
    return normalized_prob

def get_normalized_probs_from_logprob(log_prob, norm_type, Y_axis=-1):
    prob = torch.exp(log_prob)

    normalized_prob = normalize_prob(prob, norm_type, Y_axis)

    normalized_log_prob = safe_log(normalized_prob)
    assert not torch.isnan(normalized_log_prob).any()
    return normalized_prob, normalized_log_prob

def get_first_choice_probs(otr_log_prob, task):
    if isinstance(otr_log_prob, (list, tuple)):
        otr_log_prob = torch.tensor(otr_log_prob, dtype=torch.float64)
    batch_labels = torch.tensor(task2label_indices[task], dtype=torch.int64)
    orig_log_prob = torch.gather(otr_log_prob, -1, batch_labels)
    return orig_log_prob

def get_tensor_probs_from_raw(raw, dirname):
    metainfo = get_metainfo_from_dirname(dirname)

    prob_dict = dict()
    for key in ['P(x,t)', 'log_prob', 'empty_log', 'na_log', 'mask_log']:
        prob_dict[key] = torch.tensor(raw[key])

    perplexity = 1 / prob_dict['P(x,t)'] # [X]

    log_prob = prob_dict['log_prob']  # [X, Y]
    log_unnorm_empty = prob_dict['empty_log'].expand(*log_prob.size())  # [1, Y] -> [X, Y]
    log_unnorm_na = prob_dict['na_log'].expand(*log_prob.size())
    log_unnorm_mask = prob_dict['mask_log'].expand(*log_prob.size())

    if metainfo['token'] == 'first':
        if metainfo['Y'] != log_prob.size(-1):  # [X, Y'] -> [X, Y] necessary
            log_prob = get_first_choice_probs(log_prob, metainfo['task'])
            log_unnorm_empty = get_first_choice_probs(log_unnorm_empty, metainfo['task'])
            log_unnorm_na = get_first_choice_probs(log_unnorm_na, metainfo['task'])
            log_unnorm_mask = get_first_choice_probs(log_unnorm_mask, metainfo['task'])

    prob, _ = get_normalized_probs_from_logprob(log_prob, norm_type='mean', Y_axis=-1) # [X, Y]

    X, Y = prob.size()

    return {
        'perplexity': perplexity,
        'prob': prob,
        'log_unnorm_prob': log_prob,
        'log_unnorm_empty': log_unnorm_empty.expand(X, Y),
        'log_unnorm_na': log_unnorm_na.expand(X, Y),
        'log_unnorm_mask': log_unnorm_mask.expand(X, Y),
    }

def get_tensor_dict_targets_prompts(dirname):
    prompt2prob_dict = {}
    for filepath in sorted(EVAL_DIRS.joinpath(dirname).glob("*.json")):
        with open(filepath) as f:
            data = json.load(f)
        raw = data.pop('raw')
        targets = raw.pop('targets')

        prompt = get_prompt_name_from_filepath(filepath)
        prompt2prob_dict[prompt] = get_tensor_probs_from_raw(raw, dirname)

    prompts = sorted(prompt2prob_dict.keys())
    from collections import defaultdict
    tensor_dict = defaultdict(list)
    for prompt in prompts:
        prob_dict = prompt2prob_dict[prompt]
        for key in prob_dict:
            tensor_dict[key].append(prob_dict[key])
    for key in tensor_dict:
        tensor_dict[key] = torch.stack(tensor_dict[key])
    return tensor_dict, torch.tensor(targets), prompts

def get_unbalanced_tensors_and_targets(tensor_dict, targets):
    for i, target in enumerate(targets):
        t = target.item()
        if t != 0:
            tensor_dict['prob'][:, i, [0, t]] = tensor_dict['prob'][:, i, [t, 0]]
            tensor_dict['log_unnorm_prob'][:, i, [0, t]] = tensor_dict['log_unnorm_prob'][:, i, [t, 0]]

    targets = torch.zeros(targets.size())
    return tensor_dict, targets

def get_entropy(prob, sum_axis, keepdims=False):
    return -(prob * safe_log(prob)).sum(axis=sum_axis, keepdims=keepdims)

def get_one_hot(prob, Y_axis=-1):    
    return torch.nn.functional.one_hot(prob.argmax(Y_axis), prob.size(Y_axis)).float()

def get_le(template_prob):
    return get_entropy(template_prob, sum_axis=-1).mean()

def get_ge(template_prob, X_axis=0, Y_axis=-1, one_hot=False, keepdims=False):
    if one_hot:
        template_prob = get_one_hot(template_prob)
        
    d_prob = template_prob.mean(axis=X_axis, keepdims=keepdims)
    entropy = get_entropy(d_prob, sum_axis=Y_axis, keepdims=keepdims)
    return entropy

def get_s_xy(tensor_dict_prob, method, T_axis=0):
    if method == 'ELP':
        s_xy = safe_log(tensor_dict_prob).mean(T_axis)
    elif method == 'EPM':
        s_xy = tensor_dict_prob.mean(T_axis)
    elif method == 'EMV':
        s_xy = get_one_hot(tensor_dict_prob, Y_axis=-1).sum(T_axis)
    else:
        raise ValueError(method)
    return s_xy

def get_ensemble(template_prob, tensor_dict_prob, method, Y_axis=-1):
    s_xy = get_s_xy(tensor_dict_prob, method)
    return (template_prob.argmax(Y_axis) == s_xy.argmax(Y_axis)).float().mean()

def get_metric_scores(tensor_dict_prob, tensor_dict_perplexity):
    ge_ds = []
    ges = []
    les = []
    mis = []
    mi_ds = []
    elps = []
    epms = []
    emvs = []
    ppls = []

    T, X, Y = tensor_dict_prob.size()

    for t in range(T):
        template_prob = tensor_dict_prob[t]
        ge_d = get_ge(template_prob, one_hot=True)
        ge = get_ge(template_prob, one_hot=False)
        le = get_le(template_prob)

        ge_ds.append(ge_d.item())
        ges.append(ge.item())
        les.append(le.item())

        mis.append((ge - le).item())
        mi_ds.append((ge_d - le).item())

        elps.append(get_ensemble(template_prob, tensor_dict_prob, 'ELP').item())
        epms.append(get_ensemble(template_prob, tensor_dict_prob, 'EPM').item())
        emvs.append(get_ensemble(template_prob, tensor_dict_prob, 'EMV').item())

        ppls.append(tensor_dict_perplexity[t].mean().item())

    return {
        'MI': mis,
        'MI_D': mi_ds,
        'GE_D': ge_ds,
        'GE': ges,
        'LE': les,
        'ELP': elps,
        'EPM': epms,
        'EMV': emvs,
        'PPL': ppls,
    }

def get_f1_acc(predscore, targets):
    f1s = []
    accs = []

    T, X, Y = predscore.size()

    for t in range(T):
        template_predscore = predscore[t]
        predictions = template_predscore.argmax(-1)

        f1s.append(f1_score(targets, predictions, average='macro'))
        accs.append(accuracy_score(targets, predictions))
    return f1s, accs

def get_calibrated_prob_and_predscore(norm_prob, log_unnorm_prob, tensor_dict, cali_type, norm_type, X_axis=1):
  # calibration applied on normalized prob

    if cali_type == 'ours':
        cali_unnorm_prob = norm_prob / norm_prob.mean(dim=X_axis, keepdims=True)
        cali_predscore = cali_unnorm_prob
        cali_prob = normalize_prob(cali_unnorm_prob, norm_type, -1)

    elif cali_type == 'cbu':
        
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

    elif cali_type == 'pmi':
        # https://github.com/peterwestuw/surface-form-competition/blob/ffc22c6352f41ae1519c370564c9f4cb993fed75/utils.py#L288 they used sum to aggregate the tokens, but let's ignore that

        # https://github.com/peterwestuw/surface-form-competition/blob/ffc22c6352f41ae1519c370564c9f4cb993fed75/utils.py#L287
        # https://github.com/peterwestuw/surface-form-competition/blob/ffc22c6352f41ae1519c370564c9f4cb993fed75/utils.py#L399
        log_cali_unnorm_prob = log_unnorm_prob - tensor_dict['log_unnorm_empty']

        cali_predscore = log_cali_unnorm_prob  # original score
        cali_prob, _ = get_normalized_probs_from_logprob(log_cali_unnorm_prob, norm_type, -1)

    return cali_prob, cali_predscore

def get_imetric_scores(tensor_dict_prob, tensor_dict_perplexity):
    score_dict = defaultdict(list)
    prompt_dict = defaultdict(list)

    tensor_dict_prob = tensor_dict_prob.transpose(0, 1)  # [X, T, Y]
    tensor_dict_perplexity = tensor_dict_perplexity.transpose(0, 1)  # [X, T]

    ge = get_ge(tensor_dict_prob, X_axis=0, Y_axis=-1, one_hot=False)  # [T]
    ge_d = get_ge(tensor_dict_prob, X_axis=0, Y_axis=-1, one_hot=True)  # [T]

    X, T, Y = tensor_dict_prob.size()

    for i in range(X):
        instance_prob = tensor_dict_prob[i]  # [T, Y]
        
        le_i = get_entropy(instance_prob, -1)  # [T]
        neg_le_i = -le_i
        mi_i = ge - le_i
        mi_di = ge_d - le_i
        ppl_i = tensor_dict_perplexity[i]  # [T]

        score_dict['LE'].append(le_i.max().item())
        prompt_dict['LE'].append(le_i.argmax().item())

        score_dict['-LE'].append(le_i.min().item())
        prompt_dict['-LE'].append(le_i.argmin().item())

        score_dict['MI'].append(mi_i.max().item())
        prompt_dict['MI'].append(mi_i.argmax().item())

        score_dict['MI_D'].append(mi_di.max().item())
        prompt_dict['MI_D'].append(mi_di.argmax().item())

        score_dict['PPL'].append(ppl_i.max().item())
        prompt_dict['PPL'].append(ppl_i.argmax().item())

        score_dict['-PPL'].append(ppl_i.min().item())
        prompt_dict['-PPL'].append(ppl_i.argmin().item())

    return score_dict, prompt_dict

def get_filtered_prompts(prob):
    T, X, Y = prob.size()

    top2_prob = prob.topk(k=2, dim=-1, sorted=True)[0]
    conf_scores = (top2_prob[:, :, 0] - top2_prob[:, :, 1]).sum(1, keepdims=True)
    try:
        kmeans = KMeans(n_clusters=2, random_state=42, n_init='auto').fit(conf_scores)
    except ValueError:  # only one cluster
        return np.arange(T)
    return np.arange(T)[(kmeans.labels_ == kmeans.cluster_centers_.argmax(axis=0))]

def apply_filter(prob, tensor_dict, prompts):
    filtered_prompts = get_filtered_prompts(prob)

    # apply filter to prob, tensor_dict, prompts
    prob = prob[filtered_prompts, ...]
    new_tensor_dict = dict()
    for key in tensor_dict:
        new_tensor_dict[key] = tensor_dict[key][filtered_prompts, ...]
    prompts = np.asarray(prompts)[filtered_prompts].tolist()

    return prob, new_tensor_dict, prompts

def get_prob_for_scoring(tensor_dict, cali_score, cali_type, cali_norm_type):
    score_prob = tensor_dict['prob']

    if cali_score:
        score_prob, _ = get_calibrated_prob_and_predscore(score_prob, tensor_dict['log_unnorm_prob'], tensor_dict, cali_type, cali_norm_type, X_axis=1)
    
    return score_prob

def get_predscore(tensor_dict, cali_eval, cali_type, cali_norm_type):
    if cali_eval:
        _, predscore = get_calibrated_prob_and_predscore(tensor_dict['prob'], tensor_dict['log_unnorm_prob'], tensor_dict, cali_type, cali_norm_type, X_axis=1)
    else:
        predscore = tensor_dict['log_unnorm_prob']  # no need to normalize cuz we just argmax
    return predscore

def dir_metric(dirname, cali_type, cali_norm_type='mean', do_filter=False, unbalance=False):
    print(f'{dirname} cali={cali_type} cali_norm_type={cali_norm_type} do_filter={do_filter} unbalance={unbalance}')
    metainfo = get_metainfo_from_dirname(dirname)
    model = metainfo['model']
    task = metainfo['task']
    utask = "U_" + task

    dfs = []

    for combn, cali_score, cali_eval in [
        ('NN', False, False),
        ('NC', False, True),
        ('CN', True, False),
        ('CC', True, True),
    ]:
        tensor_dict, targets, prompts = get_tensor_dict_targets_prompts(dirname)
        taskname = task

        if unbalance:
            tensor_dict, targets = get_unbalanced_tensors_and_targets(tensor_dict, targets)
            taskname = utask

        # SCORE CALCULATION
        score_prob = get_prob_for_scoring(tensor_dict, cali_score, cali_type, cali_norm_type)

        # and then filter using the calibrated score
        if do_filter:
            score_prob, tensor_dict, prompts = apply_filter(score_prob, tensor_dict, prompts)

        metric_scores = get_metric_scores(score_prob, tensor_dict['perplexity'])  # perplexity is not affected by calibration

        # PREDICTION
        predscore = get_predscore(tensor_dict, cali_eval, cali_type, cali_norm_type)
        f1, acc = get_f1_acc(predscore, targets)
        dfs.append(pd.DataFrame({
            'model': [model] * len(acc),
            'task': [taskname] * len(acc),
            'prompt': prompts,
            'combn': [combn] * len(acc),
            'acc': acc,
            'f1': f1,
        } | metric_scores))
    return pd.concat(dfs, ignore_index=True)

def dir_imetric(dirname, cali_type, cali_norm_type='mean', do_filter=False, unbalance=False):
    print(f'{dirname} cali={cali_type} cali_norm_type={cali_norm_type} do_filter={do_filter} unbalance={unbalance}')
    metainfo = get_metainfo_from_dirname(dirname)
    model = metainfo['model']
    task = metainfo['task']
    utask = "U_" + task

    df_rows = []

    for combn, cali_score, cali_eval in [
        ('NN', False, False),
        ('NC', False, True),
        ('CN', True, False),
        ('CC', True, True),
    ]:
        tensor_dict, targets, prompts = get_tensor_dict_targets_prompts(dirname)
        taskname = task

        if unbalance:
            tensor_dict, targets = get_unbalanced_tensors_and_targets(tensor_dict, targets)
            taskname = utask

        # SCORE CALCULATION
        score_prob = get_prob_for_scoring(tensor_dict, cali_score, cali_type, cali_norm_type)

        # and then filter using the calibrated score
        if do_filter:
            score_prob, tensor_dict, prompts = apply_filter(score_prob, tensor_dict, prompts)

        score_dict, prompt_dict = get_imetric_scores(score_prob, tensor_dict['perplexity'])

        # PREDICTION
        predscore = get_predscore(tensor_dict, cali_eval, cali_type, cali_norm_type)  # [T, X, Y]

        T, X, Y = predscore.size()
        
        for method in score_dict.keys():
            for i in range(X):
                selected_prompt = prompt_dict[method][i]
                prediction = predscore[selected_prompt][i].argmax().item()
                target = targets[i].item()

                df_rows.append({
                    'model': model,
                    'task': taskname,
                    'combn': combn,
                    'instance': i,
                    'method': method,
                    'prompt': prompts[selected_prompt],
                    'prediction': prediction,
                    'target': target,
                    'score': score_dict[method][i],
                    'correct': prediction == target,
                })

    return pd.DataFrame(df_rows)

def get_imetric(setup, token='all', cali_type='ours', cali_norm_type='mean', do_filter=False):
    df = get_all_dirnames_df()
    dfs = []
    for dirname in tqdm(df[(df['setup'] == setup) & (df['token'] == token)].dirname):
        dfs.append(dir_imetric(dirname, cali_type, cali_norm_type, do_filter))
        if any([x in dirname for x in ['piqa', 'copa', 'hellaswag', 'storycloze']]):
            dfs.append(dir_imetric(dirname, cali_type, cali_norm_type, do_filter, unbalance=True))
    if not dfs:
        return None
    metric = pd.concat(dfs, ignore_index=True)
    return metric

def get_metric(setup, token='all', cali_type='ours', cali_norm_type='mean', do_filter=False):
    df = get_all_dirnames_df()
    dfs = []
    for dirname in tqdm(df[(df['setup'] == setup) & (df['token'] == token)].dirname):
        dfs.append(dir_metric(dirname, cali_type, cali_norm_type, do_filter))
        if any([x in dirname for x in ['piqa', 'copa', 'hellaswag', 'storycloze']]):
            dfs.append(dir_metric(dirname, cali_type, cali_norm_type, do_filter, unbalance=True))
    if not dfs:
        return None
    metric = pd.concat(dfs, ignore_index=True)
    return metric

def get_dict_key(option):
    return '__'.join([f'{key}={value}' for key, value in option.items()])

def get_metric_filename(**option):
    metric_type = option.pop('metric_type')
    prefix = {
        'metric': '',
        'ometric': 'o',
        'imetric': 'i',
    }[metric_type]

    return get_dict_key(option) + f'.{prefix}metric.csv'

def get_metric_df_and_filename(metric_type, setup, cali_type, cali_norm_type, do_filter):
    token = 'all'
    if metric_type == 'ometric':
        token = 'first'

    if metric_type == 'imetric':
        fn = get_imetric
    else:
        fn = get_metric

    df = fn(setup, token=token, cali_type=cali_type, cali_norm_type=cali_norm_type, do_filter=do_filter)
    name = get_metric_filename(metric_type=metric_type, setup=setup, cali_type=cali_type, cali_norm_type=cali_norm_type, do_filter=do_filter)
    return df, name


if __name__ == "__main__":
    save_dir = CWD.joinpath("ps_results", "metrics")
    save_dir.mkdir(exist_ok=True)

    for cali_norm_type in ['mean', 'softmax']:
        for do_filter in [False, True]:
            for metric_type in ['metric', 'ometric', 'imetric']:
                for setup in ['v12', 'v2', 'few', 'v1']:
                    for cali_type in ['ours', 'cbu', 'pmi']:
                        logging.info(f"\nmetric_type: {metric_type}\nsetup: {setup}\ncali_type: {cali_type}\ncali_norm_type: {cali_norm_type}\ndo_filter: {do_filter}\n\n")
                        filename = get_metric_filename(metric_type=metric_type, setup=setup, cali_type=cali_type, cali_norm_type=cali_norm_type, do_filter=do_filter)
                        if save_dir.joinpath(filename) in save_dir.glob("*.csv"):
                            continue
                        try:
                            metric, filename = get_metric_df_and_filename(metric_type, setup, cali_type, cali_norm_type, do_filter)
                            if metric is not None:
                                metric.to_csv(save_dir.joinpath(filename))
                                logging.info(f"\nSave the resulting metric to the following path:\n\n'{save_dir.joinpath(filename)}'\n")
                        except:
                            logging.error(traceback.format_exc())