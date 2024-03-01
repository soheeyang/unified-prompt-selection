import json
import torch
from collections import defaultdict
from extraction.loader import OutputLoader
from method.utils import (
    get_normalized_probs_from_logprob
)
from torch import Tensor
from typing import Dict, Tuple, List


class PostProcessor:
    '''
    Convert the extracted p(y|x,t) into a torch.Tensor format for calculating scores based on each prompt selection method.
    '''
    
    def __init__(self, cfg: str) -> Dict[str, Tensor]:
        self.eval_loader = OutputLoader(cfg)
    
    def get_tensor_probs_from_raw(self, raw: dict):
        prob_dict = dict()
        for key in ['P(x,t)', 'log_prob', 'empty_log', 'na_log', 'mask_log']:
            prob_dict[key] = torch.tensor(raw[key])

        perplexity = 1 / prob_dict['P(x,t)'] # [X]

        log_prob = prob_dict['log_prob']  # [X, Y]
        log_unnorm_empty = prob_dict['empty_log'].expand(*log_prob.size())  # [1, Y] -> [X, Y]
        log_unnorm_na = prob_dict['na_log'].expand(*log_prob.size())
        log_unnorm_mask = prob_dict['mask_log'].expand(*log_prob.size())

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
    
    def get_tensor_dict_and_targets(self) -> Tuple[Dict[str, Tensor], Tensor]:
        prompt2prob_dict = {}
        filepaths = self.eval_loader.get_all_eval_result_filepath()
        for filepath in sorted(filepaths):
            with open(filepath) as f:
                data = json.load(f)
            raw = data.pop('raw')
            targets = raw.pop('targets')
            filename = self.eval_loader.get_filename_from_filepath(filepath)
            prompt = self.eval_loader.get_prompt_name_from_filename(filename)
            prompt2prob_dict[prompt] = self.get_tensor_probs_from_raw(raw)

        prompts = sorted(prompt2prob_dict.keys())
        tensor_dict = defaultdict(list)
        for prompt in prompts:
            prob_dict = prompt2prob_dict[prompt]
            for key in prob_dict:
                tensor_dict[key].append(prob_dict[key])
        for key in tensor_dict:
            tensor_dict[key] = torch.stack(tensor_dict[key])
        return tensor_dict, torch.tensor(targets)
    
    def get_unbalanced_tensor_dict_and_targets(self) -> Tuple[Dict[str, Tensor], Tensor]:
        tensor_dict, targets = self.get_tensor_dict_and_targets()
        for i, target in enumerate(targets):
            t = target.item()
            if t != 0:
                tensor_dict['prob'][:, i, [0, t]] = tensor_dict['prob'][:, i, [t, 0]]
                tensor_dict['log_unnorm_prob'][:, i, [0, t]] = tensor_dict['log_unnorm_prob'][:, i, [t, 0]]

        targets = torch.zeros(targets.size(), dtype=torch.int32)
        return tensor_dict, targets
    
    def get_tasks_models_prompts_tokens(self) -> Tuple[List[str], List[str], List[str], List[str]]:
        tasks_models_prompts_tokens = []
        tasks, models, prompts, tokens = [], [], [], []

        filepaths = self.eval_loader.get_all_eval_result_filepath()
        for filepath in filepaths:
            filename = self.eval_loader.get_filename_from_filepath(filepath)
            task = self.eval_loader.get_task_name_from_filename(filename)
            model = self.eval_loader.get_model_name_from_filename(filename)
            prompt = self.eval_loader.get_prompt_name_from_filename(filename)
            token = self.eval_loader.get_token_from_filename(filename)
            tasks_models_prompts_tokens.append( (task, model, prompt, token) )
        
        for t, m, p, to in sorted(tasks_models_prompts_tokens, key=lambda x: x[2]):
            tasks.append(t)
            models.append(m)
            prompts.append(p)
            tokens.append(to)
        
        return tasks, models, prompts, tokens