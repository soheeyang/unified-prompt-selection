#!/usr/bin/env python
# coding=utf-8
# Copyright BigScience, The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
'''
This script is heavily adapted from https://github.com/bigscience-workshop/t-zero/blob/master/t0/model.py
'''

import logging
from typing import Optional, List, Dict
from abc import ABC, abstractmethod

import torch
from torch import nn, Tensor
from transformers import ( 
    AutoModelForCausalLM, 
    MODEL_FOR_CAUSAL_LM_MAPPING,
    OPTForCausalLM,
)
import transformers

logger = logging.getLogger(__name__)

class ModelBase(nn.Module, ABC):
    @abstractmethod
    def forward(self, batch):
        pass

    @staticmethod
    def from_config(config, first_token: bool, **kwargs) -> "ModelBase":
        config_name = config.__class__
        transformer_model_name = MODEL_FOR_CAUSAL_LM_MAPPING.get(config_name, None)
        if transformer_model_name is None:
            raise NotImplementedError
        
        if first_token:
            return DecoderForPromptSelectionOTR(config=config, **kwargs)
        else:
            return DecoderForPromptSelection(config=config, **kwargs)


class DecoderModel(ModelBase):
    def __init__(
        self, 
        config,
        model_name_or_path: Optional[str], 
        parallelize: bool, 
        sum_log_prob: bool,
        ignore_index: int = -100,
    ):
        super(DecoderModel, self).__init__()
        logger.info("Building DecoderModel")
        model_args = {"config": config}

        if parallelize:
            assert torch.cuda.is_available(), (
                """
                You need at least 1 GPU to call `parallelize` 
                (even though if there is only 1 GPU, there won't be any model parallelism).
                """
            )
            model_args.update({"device_map": "auto"})

        if model_name_or_path:
            model_args.update({"pretrained_model_name_or_path": model_name_or_path})
            if model_name_or_path.startswith("facebook/opt"):
                self._model = OPTForCausalLM.from_pretrained(**model_args)
            else:
                self._model = AutoModelForCausalLM.from_pretrained(**model_args)
        else:
            logger.info("Using new model from scratch")
            self._model = AutoModelForCausalLM.from_config(**model_args)

        self.sum_log_prob = sum_log_prob
        self.ignore_index = ignore_index

    def forward(self, batch: Dict[str, Tensor]) -> Tensor:
        model_inputs = self.get_model_inputs(batch)
        logits = self._model(**model_inputs).logits

        self.log_prob = self.get_label_log_prob(batch, logits)
        self.seq_Ps = self.get_seq_prob(batch, logits)

        predictions = self.log_prob.argmax(dim=-1)
        return predictions
    
    @abstractmethod
    def make_model_inputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        pass
    
    @abstractmethod
    def get_seq_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> List[float]:
        pass

    @abstractmethod
    def get_label_log_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        pass
    
    def get_device(self, input_ids: Tensor) -> str:
        return input_ids.device
    
    def get_batch_size(self, input_ids: Tensor) -> int:
        return input_ids.size(0)
    
    def get_prefix_length(self, input_ids: Tensor) -> int:
        assert input_ids.dim() == 2, (
            "Dimension of input_ids must be 2 to get correct prefix length"
        )
        return input_ids.size(1)
    
    def get_position_ids(self, model_inputs: Dict[str, Tensor]) -> Tensor:
        assert "input_ids" in model_inputs and "attention_mask" in model_inputs, (
            "batch must contain 'input_ids' and 'attention_mask' to get position_ids"
        )
        input_ids, attention_mask = model_inputs['input_ids'], model_inputs['attention_mask']
        device = self.get_device(input_ids)
        position_ids = torch.maximum(
            torch.cumsum(attention_mask.to(torch.long), dim=-1) - 1,
            torch.zeros(1, dtype=torch.long, device=device)[None, None]
        )
        return position_ids
    
    def get_model_inputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model_inputs = self.make_model_inputs(batch)
        # Set position ids correctly to take care of padding tokens between inputs_ids and labels
        if not isinstance(self._model, transformers.OPTForCausalLM):
            position_ids = self.get_position_ids(model_inputs)
            model_inputs["position_ids"] = position_ids
        return model_inputs


class DecoderForPromptSelection(DecoderModel):
    def __init__(
        self,
        config,
        model_name_or_path: Optional[str], 
        parallelize: bool, 
        sum_log_prob: bool,
        ignore_index: int = -100,
    ):
        super(DecoderForPromptSelection, self).__init__(
            config,
            model_name_or_path, 
            parallelize, 
            sum_log_prob,
            ignore_index,
        )
    
    def make_model_inputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model_inputs = {
            "input_ids": torch.cat([batch["input_ids"], batch["labels"]], dim=-1),
            "attention_mask": torch.cat([batch["attention_mask"], batch["labels_attention_mask"]], dim=-1),
        }
        return model_inputs
    
    def get_pred_logits(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        prefix_length = self.get_prefix_length(batch['input_ids'])
        return logits[:, prefix_length-1:-1]
    
    def get_label_log_prob_sum(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        pred_logits = self.get_pred_logits(batch, logits)
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(pred_logits, dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
        return seq_log_prob

    def get_label_log_prob_mean(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        pred_logits = self.get_pred_logits(batch, logits)
        label_mask = ~batch["labels_attention_mask"].unsqueeze(-1).bool()
        log_probs = torch.log_softmax(pred_logits, dim=-1)
        seq_token_log_probs = torch.gather(log_probs, -1, batch["labels"].unsqueeze(-1))
        masked_seq_token_log_probs = seq_token_log_probs.masked_fill(label_mask, torch.nan)
        seq_log_prob = masked_seq_token_log_probs.squeeze(dim=-1).nanmean(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0), -1)
        return seq_log_prob
    
    def get_label_log_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        if self.sum_log_prob:
            return self.get_label_log_prob_sum(batch, logits)
        else:
            return self.get_label_log_prob_mean(batch, logits)
        
    def get_num_class(self, targets: Tensor, batch_size: int) -> int:
        return int(batch_size / targets.size(0))

    def get_offset(self, targets: Tensor, batch_size: int) -> Tensor:
        num_class = self.get_num_class(targets, batch_size)
        return torch.arange(start=0, end=batch_size, step=num_class)
    
    def get_seq_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> List[float]:
        # To extract sequence probs
        batch_size = self.get_batch_size(batch['input_ids'])
        prefix_length = self.get_prefix_length(batch['input_ids'])
        offset = self.get_offset(batch['targets'], batch_size)

        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        labels = batch['input_ids'][offset, 1:prefix_length]
        logits = logits[offset, :prefix_length-1]
        seq_Ps = [
            torch.exp(-criterion(logits[i], labels[i])).cpu().item()
            for i in range(batch['targets'].size(0))
        ]
        return seq_Ps


class DecoderForPromptSelectionOTR(DecoderModel):
    def __init__(
        self,
        config,
        model_name_or_path: Optional[str], 
        parallelize: bool, 
        sum_log_prob: bool,
        ignore_index: int = -100,
        otr_label_indices: Optional[Tensor] = None,
    ):
        super(DecoderForPromptSelectionOTR, self).__init__(
            config,
            model_name_or_path, 
            parallelize, 
            sum_log_prob,
            ignore_index,
        )
        self.otr_label_indices = otr_label_indices
    
    def set_otr_label_indices(self, otr_label_indices: Tensor) -> None:
        self.otr_label_indices = otr_label_indices
    
    def make_model_inputs(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        model_inputs = {"input_ids": batch['input_ids'], "attention_mask": batch['attention_mask']}
        return model_inputs
    
    def otr_log_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        assert self.otr_label_indices is not None, (
            "For OTR, model object must have 'otr_label_indices' attribute"
        )
        device = self.get_device(batch['input_ids'])
        batch_size = self.get_batch_size(batch['input_ids'])

        last_logit = logits[:, -1, :].unsqueeze(1).to(torch.float64)
        logprobs = torch.log_softmax(last_logit, dim=-1)
        otr_label_indices = self.otr_label_indices.repeat(batch_size).view(batch_size, 1, -1).contiguous()
        otr_log_prob = torch.gather(logprobs, -1, otr_label_indices.to(device)).squeeze(1)
        return otr_log_prob
    
    def get_label_log_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> Tensor:
        otr_log_prob = self.otr_log_prob(batch, logits)
        seq_log_prob = torch.gather(otr_log_prob, -1, batch['labels'])
        return seq_log_prob
    
    def get_seq_prob(self, batch: Dict[str, Tensor], logits: Tensor) -> List[float]:
        # To extract sequence probs
        prefix_length = self.get_prefix_length(batch['input_ids'])
        criterion = nn.CrossEntropyLoss(reduction='mean', ignore_index=self.ignore_index)
        labels = batch['input_ids'][:, 1:prefix_length]
        logits = logits[:, :prefix_length-1].to(torch.float64)
        seq_Ps = [
            torch.exp(-criterion(logits[i], labels[i])).cpu().item()
            for i in range(batch['input_ids'].size(0))
        ]
        return seq_Ps