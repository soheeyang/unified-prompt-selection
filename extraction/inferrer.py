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
"""
Reproduce the p(y|x,t) extraction in `Improving Probability-based Prompt Selection Through Unified Evaluation and Analysis` using PyTorch.

This script is heavily adapted from https://github.com/bigscience-workshop/t-zero/blob/master/evaluation/run_eval.py
"""

import logging
import traceback
import torch
from collections import defaultdict
from itertools import permutations
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from accelerate import Accelerator
from extraction.model.data_collator import DataCollatorForMultipleChoice
from transformers import DataCollatorWithPadding
from datasets import Dataset
from extraction.utils import *
from extraction.saver import OutputSaver
from extraction.preprocess import (
    PreprocessorForInference, 
    ConverterForInference,
)
from abc import ABC, abstractmethod
from typing import List
from omegaconf import DictConfig


logger = logging.getLogger(__name__)

class Inferrer(ABC):
    def __init__(self, cfg: DictConfig):
        set_seed(cfg.seed)
        self.seed = cfg.seed
        self.first_token = cfg.first_token
        self.dataset_name = cfg.dataset.dataset_name
        self.dataset_config_name = cfg.dataset.dataset_config_name
        self.model_name_or_path = cfg.decoder.model_name_or_path
        self.template_names = cfg.prompt.template_names
        self.max_length = cfg.decoder.max_length
        self.parallelize = cfg.decoder.parallelize
        self.per_device_eval_batch_size = cfg.decoder.per_device_eval_batch_size
        self.num_samples = cfg.num_samples
        self.mixed_precision = cfg.mixed_precision
    
        # Initialize the accelerator. We will let the accelerator handle device placement for us.
        self.accelerator = Accelerator(mixed_precision=self.mixed_precision)
        self.saver = OutputSaver(
            self.dataset_name,
            self.dataset_config_name,
            self.model_name_or_path,
            self.first_token,
            self.num_samples,
        )
        set_logger(logger, self.accelerator)

        # Handle the output directory creation
        if self.accelerator.is_main_process:
            self.saver.create_output_dir(cfg)
        self.accelerator.wait_for_everyone()

        # In distributed evaluation, the load_dataset function guarantee that only one local process can concurrently
        # download the dataset.
        self.raw_datasets = load_shuffled_subset_raw_datasets(cfg)
        self.column_names = self.raw_datasets.column_names
        self.empty_input = "N/A,[MASK],"

        # In distributed training, the .from_pretrained methods guarantee that only one local process can concurrently
        # download model & vocab.
        self.model, self.tokenizer = load_pretrained_model_and_tokenizer(cfg)
    
    @abstractmethod
    def extract(self):
        pass


class InferrerForZeroshot(Inferrer):
    def __init__(self, cfg: DictConfig):
        super(InferrerForZeroshot, self).__init__(cfg)

    def extract(self):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        for template_name in self.template_names:
            try:
                self.accelerator.print("***** Running template {} *****".format(template_name))

                template = get_template(
                    template_name, self.dataset_name, self.dataset_config_name
                )
                preprocessor = PreprocessorForInference(
                    self.tokenizer,
                    template,
                    self.dataset_name,
                    self.dataset_config_name,
                    self.column_names,
                    self.max_length
                )

                def preprocess_for_zeroshot(examples, fewshot='', forced_input=None):
                    return preprocessor(
                        examples,
                        fewshot=fewshot,
                        forced_input=forced_input,
                    )
                
                with self.accelerator.main_process_first():
                    eval_dataset = self.raw_datasets.map(
                        preprocess_for_zeroshot, batched=True, remove_columns=self.column_names
                    )

                # Log a few random samples from the eval set:
                log_a_few_random_samples(logger, eval_dataset)

                # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                data_collator = DataCollatorForMultipleChoice(
                    self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.mixed_precision == 'fp16' else None)
                )
                eval_dataloader = DataLoader(eval_dataset, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size)
                
                # Use the device given by the `accelerator` object.
                if not self.parallelize:
                    self.model.to(self.accelerator.device)

                # Prepare everything with our `accelerator`.
                eval_dataloader = self.accelerator.prepare(eval_dataloader)

                with self.accelerator.main_process_first():
                    raw_data = self.raw_datasets.select([1])
                    empty_inputs = self.empty_input.split(",")
                    eval_empty_input_datasets = get_eval_empty_input_datasets(
                        preprocess_for_zeroshot, self.column_names, raw_data, empty_inputs
                    )
                    eval_empty_input_dataloaders = get_eval_empty_input_dataloaders(
                        data_collator, self.per_device_eval_batch_size, eval_empty_input_datasets,
                    )
                    eval_empty_input_iterators = get_eval_empty_input_iterators(
                        self.accelerator, eval_empty_input_dataloaders
                    )

                # Eval!
                total_batch_size = self.per_device_eval_batch_size * self.accelerator.num_processes

                logger.info("***** Running evaluation *****")
                logger.info(f"  Num examples = {len(eval_dataset)}")
                logger.info(f"  Instantaneous batch size per device = {self.per_device_eval_batch_size}")
                logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
                # Only show the progress bar once on each machine.
                progress_bar = tqdm(range(len(eval_dataloader)), disable=not self.accelerator.is_local_main_process)
                result_dict = defaultdict(list)
                self.model.eval()
                for batch in eval_dataloader:
                    with torch.no_grad():
                        while eval_empty_input_iterators:
                            eval_empty_input_iterator = eval_empty_input_iterators.pop()
                            empty_input = empty_inputs.pop()
                            empty_input_batch = next(eval_empty_input_iterator)  # proceed to next batch
                            if empty_input_batch["input_ids"].nelement() == 0:
                                raise ValueError("Empty prompt is given. Cannot apply PMI or MI.")
                            
                            _ = self.model(empty_input_batch)
                            result_dict = update_result_dict_for_empty(
                                result_dict, empty_input, self.model
                            )

                        predictions = self.model(batch)
                        result_dict = update_result_dict(
                            result_dict, batch["input_ids"], batch["targets"], predictions, self.model
                        )
                    
                    progress_bar.update(1)

                results = get_results(
                    result_dict,
                    self.tokenizer,
                    self.dataset_name,
                    self.dataset_config_name,
                    template_name,
                    None,
                    None,
                )

                if self.accelerator.is_main_process:
                    self.saver.save_results(results, template_name, None)
                    
            except:
                error_msg = traceback.format_exc()
                logger.error(f"\n{error_msg}")


class InferrerForFewshot(Inferrer):
    def __init__(self, cfg: DictConfig):
        super(Inferrer, self).__init__(cfg)
        self.template_name = 'fewshot'
        self.train_datasets = load_train_datasets(cfg.dataset.DATASET_KWARGS)
        self.train_datasets = shuffle_dataset(self.train_datasets, self.seed)

        self.num_shots = cfg.fewshot.split(',')
        self.shot2range = {
            '4': (list(range(3)), list(range(24))),
            '2': (list(range(10)), list(range(2))),
            '1': (list(range(8)), list(range(1))),
        }
        self.shot2indices = {
            '4': [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
            '2': [[13, 14], [15, 16], [17, 18], [19, 20], [21, 22], 
                [23, 24], [25, 26], [27, 28], [29, 30], [31, 32]],
            '1': [[33], [34], [35], [36], [37], [38], [39], [40]]
        }

        self.shot2permutations = {
            '4': list(permutations([0, 1, 2, 3], 4)),
            '2': list(permutations([0, 1], 2)),
            '1': list(permutations([0], 1)),
        }
    
    def get_fewshot_indices(self, shot: str, set_idx: int, permutation_idx: int) -> List[int]:
        '''
        | shot | set range | permutation range |
        |   4  |   0 ~ 2   |       0 ~ 23      |
        |   2  |   0 ~ 4   |       0 ~ 1       |
        |   1  |   0 ~ 7   |         0         |
        '''

        def permute(a, b):
            return [a[idx] for idx in b]
        set_indices = self.shot2indices[shot][set_idx]
        permutation = self.shot2permutations[shot][permutation_idx]
        return permute(set_indices, permutation)
    
    def extract(self):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        template = get_template(
            self.template_name, self.dataset_name, self.dataset_config_name
        )
        preprocessor = PreprocessorForInference(
            self.tokenizer,
            template,
            self.dataset_name,
            self.dataset_config_name,
            self.column_names,
            self.max_length
        )
        for num_shot in self.num_shots:
            set_range, permutation_range = self.shot2range[num_shot]
            for set_idx in set_range:
                for permutation_idx in permutation_range:
                    fewshot_indices = self.get_fewshot_indices(num_shot, set_idx, permutation_idx)
                    fewshot = self.train_datasets.select(fewshot_indices)
                    labeled_examples = []
                    for shot in fewshot:
                        input, target = template.apply(shot)
                        labeled_examples.append(f"{input} {target}")
                    fewshot = "\n\n".join(labeled_examples) + "\n\n"

                    try:
                        self.accelerator.print("***** Running template {} *****".format(self.template_name))

                        def preprocess_for_fewshot(examples, fewshot=fewshot, forced_input=None):
                            return preprocessor(
                                examples,
                                fewshot=fewshot,
                                forced_input=forced_input,
                            )
                        
                        with self.accelerator.main_process_first():
                            eval_dataset = self.raw_datasets.map(
                                preprocess_for_fewshot, batched=True, remove_columns=self.column_names
                            )

                        # Log a few random samples from the eval set:
                        log_a_few_random_samples(logger, eval_dataset)

                        # Otherwise, `DataCollatorWithPadding` will apply dynamic padding for us (by padding to the maximum length of
                        # the samples passed). When using mixed precision, we add `pad_to_multiple_of=8` to pad all tensors to multiple
                        # of 8s, which will enable the use of Tensor Cores on NVIDIA hardware with compute capability >= 7.5 (Volta).
                        data_collator = DataCollatorForMultipleChoice(
                            self.tokenizer, pad_to_multiple_of=(8 if self.accelerator.mixed_precision == 'fp16' else None)
                        )
                        eval_dataloader = DataLoader(
                            eval_dataset, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size
                        )
                        
                        # Use the device given by the `accelerator` object.
                        if not self.parallelize:
                            self.model.to(self.accelerator.device)

                        # Prepare everything with our `accelerator`.
                        eval_dataloader = self.accelerator.prepare(eval_dataloader)

                        with self.accelerator.main_process_first():
                            raw_data = self.raw_datasets.select([1])
                            empty_inputs = self.empty_input.split(",")
                            eval_empty_input_datasets = get_eval_empty_input_datasets(
                                preprocess_for_fewshot, self.column_names, raw_data, empty_inputs
                            )
                            eval_empty_input_dataloaders = get_eval_empty_input_dataloaders(
                                data_collator, self.per_device_eval_batch_size, eval_empty_input_datasets,
                            )
                            eval_empty_input_iterators = get_eval_empty_input_iterators(
                                self.accelerator, eval_empty_input_dataloaders
                            )

                        # Infer!
                        total_batch_size = self.per_device_eval_batch_size * self.accelerator.num_processes

                        logger.info("***** Running evaluation *****")
                        logger.info(f"  Num examples = {len(eval_dataset)}")
                        logger.info(f"  Instantaneous batch size per device = {self.per_device_eval_batch_size}")
                        logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
                        # Only show the progress bar once on each machine.
                        progress_bar = tqdm(range(len(eval_dataloader)), disable=not self.accelerator.is_local_main_process)
                        result_dict = defaultdict(list)
                        self.model.eval()
                        for batch in eval_dataloader:
                            with torch.no_grad():
                                while eval_empty_input_iterators:
                                    eval_empty_input_iterator = eval_empty_input_iterators.pop()
                                    empty_input = empty_inputs.pop()
                                    empty_input_batch = next(eval_empty_input_iterator)  # proceed to next batch
                                    if empty_input_batch["input_ids"].nelement() == 0:
                                        raise ValueError("Empty prompt is given. Cannot apply PMI or MI.")
                                    
                                    _ = self.model(empty_input_batch)
                                    result_dict = update_result_dict_for_empty(
                                        result_dict, empty_input, self.model
                                    )

                                predictions = self.model(batch)
                                result_dict = update_result_dict(
                                    result_dict, batch["input_ids"], batch["targets"], predictions, self.model
                                )
                            
                            progress_bar.update(1)

                        results = get_results(
                            result_dict,
                            self.tokenizer,
                            self.dataset_name,
                            self.dataset_config_name,
                            self.template_name,
                            None,
                            None,
                        )

                        if self.accelerator.is_main_process:
                            fewshot_info = f"{num_shot}_{set_idx}_{permutation_idx}"
                            self.saver.save_results(results, self.template_name, None, fewshot_info)
                            
                    except:
                        error_msg = traceback.format_exc()
                        logger.error(f"\n{error_msg}")


class InferrerForOTR(Inferrer):
    def __init__(self, cfg: DictConfig):
        super(InferrerForOTR, self).__init__(cfg)

        # Prepare for OTR
        self.task = get_task_name(self.dataset_name, self.dataset_config_name)
        self.df_raw = get_raw_df_for_otr(self.raw_datasets, self.task)
        self.num_classes, self.choice_cols, self.label_col, self.is_dynamic = get_dataset_info_for_otr(cfg.dataset.DATASET_INFO)
        vocabs = dict(sorted(self.tokenizer.vocab.items(), key=lambda x: x[1], reverse=False))
        self.vocabs = {self.tokenizer.decode(token_id): token_id for token_id in vocabs.values()}

    def extract(self):
        # Preprocessing the datasets.
        # First we tokenize all the texts.
        for template_name in self.template_names:
            try:
                self.accelerator.print("***** Running template {} *****".format(template_name))
                template = get_template(
                    template_name, self.dataset_name, self.dataset_config_name
                )
                converter = ConverterForInference(
                    self.task,
                    self.label_col,
                    self.choice_cols,
                    self.num_classes,
                    self.tokenizer,
                    template,
                    self.dataset_name,
                    self.dataset_config_name,
                    self.column_names,
                    self.max_length,   
                )
                converter.set_padding_side_to_right()

                if self.is_dynamic:
                    def preprocess_for_otr(df_raw, forced_input=None):
                        return converter.convert_dynamic_into_otr(
                            df_raw=df_raw,
                            forced_input=forced_input
                        )
                else:
                    def preprocess_for_otr(df_raw, forced_input=None):
                        return converter.convert_static_into_otr(
                            df_raw=df_raw,
                            forced_input=forced_input
                        )

                input_ids, labels, otr_labels, targets = preprocess_for_otr(self.df_raw)
                otr_label2id = {label: _id for _id, label in enumerate(otr_labels)}
                otr_label_indices = torch.tensor(list(otr_label2id.keys()))
                label_indices = [
                    [otr_label2id[otr_label] for otr_label in instance_labels] 
                    for instance_labels in labels
                ]
                eval_dict = {"input_ids": input_ids, "labels": label_indices, "targets": targets}

                self.model.set_otr_label_indices(otr_label_indices)

                data_collator = DataCollatorWithPadding(
                    converter.tokenizer, pad_to_multiple_of=(8 if self.accelerator.mixed_precision == 'fp16' else None)
                )
                eval_dataset = Dataset.from_dict(eval_dict)
                eval_dataloader = DataLoader(
                    eval_dataset, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size
                )
                
                # Log a few random samples from the eval set:
                log_a_few_random_samples(logger, eval_dataset)
                
                # Use the device given by the `accelerator` object.
                if not self.parallelize:
                    self.model.to(self.accelerator.device)

                # Prepare everything with our `accelerator`.
                eval_dataloader = self.accelerator.prepare(eval_dataloader)

                with self.accelerator.main_process_first():
                    raw_data = self.df_raw.loc[0, :].to_frame().T
                    empty_inputs = self.empty_input.split(",")
                    eval_empty_input_iterators = []
                    for empty_input in empty_inputs:
                        input_ids, labels = preprocess_for_otr(raw_data, forced_input=empty_input)
                        label_indices = [
                            [otr_label2id[otr_label] for otr_label in instance_labels] 
                            for instance_labels in labels
                        ]
                        empty_dict = {"input_ids": input_ids, "labels": label_indices}
                        eval_empty_input_dataset = Dataset.from_dict(empty_dict)
                        eval_empty_input_dataloader = DataLoader(
                            eval_empty_input_dataset, collate_fn=data_collator, batch_size=self.per_device_eval_batch_size
                        )
                        eval_empty_input_dataloader = self.accelerator.prepare(eval_empty_input_dataloader)
                        eval_empty_input_iterator = iter(eval_empty_input_dataloader)
                        eval_empty_input_iterators.append(eval_empty_input_iterator)

                converter.set_padding_side_to_left()

                # Eval!
                total_batch_size = self.per_device_eval_batch_size * self.accelerator.num_processes

                logger.info("***** Running evaluation *****")
                logger.info(f"  Num examples = {len(eval_dataset)}")
                logger.info(f"  Instantaneous batch size per device = {self.per_device_eval_batch_size}")
                logger.info(f"  Total eval batch size (w. parallel, distributed) = {total_batch_size}")
                # Only show the progress bar once on each machine.
                progress_bar = tqdm(range(len(eval_dataloader)), disable=not self.accelerator.is_local_main_process)
                result_dict = defaultdict(list)
                self.model.eval()
                for batch in eval_dataloader:
                    with torch.no_grad():
                        while eval_empty_input_iterators:
                            eval_empty_input_iterator = eval_empty_input_iterators.pop()
                            empty_input = empty_inputs.pop()
                            empty_input_batch = next(eval_empty_input_iterator)  # proceed to next batch
                            if empty_input_batch["input_ids"].nelement() == 0:
                                raise ValueError("Empty prompt is given. Cannot apply PMI or MI.")
                            
                            _ = self.model(empty_input_batch)
                            result_dict = update_result_dict_for_empty(
                                result_dict, empty_input, self.model
                            )

                        predictions = self.model(batch)
                        result_dict = update_result_dict(
                            result_dict, batch["input_ids"], batch["targets"], predictions, self.model
                        )
                    
                    progress_bar.update(1)

                results = get_results(
                    result_dict,
                    self.tokenizer,
                    self.dataset_name,
                    self.dataset_config_name,
                    template_name,
                    None,
                    None,
                )

                if self.accelerator.is_main_process:
                    self.saver.save_results(results, template_name, None)
                    
            except:
                error_msg = traceback.format_exc()
                logger.error(f"\n{error_msg}")