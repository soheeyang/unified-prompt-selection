import logging
from logging import Logger
from accelerate import Accelerator
import os
import random
from collections import defaultdict
import datasets
from datasets import Dataset
import torch
from torch import Tensor
import numpy as np
from pandas import DataFrame
from datasets import load_dataset, load_metric
from torch.utils.data import DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)
from extraction.promptsource import DatasetTemplates, Template
from extraction.model.decoder import (
    ModelBase, 
    DecoderForPromptSelection, 
    DecoderForPromptSelectionOTR,
)
from typing import Optional, Tuple, Callable, Union, List, Iterable, Dict
from omegaconf import DictConfig


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    # torch.use_deterministic_algorithms(True) > RuntimeError
    os.environ["CUBLAS_WORKSPACE_CONFIG"]=":4096:2"

def set_logger(logger: Logger, accelerator: Accelerator) -> None:
    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state)

    # Setup logging, we only want one process per machine to log things on the screen.
    # accelerator.is_local_main_process is only True for one process per machine.
    logger.setLevel(logging.INFO if accelerator.is_local_main_process else logging.ERROR)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

def get_task_name(dataset_name: str, dataset_config_name: Optional[str]) -> str:
    task = (
        f"{dataset_name}" 
        if dataset_config_name is None else 
        f"{dataset_name}/{dataset_config_name}"
    )
    return task

def load_train_datasets(dataset_kwargs) -> Dataset:
    dataset_kwargs = dataset_kwargs.copy()
    task = get_task_name(dataset_kwargs.path, dataset_kwargs.name)
    split = 'validation' if task == "story_cloze/2016" else 'train'
    dataset_kwargs.update(split=split)
    return load_dataset(**dataset_kwargs)

def get_dataset_info_for_otr(dataset_info: dict) -> Tuple[int, str, str, bool]:
    num_classes = dataset_info['num_classes']
    choice_cols = dataset_info['choices']
    label_col = dataset_info['label']
    is_dynamic = dataset_info['is_dynamic']
    return num_classes, choice_cols, label_col, is_dynamic

def get_raw_df(raw_datasets: Dataset) -> DataFrame:
    return raw_datasets.to_pandas()

def get_raw_df_for_otr(raw_datasets: Dataset, task: str) -> DataFrame:
    df_raw = get_raw_df(raw_datasets)
    if task == "newspop":
        topic2label = {"economy": 0, "microsoft": 1, "obama": 2, "palestine": 3}
        df_raw['label'] = df_raw['topic'].map(topic2label)
    return df_raw

def shuffle_dataset(raw_datasets: Dataset, seed: int) -> Dataset:
    return raw_datasets.shuffle(seed=seed)

def get_sub_samples(raw_datasets: Dataset, num_samples: int) -> Dataset:
    return raw_datasets.select(range(min(len(raw_datasets), num_samples)))

def load_shuffled_subset_raw_datasets(cfg: DictConfig) -> Dataset:
    raw_datasets = load_dataset(**cfg.dataset.DATASET_KWARGS)
    # Trim a number of evaluation examples
    if cfg.num_samples is not None:
        raw_datasets = shuffle_dataset(raw_datasets, cfg.seed)
        raw_datasets = get_sub_samples(raw_datasets, cfg.num_samples)
    return raw_datasets

def get_config(config_name: str, model_name_or_path: str):
    if config_name:
        config = AutoConfig.from_pretrained(config_name)
    elif model_name_or_path:
        config = AutoConfig.from_pretrained(model_name_or_path)
    else:
        raise ValueError(
            "Either `args.config_name` or `args.model_name_or_path` should be provided."
        )
    return config

def check_tokenizer_pad_token(
    tokenizer: PreTrainedTokenizerBase
) -> PreTrainedTokenizerBase:
    if tokenizer.pad_token is None:
        for token in [tokenizer.eos_token, tokenizer.bos_token, tokenizer.sep_token]:
            if token is not None:
                tokenizer.pad_token = token
        if tokenizer.pad_token is None:
            raise ValueError("Please define a pad token id.")
    return tokenizer

def get_tokenizer(
    tokenizer_name: str, model_name_or_path: str, use_slow_tokenizer: bool
) -> PreTrainedTokenizerBase:
    if tokenizer_name:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=not use_slow_tokenizer, padding_side="left")
    elif model_name_or_path:
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=not use_slow_tokenizer, padding_side="left")
    else:
        raise ValueError(
            "You are instantiating a new tokenizer from scratch. This is not supported by this script."
            "You can do it from another script, save it, and load it from here, using --tokenizer_name."
        )
    return check_tokenizer_pad_token(tokenizer)

def get_model(
    config, 
    model_name_or_path: str, 
    parallelize: bool, 
    sum_log_prob: bool, 
    ignore_index: int, 
    first_token: bool,
) -> Union[DecoderForPromptSelection, DecoderForPromptSelectionOTR]:
    model = ModelBase.from_config(
        config=config,
        first_token=first_token,
        model_name_or_path=model_name_or_path,
        parallelize=parallelize,
        sum_log_prob=sum_log_prob,
        ignore_index=ignore_index,
    )
    return model

def load_pretrained_model_and_tokenizer(
    cfg: DictConfig
) -> Tuple[
    Union[DecoderForPromptSelection, DecoderForPromptSelectionOTR], 
    PreTrainedTokenizerBase
]:
    config = get_config(cfg.decoder.config_name, cfg.decoder.model_name_or_path)
    tokenizer = get_tokenizer(cfg.decoder.tokenizer_name, cfg.decoder.model_name_or_path, cfg.decoder.use_slow_tokenizer)
    model = get_model(
        config=config,
        model_name_or_path=cfg.decoder.model_name_or_path,
        parallelize=cfg.decoder.parallelize,
        sum_log_prob=cfg.sum_log_prob,
        ignore_index=cfg.decoder.ignore_index,
        first_token=cfg.first_token
    )
    return model, tokenizer

def get_prompts(dataset_name: str, dataset_config_name: str) -> DatasetTemplates:
    task = get_task_name(dataset_name, dataset_config_name)
    return DatasetTemplates(task)

def get_template(
    template_name: str, dataset_name: str, dataset_config_name: Optional[str]
) -> Template:
    prompts = get_prompts(dataset_name, dataset_config_name)
    return prompts[template_name]

def log_a_few_random_samples(logger: Logger, eval_dataset: Dataset) -> None:
    for index in random.sample(range(len(eval_dataset)), 3):
        logger.info(f"Sample {index} of the training set: {eval_dataset[index]}.")

def get_eval_empty_input_datasets(
    preprocess_fn: Callable, 
    column_names: Union[List[str], str], 
    raw_data: Dataset, 
    empty_inputs: List[str]
) -> List[Dataset]:
    eval_empty_input_datasets = [
        raw_data.map(
            lambda examples: preprocess_fn(examples, forced_input=empty_input),
            batched=True, remove_columns=column_names
        )
        for empty_input in empty_inputs
    ]
    return eval_empty_input_datasets

def get_eval_empty_input_dataloaders(
    data_collator: Callable, 
    batch_size: int, 
    eval_empty_input_datasets: List[Dataset],
) -> List[DataLoader]:
    eval_empty_input_dataloaders = [
        DataLoader(
            eval_empty_input_dataset, collate_fn=data_collator, batch_size=batch_size
        )
        for eval_empty_input_dataset in eval_empty_input_datasets
    ]
    return eval_empty_input_dataloaders

def get_eval_empty_input_iterators(
    accelerator: Accelerator, eval_empty_input_dataloaders: List[DataLoader]
) -> List[Iterable]:
    eval_empty_input_iterators = list()
    for eval_empty_input_dataloader in eval_empty_input_dataloaders:
        eval_empty_input_dataloader = accelerator.prepare(eval_empty_input_dataloader)
        eval_empty_input_iterator = iter(eval_empty_input_dataloader)
        eval_empty_input_iterators.append(eval_empty_input_iterator)
    return eval_empty_input_iterators

def get_metrics():
    metric_acc = load_metric("accuracy")
    metric_f1 = load_metric("f1")
    return metric_acc, metric_f1

def add_pred_and_target_to_metrics(
    metric_acc, metric_f1, predictions: Tensor, targets: Tensor, accelerator: Accelerator
):
    metric_acc.add_batch(
        predictions=accelerator.gather(predictions),
        references=accelerator.gather(targets),
    )
    metric_f1.add_batch(
        predictions=accelerator.gather(predictions),
        references=accelerator.gather(targets),
    )
    return metric_acc, metric_f1

def get_inference_result(model) -> Tuple[List[float], List[float]]:
    log_prob = model.log_prob.cpu().tolist()
    seq_Ps = model.seq_Ps
    return log_prob, seq_Ps

def get_inference_info(input_ids, targets: Tensor, predictions: Tensor) -> Tuple[List[int], List[int], List[int]]:
    input_ids = input_ids.cpu().tolist()
    targets = targets.cpu().tolist()
    predictions = predictions.cpu().tolist()
    return input_ids, targets, predictions

def get_instance_acc(predictions: Tensor, targets: Tensor) -> List[int]:
    return (predictions == targets).int().cpu().tolist()

def update_result_dict_for_empty(result_dict: defaultdict, empty_input: str, model) -> defaultdict:
    log_prob, seq_Ps = get_inference_result(model)
    result_dict[f"{empty_input}_log"].extend(log_prob)
    if empty_input == '':
        result_dict['P(t)'].extend(seq_Ps)
    return result_dict

def update_result_dict(
    result_dict: defaultdict, input_ids: Tensor, targets: Tensor, predictions: Tensor, model
) -> defaultdict:
    instance_accs = get_instance_acc(predictions, targets)
    input_ids, targets, predictions = get_inference_info(
        input_ids, targets, predictions
    )
    log_prob, seq_Ps = get_inference_result(model)

    result_dict['all_log_prob'].extend(log_prob)
    result_dict['P(x,t)'].extend(seq_Ps)
    result_dict['all_inputs'].extend(input_ids)
    result_dict['all_predictions'].extend(predictions)
    result_dict['all_targets'].extend(targets)
    result_dict['all_accuracy'].extend(instance_accs)
    return result_dict

def get_eval_acc_and_f1(metric_acc, metric_f1) -> Tuple[float, float]:
    eval_acc = metric_acc.compute()
    eval_f1 = metric_f1.compute(average="macro")
    return eval_acc, eval_f1

def get_results(
    result_dict: defaultdict,
    tokenizer,
    dataset_name: str,
    dataset_config_name: Optional[str],
    template_name: str,
    eval_acc: float,
    eval_f1: float,
) -> dict:
    results = {
        "dataset_name": dataset_name,
        "dataset_config_name": dataset_config_name,
        "template_name": template_name,
        "evaluation": eval_acc,
        "raw": {
            "inputs": tokenizer.batch_decode(
                result_dict['all_inputs'], skip_special_tokens=True
            ),
            "predictions": result_dict['all_predictions'],
            "targets": result_dict['all_targets'],
            "accuracy": result_dict['all_accuracy'],
            "f1": eval_f1['f1'] if eval_f1 is not None else eval_f1,
            'log_prob': result_dict['all_log_prob'],
            'empty_log': result_dict['_log'],
            'na_log': result_dict['N/A_log'],
            'mask_log': result_dict['[MASK]_log'],
            'P(x,t)': result_dict['P(x,t)'],
            'P(t)': result_dict['P(t)'],
        }
    }
    return results