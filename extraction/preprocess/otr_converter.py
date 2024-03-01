import torch
from abc import ABC, abstractmethod
from extraction.preprocess.utils import PreprocessorUtils
from extraction.promptsource import Template
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, List, Tuple
from pandas import DataFrame, Series


class Converter:
    def __init__(
        self,
        task: str,
        label_col: str,
        choice_cols: str,
        num_classes: int,
        tokenizer: PreTrainedTokenizerBase,
        template: Template,
        dataset_name: str,
        dataset_config_name: Optional[str],
        column_names: Union[List[str], str],
        max_length: int,
    ):
        self.task = task
        self.label_col = label_col
        self.choice_cols = choice_cols
        self.num_classes = num_classes
        self.tokenizer = tokenizer
        self.template = template
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.column_names = column_names
        self.max_length = max_length

        self.utils = PreprocessorUtils(
            self.tokenizer,
            self.template,
            self.dataset_name,
            self.dataset_config_name,
            self.column_names,
            self.max_length,
        )

    def set_padding_side_to_right(self):
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'right'
        self.utils.tokenizer.padding_side = 'right'
        self.utils.tokenizer.truncation_side = 'right'

    def set_padding_side_to_left(self):
        self.tokenizer.padding_side = 'left'
        self.tokenizer.truncation_side = 'left'
        self.utils.tokenizer.padding_side = 'left'
        self.utils.tokenizer.truncation_side = 'left'
    
    def get_diff_point(self, encoded_answer_list: List[int]) -> int:
        temp_answer_list = encoded_answer_list.copy()
        answer_n = temp_answer_list.pop()
        diff_list = [answer_n != answer for answer in temp_answer_list]
        diff_point = torch.vstack(diff_list).T.sum(-1).argmax(-1).item()
        return diff_point


class ConverterForEvaluation(Converter):
    def __init__(
        self,
        task: str,
        label_col: str,
        choice_cols: str,
        num_classes: int,
        tokenizer: PreTrainedTokenizerBase,
        template: Template,
        dataset_name: str,
        dataset_config_name: Optional[str],
        column_names: Union[List[str], str],
        max_length: int,
    ):
        super(ConverterForEvaluation, self).__init__(
            task=task,
            label_col=label_col,
            choice_cols=choice_cols,
            num_classes=num_classes,
            tokenizer=tokenizer,
            template=template,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            column_names=column_names,
            max_length=max_length,
        )

    def convert_static_into_otr(
        self, df_raw: DataFrame, forced_input: Optional[str] = None
    ) -> Tuple[List[List[int]], List[List[int]], List[int], Union[Series, List[int]]]:
        input_ids, labels = [], []
        for idx in range(len(df_raw)):
            ex = df_raw.loc[idx, :]
            input, target, ex_answer_choices = self.utils.get_template_info(ex)
            input, target, ex_answer_choices = self.utils.decide_target_space_position(
                input, target, ex_answer_choices
            )

            encoded_answers = [
                self.tokenizer.encode(answer, add_special_tokens=False)[0]
                for answer in ex_answer_choices
            ]

            # Use the forced_input function to force inputs such as empty, mask, and n/a.
            if forced_input is not None:
                input = self.utils.get_forced_input(ex, forced_input)
                encoded_input = self.tokenizer.encode(input, add_special_tokens=True)
                input_ids.append(encoded_input)
                labels.append(encoded_answers)
                return input_ids, labels
            
            encoded_input = self.tokenizer.encode(input, add_special_tokens=True)

            input_ids.append(encoded_input)
            labels.append(encoded_answers)
            otr_labels = encoded_answers

        return input_ids, labels, otr_labels, df_raw[self.label_col]
    
    def convert_dynamic_into_otr(
        self, df_raw: DataFrame, forced_input: Optional[str] = None
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
        input_ids, labels, otr_labels, targets = [], [], [], []
        for idx in range(len(df_raw)):
            choices = df_raw.loc[idx, self.choice_cols]
            choices = [f' {choice}' for choice in choices]
            choice_ids = self.tokenizer(
                choices,
                add_special_tokens=False,
                return_tensors='pt',
                padding=True
            )['input_ids']
            encoded_answer_list = [choice_ids[i] for i in range(self.num_classes)]

            ex = df_raw.loc[idx, :]
            input = self.template.apply(ex)[0]
            target = int(ex[self.label_col])
            if self.task == "story_cloze/2016":
                target -= 1

            encoded_correct_answer = encoded_answer_list[target]

            diff_point = self.get_diff_point(encoded_answer_list)

            label = [encoded_answer.tolist()[diff_point] for encoded_answer in encoded_answer_list]
            # if # of choices > 2, to make label unique
            if self.num_classes > 2:
                correct_answer = label[target]
                wrong_answers = [answer for answer in label if answer != correct_answer]
                wrong_answer = wrong_answers[0]
                for idx, answer in enumerate(label):
                    if idx == target:
                        continue
                    elif answer == correct_answer:
                        label[idx] = wrong_answer

            # Use the forced_input function to force inputs such as empty, mask, and n/a.
            if forced_input is not None:
                input = self.utils.get_forced_input(ex, forced_input)
                encoded_input = self.tokenizer.encode(input, add_special_tokens=True)
                input_ids.append(encoded_input)
                labels.append(label)
                return input_ids, labels

            encoded_input = self.tokenizer.encode(input, add_special_tokens=True)
            input = encoded_input + encoded_correct_answer[:diff_point].tolist()

            input_ids.append(input)
            labels.append(label)
            otr_labels.extend(label)
            targets.append(target)

        return input_ids, labels, sorted(list(set(otr_labels))), targets


class ConverterForInference(Converter):
    def __init__(
        self,
        task: str,
        label_col: str,
        choice_cols: str,
        num_classes: int,
        tokenizer: PreTrainedTokenizerBase,
        template: Template,
        dataset_name: str,
        dataset_config_name: Optional[str],
        column_names: Union[List[str], str],
        max_length: int,
    ):
        super(ConverterForInference, self).__init__(
            task=task,
            label_col=label_col,
            choice_cols=choice_cols,
            num_classes=num_classes,
            tokenizer=tokenizer,
            template=template,
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            column_names=column_names,
            max_length=max_length,
        )

    def convert_static_into_otr(
        self, df_raw: DataFrame, forced_input: Optional[str] = None
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
        input_ids, labels, targets = [], [], []
        for idx in range(len(df_raw)):
            ex = df_raw.loc[idx, :]
            input, _, ex_answer_choices = self.utils.get_template_info(ex)
            target = ex_answer_choices[0]

            input, target, ex_answer_choices = self.utils.decide_target_space_position(
                input, target, ex_answer_choices
            )

            encoded_answers = [
                self.tokenizer.encode(answer, add_special_tokens=False)[0]
                for answer in ex_answer_choices
            ]

            # Use the forced_input function to force inputs such as empty, mask, and n/a.
            if forced_input is not None:
                input = self.utils.get_forced_input(ex, forced_input)
                encoded_input = self.tokenizer.encode(input, add_special_tokens=True)
                input_ids.append(encoded_input)
                labels.append(encoded_answers)
                return input_ids, labels
            
            encoded_input = self.tokenizer.encode(input, add_special_tokens=True)

            input_ids.append(encoded_input)
            labels.append(encoded_answers)
            otr_labels = encoded_answers
            targets.append(-1)

        return input_ids, labels, otr_labels, targets
    
    def convert_dynamic_into_otr(
        self, df_raw: DataFrame, forced_input: Optional[str] = None
    ) -> Tuple[List[List[int]], List[List[int]], List[int], List[int]]:
        input_ids, labels, otr_labels, targets = [], [], [], []
        for idx in range(len(df_raw)):
            choices = df_raw.loc[idx, self.choice_cols]
            choices = [f' {choice}' for choice in choices]
            choice_ids = self.tokenizer(
                choices,
                add_special_tokens=False,
                return_tensors='pt',
                padding=True
            )['input_ids']
            encoded_answer_list = [choice_ids[i] for i in range(self.num_classes)]

            ex = df_raw.loc[idx, :]
            input = self.template.apply(ex)[0]
            
            diff_point = self.get_diff_point(encoded_answer_list)

            label = [encoded_answer.tolist()[diff_point] for encoded_answer in encoded_answer_list]
            if self.num_classes > 2:
                assert self.label_col is not None and ex[self.label_col], (
                    "If the number of classes is more than one and the label value does not exist, "
                    "inference is not possible with the current code."
                )
                target = int(ex[self.label_col])
                correct_answer = label[target]
                wrong_answers = [answer for answer in label if answer != correct_answer]
                wrong_answer = wrong_answers[0]
                for idx, answer in enumerate(label):
                    if idx == target:
                        continue
                    elif answer == correct_answer:
                        label[idx] = wrong_answer
                encoded_correct_answer = encoded_answer_list[target]
            else:
                encoded_correct_answer = encoded_answer_list[0]
            
            # Use the forced_input function to force inputs such as empty, mask, and n/a.
            if forced_input is not None:
                input = self.utils.get_forced_input(ex, forced_input)
                encoded_input = self.tokenizer.encode(input, add_special_tokens=True)
                input_ids.append(encoded_input)
                labels.append(label)
                return input_ids, labels

            encoded_input = self.tokenizer.encode(input, add_special_tokens=True)
            input = encoded_input + encoded_correct_answer[:diff_point].tolist()

            input_ids.append(input)
            labels.append(label)
            otr_labels.extend(label)
            targets.append(-1)

        return input_ids, labels, sorted(list(set(otr_labels))), targets