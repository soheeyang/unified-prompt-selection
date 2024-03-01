from abc import ABC, abstractmethod
from extraction.preprocess.utils import PreprocessorUtils
from extraction.promptsource import Template
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, List, Dict
from torch import Tensor


class Preprocessor(ABC):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        template: Template,
        dataset_name: str,
        dataset_config_name: Optional[str],
        column_names: Union[List[str], str],
        max_length: int,
    ):
        self.tokenizer = tokenizer
        self.template = template
        self.dataset_name = dataset_name
        self.dataset_config_name = dataset_config_name
        self.column_names = column_names
        self.max_length = max_length
        self.is_opt = self.tokenizer.name_or_path.startswith("facebook/opt")

        self.utils = PreprocessorUtils(
            self.tokenizer,
            self.template,
            self.dataset_name,
            self.dataset_config_name,
            self.column_names,
            self.max_length,
        )

    def _get_newspop_target(self, option: str, ex_answer_choices: List[str]) -> str:
        options = ["Economy", "Microsoft", "Obama", "Palestine"]
        option2target = {
            option: answer_choice 
            for option, answer_choice in zip(options, ex_answer_choices)
        }
        return option2target[option]

    def _get_tokenized_inputs(self, input_texts: List[str]):
        tokenized_inputs = self.tokenizer(
            input_texts,
            padding=False,
            max_length=self.max_length,
            truncation=True,
            add_special_tokens=self.is_opt
        )
        return tokenized_inputs

    def _get_tokenized_targets(self, answer_choices_texts: List[str]):
        tokenized_targets = [
            self.tokenizer(
                ans_choi,
                # padding is on the right here.
                padding=False,
                max_length=self.max_length,
                truncation=True,
                add_special_tokens=not self.is_opt,
            )
            for ans_choi in answer_choices_texts
        ]
        return tokenized_targets
    
    @abstractmethod
    def _get_features(self, bs):
        pass
    
    @abstractmethod
    def __call__(self, examples, fewshot, forced_input):
        pass


class PreprocessorForEvaluation(Preprocessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        template: Template,
        dataset_name: str,
        dataset_config_name: Optional[str],
        column_names: Union[List[str], str],
        max_length: int,
    ):
        super(PreprocessorForEvaluation, self).__init__(
            tokenizer,
            template,
            dataset_name,
            dataset_config_name,
            column_names,
            max_length,
        )
    
    def __call__(
        self,
        examples,
        fewshot: str = '',
        forced_input: Optional[str] = None
    ) -> Dict[str, Union[List[Tensor], List[int]]]:
        bs = len(examples[self.column_names[0]])

        self.input_texts = []
        self.target_texts = []
        self.answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in self.column_names
            }
            
            input, target, ex_answer_choices = self.utils.get_template_info(ex)

            if self.dataset_name == "newspop":
                target = self._get_newspop_target(target, ex_answer_choices)
            
            # Use the forced_input function to force inputs such as empty, mask, and n/a.
            if forced_input is not None:
                input = self.utils.get_forced_input(ex, forced_input)
            
            input, target, ex_answer_choices = self.utils.decide_target_space_position(
                input, target, ex_answer_choices
            )
            assert target in ex_answer_choices
            self.input_texts.append(fewshot + input)
            self.target_texts.append(target)
            self.answer_choices_texts.append(ex_answer_choices)

        self.tokenized_inputs = self._get_tokenized_inputs(self.input_texts)
        self.tokenized_targets = self._get_tokenized_targets(self.answer_choices_texts)
        features = self._get_features(bs)

        return features
    
    def _get_features(self, bs: int) -> Dict[str, Union[List[Tensor], List[int]]]:
        features = {
            k: [
                [elem for _ in range(len(self.tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in self.tokenized_inputs.items()
        }
        features["labels"] = [
            self.tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]
        features["labels_attention_mask"] = [
            self.tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [
            self.answer_choices_texts[idx].index(t)
            for idx, t in enumerate(self.target_texts)
        ]
        return features

    
class PreprocessorForInference(Preprocessor):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        template: Template,
        dataset_name: str,
        dataset_config_name: Optional[str],
        column_names: Union[List[str], str],
        max_length: int,
    ):
        super(PreprocessorForInference, self).__init__(
            tokenizer,
            template,
            dataset_name,
            dataset_config_name,
            column_names,
            max_length,
        )
    
    def __call__(
        self,
        examples,
        fewshot: str = '',
        forced_input: Optional[str] = None
    ) -> Dict[str, Union[List[Tensor], List[int]]]:
        bs = len(examples[self.column_names[0]])

        self.input_texts = []
        self.target_texts = []
        self.answer_choices_texts = []
        for i in range(bs):
            ex = {
                k: examples[k][i]
                for k in self.column_names
            }
            
            input, _, ex_answer_choices = self.utils.get_template_info(ex)
            target = ex_answer_choices[0]
            
            # Use the forced_input function to force inputs such as empty, mask, and n/a.
            if forced_input is not None:
                input = self.utils.get_forced_input(ex, forced_input)
            
            input, target, ex_answer_choices = self.utils.decide_target_space_position(
                input, target, ex_answer_choices
            )
            
            self.input_texts.append(fewshot + input)
            self.answer_choices_texts.append(ex_answer_choices)

        self.tokenized_inputs = self._get_tokenized_inputs(self.input_texts)
        self.tokenized_targets = self._get_tokenized_targets(self.answer_choices_texts)
        features = self._get_features(bs)

        return features
    
    def _get_features(self, bs: int) -> Dict[str, Union[List[Tensor], List[int]]]:
        features = {
            k: [
                [elem for _ in range(len(self.tokenized_targets[idx]["input_ids"]))]
                for idx, elem in enumerate(v)
            ]
            for k, v in self.tokenized_inputs.items()
        }
        features["labels"] = [
            self.tokenized_targets[idx]["input_ids"]
            for idx in range(bs)
        ]
        features["labels_attention_mask"] = [
            self.tokenized_targets[idx]["attention_mask"]
            for idx in range(bs)
        ]
        features["targets"] = [-1 for idx in range(bs)]
        return features