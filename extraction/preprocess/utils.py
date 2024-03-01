from extraction.promptsource import Template
from transformers import PreTrainedTokenizerBase
from typing import Optional, Union, List, Tuple


class PreprocessorUtils:
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
    
    def get_prompt_input_and_target(self, ex):
        prompt = self.template.apply(ex)
        if len(prompt) == 2: # If target is set
            input, target = prompt
        else:
            input, target = prompt[0], None
        return input, target

    def get_template_info(self, ex) -> Tuple[str, str, List[str]]:
        input, target = self.get_prompt_input_and_target(ex)
        ex_answer_choices = self.template.get_answer_choices_list(ex)
        return input, target, ex_answer_choices
    
    def get_forced_input(self, ex, forced_input: Optional[str]) -> str:
        if self.dataset_name == "newspop":
            ex = {k: forced_input if k != 'topic' else v for k, v in ex.items()}
        elif self.dataset_config_name == "copa":
            ex = {k: forced_input if k not in ['question', 'label'] else v for k, v in ex.items()}
        else:
            ex = {k: forced_input if k != 'label' else v for k, v in ex.items()}
        input, _ = self.template.apply(ex)
        return input
    
    def decide_target_space_position(
        self, input: str, target: str, ex_answer_choices: List[str]
    ) -> Tuple[str, str, List[str]]:
        target = " " + target
        encoded_target = self.tokenizer.encode(target, add_special_tokens=False)
        first_target_token = self.tokenizer.decode(encoded_target[0])
        if first_target_token.isspace():
            input += " "
            target = target.lstrip()
        else:
            ex_answer_choices = [" " + choice for choice in ex_answer_choices]
        return input, target, ex_answer_choices