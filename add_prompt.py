from copy import deepcopy
from extraction.promptsource.templates import DatasetTemplates, Template
from extraction.utils import get_task_name
from typing import Tuple, Dict, List, Union
from omegaconf import DictConfig
import hydra


@hydra.main(version_base=None, config_path="conf", config_name="config_custom_prompt")
def main(cfg: DictConfig):
    task_name = get_task_name(cfg.dataset.dataset_name, cfg.dataset.dataset_config_name)
    text_formats, jinja_suffix = get_template_info(**cfg.dataset.TEMPLATE_INFO)
    
    templates = DatasetTemplates(task_name)
    base_template_dict, p_names = get_base_template_dict(templates)
    new_template_dict = generate_template_dict(
        base_template_dict, jinja_suffix, text_formats, p_names
    )
    
    new_template = Template(
        name=new_template_dict['name'],
        jinja=new_template_dict['jinja'],
        reference=new_template_dict['reference'],
        metadata=Template.Metadata(**new_template_dict['metadata']),
        answer_choices=new_template_dict['answer_choices']
    )

    templates.add_template(new_template)

def get_template_info(text_formats, jinja_suffix):
    return text_formats, jinja_suffix

def get_base_template_dict(templates: DatasetTemplates) -> Tuple[Dict[str, Union[str, Template.Metadata]], List[str]]:
    template_dicts, base_template_dict, p_names = [], None, []
    for template in list(templates.templates.values()):
        template_dict = template.__dict__
        p_name = template_dict['name']
        if not isinstance(template_dict['metadata'], dict):
            template_dict['metadata'] = template_dict['metadata'].__dict__
        if template_dict['name'] == "prompt_00":
            base_template_dict = template_dict
        template_dicts.append(template_dict)
        p_names.append(p_name)

    if base_template_dict is None:
        base_template_dict = template_dicts[0]
    
    return base_template_dict, p_names

def get_instruction(ex_jinja: str, text_formats: List[str]) -> str:
    while True:
        jinja = input(
            f"\n\nInstruction example:\n{ex_jinja}\n\nEnter the instruction for the new prompt, using the example format provided above:\n"
        )

        missing_text = False
        for text_format in text_formats:
            if text_format not in jinja:
                print(f"\nThe instruction is missing '{text_format}'. Please enter the instruction again.\n")
                missing_text = True
        if missing_text:
            continue

        check_jinja = input(
            f"\nPlease verify that the entered instruction is correct:\n{jinja}\n\nIf the instruction is correct, press 'y'; otherwise, enter another key.\nIf you choose another key, you will be able to edit the instruction: "
        ).strip().lower()

        if check_jinja == "y":
            break

    return jinja

def get_answer_choices(ex_answer_choices: str) -> str:
    num_choices = len(ex_answer_choices.split(" ||| "))

    while True:
        answer_choices = input(
            f"\n\nAnswer_choices example:\n{ex_answer_choices}\n\nEnter the answer_choices for the new prompt, using the example format provided above:\n"
        )

        if " ||| " not in answer_choices:
            print("\nEach answer_choice must be entered separated by \" ||| \". Please enter the answer_choices again.\n")
            continue
        
        new_num_choices = len(answer_choices.split(" ||| "))
        if num_choices != new_num_choices:
            print(f"\nThe number of answer_choices must be {num_choices}. Please enter the answer_choices again.\n")
            continue

        check_answer_choices = input(
            f"\nPlease verify that the entered answer_choices are correct:\n{answer_choices}\n\nIf the answer_choices are correct, press 'y'; otherwise, enter another key.\nIf you choose another key, you will be able to edit the answer_choices: "
        ).strip().lower()

        if check_answer_choices == "y":
            break

    return answer_choices

def get_prompt_name(p_names: List[str]) -> str:
    while True:
        prompt_name = input("\n\nEnter the prompt name: ")

        if prompt_name in p_names:
            print(f"\nA prompt with the same name already exists.\n")
            continue

        check_prompt_name = input(
            f"\nPlease verify that the entered prompt name is correct:\n{prompt_name}\n\nIf the prompt name is correct, press 'y'; otherwise, enter another key.\nIf you choose another key, you will be able to edit the prompt name: "
        ).strip().lower()

        if check_prompt_name == "y":
            break

    return prompt_name

def generate_template_dict(
    base_template_dict: Dict[str, Union[str, Template.Metadata]], 
    jinja_suffix: str, 
    text_formats: List[str], 
    p_names: List[str]
) -> Dict[str, Union[str, Dict[str, str]]]:
    ex_jinja = base_template_dict['jinja'].split(jinja_suffix)[0]
    ex_answer_choices = base_template_dict['answer_choices']

    jinja = get_instruction(ex_jinja, text_formats)
    jinja += jinja_suffix

    answer_choices = get_answer_choices(ex_answer_choices)
    choices_in_prompt = False
    for ans_choice in answer_choices.split(" ||| "):
        if ans_choice in jinja:
            choices_in_prompt = True

    prompt_name = get_prompt_name(p_names)

    template_dict = deepcopy(base_template_dict)
    template_dict['jinja'] = jinja
    template_dict['answer_choices'] = answer_choices
    template_dict['metadata']['choices_in_prompt'] = choices_in_prompt
    template_dict['reference'] = 'prompt_maker'
    template_dict['name'] = prompt_name
    return template_dict


if __name__ == "__main__":
    main()