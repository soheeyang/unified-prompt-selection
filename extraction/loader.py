from pathlib import Path
from extraction.saver import OutputSaver
from typing import List
from omegaconf import DictConfig


class OutputLoader:
    """
    For loading extracted p(y|x,t) outputs.
    
    It was created to be utilized when getting a prompt selection score.
    """

    def __init__(
        self,
        cfg: DictConfig,
    ):
        self.output_dir = OutputSaver.get_extracted_output_dir(cfg)
    
    def get_all_eval_result_filepath(self) -> List[str]:
        return self.output_dir.glob("*.json")
    
    def get_filename_from_filepath(self, filepath: Path) -> str:
        return filepath.name
    
    def get_task_name_from_filename(self, filename: str) -> str:
        dataset_name = filename.split("__")[0]
        dataset_config_name = filename.split("__")[1]
        task_name = (
            dataset_name 
            if dataset_config_name == "None" else
            f"{dataset_name}--{dataset_config_name}"
        )
        return task_name
    
    def get_model_name_from_filename(self, filename: str) -> str:
        model_name = filename.split("__")[2]
        return model_name
    
    def get_prompt_name_from_filename(self, filename: str) -> str:
        prompt_name = filename.split("__")[3]
        return prompt_name
    
    def get_token_from_filename(self, filename: str) -> str:
        token = filename.split("__")[4]
        return token