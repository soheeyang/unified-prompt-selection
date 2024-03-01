import json
from pathlib import Path
from typing import Optional, Dict
from omegaconf import DictConfig


class OutputSaver:
    """
    For storing extraction results.
    It is intended to be utilized as a component of an evaluator and inferrer.

    Usage example:
        output_saver = OutputSaver(
            dataset_name, dataset_config_name, model_name_or_path, first_token, num_samples
        )
        output_saver.create_output_dir(cfg)

        results = get_results(...)
        template_name = args.template_names.split(',')[0]
        eval_acc, _ = get_eval_acc_and_f1(...)

        output_saver.save_results(results, template_name, eval_acc)

    """
    
    result_dir = Path.cwd().joinpath("extraction", "results")

    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str],
        model_name_or_path: str,
        first_token: bool,
        num_samples: int,
    ) -> None:
        self.output_dir = None
        self.dataset_name = dataset_name.replace('/', '--')
        self.dataset_config_name = dataset_config_name
        self.model_name_or_path = model_name_or_path.replace('/', '--')
        self.token = "first_token" if first_token else "all_tokens"
        self.num_samples = num_samples
        
    @classmethod
    def get_extracted_output_dirpath_name(cls, cfg: DictConfig) -> str:
        dataset = f"{cfg.dataset.dataset_name}_{cfg.dataset.dataset_config_name}_{cfg.dataset.split}"
        decoder = f"{cfg.decoder.model_name_or_path}"
        prompt = f"{cfg.prompt.prompt_config_name}"
        extraction = f"first_token={cfg.first_token}__sum_log_prob={cfg.sum_log_prob}__num_samples={cfg.num_samples}__seed={cfg.seed}__fewshot={cfg.fewshot}__do_eval={cfg.do_eval}"
        return f"dataset={dataset}__decoder={decoder}__prompt={prompt}__{extraction}".replace('/', '--')
    
    @classmethod
    def get_extracted_output_dir(cls, cfg: DictConfig) -> Path:
        output_dir = cls.get_extracted_output_dirpath_name(cfg)
        return cls.result_dir.joinpath(output_dir)
    
    def get_output_file_name(self, template_name: str, eval_acc: Optional[Dict[str, float]]) -> str:
        if eval_acc is not None:
            return f"{self.dataset_name}__{self.dataset_config_name}__{self.model_name_or_path}__{template_name}__{self.token}__{eval_acc['accuracy']:.4f}.json"
        else:
            return f"{self.dataset_name}__{self.dataset_config_name}__{self.model_name_or_path}__{template_name}__{self.token}__{eval_acc}.json"
    
    def get_output_file_name_for_fewshot(self, fewshot_info: str, eval_acc: Optional[Dict[str, float]]) -> str:
        if eval_acc is not None:
            return f"{self.dataset_name}__{self.dataset_config_name}__{self.model_name_or_path}__{fewshot_info}__{self.token}__{eval_acc['accuracy']:.4f}.json"
        else:
            return f"{self.dataset_name}__{self.dataset_config_name}__{self.model_name_or_path}__{fewshot_info}__{self.token}__{eval_acc}.json"
    
    def create_output_dir(self, cfg: DictConfig) -> None:
        self.output_dir = self.get_extracted_output_dir(cfg)
        self.output_dir.mkdir(exist_ok=True)
        
    def save_results(
        self, results: dict, template_name: str, eval_acc: Optional[Dict[str, float]], fewshot_info: Optional[int] = None
    ) -> None:
        assert self.output_dir is not None, (
            "In order to utilize save_results, we need to create an output_dir utilizing the create_output_dir method."
        )
        if not fewshot_info:
            output_file_name = self.get_output_file_name(template_name, eval_acc)
        else:
            output_file_name = self.get_output_file_name_for_fewshot(fewshot_info, eval_acc)
        
        with open(self.output_dir.joinpath(output_file_name), "w") as f:
            json.dump(results, f, indent=4)