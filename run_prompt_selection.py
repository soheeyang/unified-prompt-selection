import os
import logging
import glob
import torch
import numpy as np
from pytimedinput import timedInput
from extraction.inferrer import (
    InferrerForZeroshot,
    InferrerForFewshot,
    InferrerForOTR,
)
from extraction.evaluator import (
    EvaluatorForZeroshot,
    EvaluatorForFewshot,
    EvaluatorForOTR,
)
from extraction.saver import OutputSaver
from method.postprocessor import PostProcessor
from method.score import get_ps_result
from method.utils import (
    get_summary_result,
    save_summary_result,
)
from omegaconf import DictConfig
import hydra

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger.setLevel(logging.INFO)


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):
    extracted_output_dirpath_name = OutputSaver.get_extracted_output_dirpath_name(cfg)
    extracted_output_dir = OutputSaver.get_extracted_output_dir(cfg)
    
    if extracted_output_dir.exists():
        extract = False

        if cfg.fewshot:
            logger.info(
            f"""

            The output_dir already exists.
            
            Press 'y' if you want to skip P(y|x,t) extraction, 'n' otherwise.
            If you don't enter anything within 15 seconds, the extraction will proceed automatically.
            """
            )
            skip_extraction, time_out = timedInput("Skip: ", timeout=15, maxLength=1, allowCharacters="YyNn")
            if skip_extraction.lower() != 'y' or time_out:
                extract = True

        else:
            extracted_filenames = [
                extracted_f_name.split("/")[-1]
                for extracted_f_name in glob.glob(os.path.join(extracted_output_dir, "*.json"))
            ]
            extracted_t_names = set([f_name.split('__')[3] for f_name in extracted_filenames])
            t_names = set(cfg.prompt.template_names)
            unextracted_t_names = t_names - extracted_t_names

            if unextracted_t_names:
                unextracted_t_names = list(unextracted_t_names)
                
                logger.info(
                f"""

                The output_dir exists, but the P(y|x,t) extraction was not completed for the following templates.

                {unextracted_t_names}

                Press 'y' if you want further P(y|x,t) extraction to proceed only for that templates, otherwise press 'n'.                
                If you don't enter anything within 15 seconds, the extraction will proceed automatically.
                """
                )
                eval_flag, time_out = timedInput("Extraction: ", timeout=15, maxLength=1, allowCharacters="YyNn")
                if eval_flag.lower() == "y" or time_out:
                    cfg.prompt.template_names = unextracted_t_names
                    extract = True
    else:
        extract = True

    if extract:
        if cfg.first_token:
            extractor = (
                InferrerForOTR(cfg) 
                if not cfg.do_eval else 
                EvaluatorForOTR(cfg)
            )
        elif cfg.fewshot:
            extractor = (
                InferrerForFewshot(cfg) 
                if not cfg.do_eval else 
                EvaluatorForFewshot(cfg)
            )
        else:
            extractor = (
                InferrerForZeroshot(cfg) 
                if not cfg.do_eval else 
                EvaluatorForZeroshot(cfg)
            )
        extractor.extract()

    torch.set_default_dtype(torch.float64)

    ps_result = get_ps_result(
        method=cfg.method.method,
        post_processor=PostProcessor(cfg),
        one_hot=cfg.method.one_hot,
        cali_type=cfg.calibration.cali_type, 
        cali_norm_type=cfg.calibration.cali_norm_type, 
        filter=cfg.filter, 
        unbalance=cfg.unbalance,
        is_dynamic=cfg.dataset.DATASET_INFO.is_dynamic,
        select_for_each_x=cfg.method.select_for_each_x,
    )

    summary_result = get_summary_result(
        ps_result,
        method=cfg.method.method,
        cali_type=cfg.calibration.cali_type,
        cali_norm_type=cfg.calibration.cali_norm_type,
        select_for_each_x=cfg.method.select_for_each_x,
    )

    ps_result_dir = save_summary_result(
        summary_result,
        extracted_output_dirpath_name,
        method=cfg.method.method,
        first_token=cfg.first_token,
        one_hot=cfg.method.one_hot,
        select_for_each_x=cfg.method.select_for_each_x,
        cali_type=cfg.calibration.cali_type,
        cali_norm_type=cfg.calibration.cali_norm_type,
        filter=cfg.filter, 
        unbalance=cfg.unbalance,
    )

    eval_txt = f"""
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    Evaluation results using predictions from selected prompt.

    * X
       Accuracy:   {summary_result['X']['accuracy']:.4f}
       F1 score:   {summary_result['X']['macro_f1']:.4f}
       Prediction: {summary_result['X']['prediction'][:50]} ...
       Target:     {summary_result['X']['target'][:50]} ...
       Correct:    {(np.array(summary_result['X']['target'][:50], dtype=np.int32) == np.array(summary_result['X']['prediction'][:50], dtype=np.int32)).astype(int).tolist()} ...

    * A
       Accuracy:   {summary_result['A']['accuracy']:.4f}
       F1 score:   {summary_result['A']['macro_f1']:.4f}
       Prediction: {summary_result['A']['prediction'][:50]} ...
       Target:     {summary_result['A']['target'][:50]} ...
       Correct:    {(np.array(summary_result['A']['target'][:50], dtype=np.int32) == np.array(summary_result['A']['prediction'][:50], dtype=np.int32)).astype(int).tolist()} ...

    * P
       Accuracy:   {summary_result['P']['accuracy']:.4f}
       F1 score:   {summary_result['P']['macro_f1']:.4f}
       Prediction: {summary_result['P']['prediction'][:50]} ...
       Target:     {summary_result['P']['target'][:50]} ...
       Correct:    {(np.array(summary_result['P']['target'][:50], dtype=np.int32) == np.array(summary_result['P']['prediction'][:50], dtype=np.int32)).astype(int).tolist()} ...

    * PA
       Accuracy:   {summary_result['PA']['accuracy']:.4f}
       F1 score:   {summary_result['PA']['macro_f1']:.4f}
       Prediction: {summary_result['PA']['prediction'][:50]} ...
       Target:     {summary_result['PA']['target'][:50]} ...
       Correct:    {(np.array(summary_result['PA']['target'][:50], dtype=np.int32) == np.array(summary_result['PA']['prediction'][:50], dtype=np.int32)).astype(int).tolist()} ...

    Note that some datasets are missing label values, 
    so check the target results.

    The predictions of the selected prompt were saved in the following file.

    '{ps_result_dir}'
    
    --------------------------------------------------------------------------------------------------------------------------------------------------------------------------
    """
    logger.info(eval_txt)
    
if __name__ == "__main__":
    main()