import os
import logging
import glob
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


if __name__ == "__main__":
    main()