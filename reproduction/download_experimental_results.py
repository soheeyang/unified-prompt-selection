import os
import logging
import argparse
import traceback
from pathlib import Path
from huggingface_hub import hf_hub_download


logger = logging.getLogger(__name__)
logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
logger.setLevel(logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Download experimental results in UPS.")
    parser.add_argument(
        "--result",
        type=str,
        default=None,
        help="Pass in the experiment results to download. You can choose 'inference' or 'prompt_selection'. If not specified, both results will be downloaded."
    )
    return parser.parse_args()

def get_cwd() -> Path:
    CWD = Path.cwd()
    return CWD if 'reproduction' == CWD.name else CWD.joinpath('reproduction')

def get_hf_hub_cache_dir() -> Path:
    hf_home = os.getenv("HF_HOME")
    if hf_home is None:
        hf_home = Path.home().joinpath(".cache", "huggingface")
    hf_hub_cache_dir = Path.joinpath(hf_home, "hub")
    return hf_hub_cache_dir

def get_hf_hub_cache_download_dir(hf_hub_cache_dir: Path) -> Path:
    return hf_hub_cache_dir.joinpath("datasets--gimmaru--ups_reproduction")

def download_result(FILE_NAME: Path, CWD: Path, OUTPUT_FILE: Path) -> None:
    logger.info("Start downloading.")
    download_complete = False
    hf_hub_cache_dir = get_hf_hub_cache_dir()
    hf_hub_cache_dl_dir = get_hf_hub_cache_download_dir(hf_hub_cache_dir)

    try:
        hf_hub_download(
            repo_id='gimmaru/ups_reproduction',
            filename=FILE_NAME,
            repo_type='dataset',
            local_dir=CWD,
            local_dir_use_symlinks=True,
        )
        logger.info("Finish downloading.")
        download_complete = True

    except:
        error_msg = traceback.format_exc()
        logger.error(f"\n{error_msg}\n")
        
        os.system(f"rm {hf_hub_cache_dir.joinpath('tmp*')}")
        logger.info(f"""

        Clean up the temporary cache files inside the huggingface hub cache directory.

        huggingface hub cache directory:
        '{hf_hub_cache_dir}'
        """)

    if download_complete:
        logger.info("Start decompressing.")
        os.system(f"tar -zxvf {OUTPUT_FILE} -C {CWD}")
        logger.info("Finish decompressing.")

        if OUTPUT_FILE.exists():
            logger.info("Start removing compressed file.")
            os.system(f"rm {OUTPUT_FILE}")

            os.system(f"rm -r {hf_hub_cache_dl_dir}")
            logger.info(f"""

            Clean up the cache directory.

            '{hf_hub_cache_dl_dir}\n'
            """)

            logger.info("Finish removing compressed file.")


if __name__ == "__main__":
    args = parse_args()

    if args.result == 'inference':
        file_names = ["ex_eval_results.tar.gz"]
    elif args.result == 'prompt_selection':
        file_names = ["ps_results.tar.gz"]
    else:
        file_names = ["ex_eval_results.tar.gz", "ps_results.tar.gz"]
    
    CWD = get_cwd()
    for FILE_NAME in file_names:
        OUTPUT_FILE = CWD.joinpath(FILE_NAME)
        download_result(FILE_NAME, CWD, OUTPUT_FILE)