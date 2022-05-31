import json
import os
import re
import shutil
import logging
from subprocess import run
from typing import Any, Dict, List, Optional, Pattern


def maybe_download_from_aws(file, storage_dir: str = "~/.config/laughter_detector/", aws_dir = "s3://charactr-models/laughter-detector/"):
    storage_dir = os.path.expanduser(storage_dir)
    local_path = os.path.join(storage_dir, file)
    aws_path = aws_dir + file
    if not os.path.isfile(local_path):
        logging.info(f"Try to download: {aws_path}")
        os.makedirs(storage_dir, exist_ok=True)
        run(["aws", "s3", "cp", aws_path, storage_dir])
    elif os.path.isfile(local_path):
        logging.info(f"File {local_path} found")
    else:
        raise FileNotFoundError
    return local_path