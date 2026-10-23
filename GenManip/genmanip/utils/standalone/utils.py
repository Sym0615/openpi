"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import hashlib
import logging
import os
import random
import sys
from typing import Any


class ColorFormatter(logging.Formatter):
    COLORS = {
        logging.DEBUG: "\033[36m",  # Cyan
        logging.INFO: "\033[32m",  # Green
        logging.WARNING: "\033[33m",  # Yellow
        logging.ERROR: "\033[31m",  # Red
        logging.CRITICAL: "\033[1;31m",  # Bold Red
    }
    RESET = "\033[0m"

    def format(self, record):
        color = self.COLORS.get(record.levelno, "")
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        record.msg = f"{color}{record.msg}{self.RESET}"
        return super().format(record)


def tuple_to_list(data: Any) -> list:
    if isinstance(data, tuple):
        data = list(data)
        for i in range(len(data)):
            data[i] = tuple_to_list(data[i])
    return data


def generate_hash(text: str, algorithm: str = "sha256") -> str:
    if not isinstance(text, str):
        raise ValueError("Input text must be a string")
    hash_object = hashlib.sha256()
    hash_object.update(text.encode("utf-8"))
    return hash_object.hexdigest()


def get_nth_item_from_dict(d: dict, n: int) -> tuple[Any, Any]:
    if n < 0 or n >= len(d):
        raise IndexError("Index out of range")
    return list(d.items())[n]


def setup_logger() -> logging.Logger:
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    return logger


def to_list(data: Any) -> list:
    res = []
    if data is not None:
        res = [_ for _ in data]
    return res


def parse_demogen_config(config: dict) -> list[dict]:
    demogen_config_list = config["demonstration_configs"]
    return demogen_config_list


def parse_eval_config(config: dict) -> list[dict]:
    eval_config_list = config["evaluation_configs"]
    return eval_config_list


def parse_evalgen_config(config: dict) -> list[dict]:
    evalgen_config_list = config["evaluation_configs"]
    return evalgen_config_list


def compare_articulation_status(status1: Any, status2: Any) -> bool:
    if type(status1) == list:
        for s1, s2 in zip(status1, status2):
            if not compare_articulation_status(s1, s2):
                return False
        return True
    else:
        if not (type(status1) == float and type(status2) == list and len(status2) == 2):
            return False
        else:
            return status2[0] <= status1 <= status2[1]


def parse_usda(usda_path: str) -> str:
    with open(usda_path, "r") as f:
        usda_content = f.read()
    usda_content = usda_content.split("\n")
    for line in usda_content:
        if "prepend payload = @" in line:
            usda_path = line.split("@")[1]
            break
    return usda_path


def check_usda_exist(default_config: dict, usd_name: str) -> bool:
    usd_path = parse_usda(
        os.path.join(
            default_config["ASSETS_DIR"],
            f"{usd_name}.usda",
        )
    )
    usd_path = os.path.join(
        default_config["ASSETS_DIR"],
        *usd_name.split("/")[:-1],
        usd_path,
    )
    if not os.path.exists(usd_path):
        return False
    else:
        return True


def check_proxy_exist() -> bool:
    if (
        os.environ.get("http_proxy") is not None
        or os.environ.get("https_proxy") is not None
        or os.environ.get("all_proxy") is not None
        or os.environ.get("HTTP_PROXY") is not None
        or os.environ.get("HTTPS_PROXY") is not None
        or os.environ.get("ALL_PROXY") is not None
    ):
        return True
    else:
        return False


def parse_restart_per_success(demogen_config: dict) -> int:
    if "restart_per_success" in demogen_config:
        return demogen_config["restart_per_success"]
    elif "num_episode" in demogen_config:
        return 10**9
    elif "num_test" in demogen_config:
        return 10**9
    else:
        assert False, "demogen config is out of date, please update it."


def parse_restart_per_failed(demogen_config: dict) -> int:
    if "restart_per_failed" in demogen_config:
        return demogen_config["restart_per_failed"]
    else:
        return 10**9
