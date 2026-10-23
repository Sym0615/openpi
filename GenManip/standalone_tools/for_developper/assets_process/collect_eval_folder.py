import os
import argparse
import json
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-l", "--log_dir_parent", type=str, required=True)
    parser.add_argument("-r", "--recursive", action="store_true")
    return parser.parse_args()


def collect_eval_pkl(log_dir_parent):
    cnt = 0
    for log_dir in os.listdir(log_dir_parent):
        if os.path.isdir(os.path.join(log_dir_parent, log_dir)):
            os.rename(
                os.path.join(log_dir_parent, log_dir),
                os.path.join(log_dir_parent, f"{str(cnt).zfill(3)}"),
            )
            cnt += 1


if __name__ == "__main__":
    args = parse_args()
    if args.recursive:
        for log_dir in os.listdir(args.log_dir_parent):
            if os.path.isdir(os.path.join(args.log_dir_parent, log_dir)):
                collect_eval_pkl(os.path.join(args.log_dir_parent, log_dir))
    else:
        collect_eval_pkl(args.log_dir_parent)
