import os
import sys
import json
import ray
import argparse

current_dir = os.path.abspath(os.path.dirname(__file__))
sys.path.append(current_dir)

from genmanip.core.evaluator.eval_server import EvalServer
from genmanip.core.evaluator.isaac_worker_pool import IsaacWorkerPool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the FastAPI server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8087,
        help="Port for the FastAPI server",
    )
    parser.add_argument(
        "--run_id",
        default=None,
        type=str,
        help="Run id for the evaluation",
    )
    parser.add_argument(
        "-l",
        "--local",
        action="store_true",
        help="Run in local mode, a quick command to enable Isaac Sim GUI",
    )
    parser.add_argument(
        "-n",
        "--num_steps",
        type=int,
        default=None,
        help="Number of steps to run the evaluation",
    )
    parser.add_argument(
        "-ira",
        "--is_relative_action",
        action="store_true",
        help="Run in relative action mode, the action is relative to the last action",
    )

    parser.add_argument(
        "-wor",
        "--without_render",
        action="store_true",
        help="Run in without render mode, only record the data",
    )
    parser.add_argument(
        "-rr",
        "--random_randomization",
        action="store_true",
        help="Run in random randomization mode, enable randomization configs in eval config",
    )
    parser.add_argument(
        "--episode_recorder_save_every",
        type=int,
        default=0,
        help="Save one episode recorder frame every N steps; 0 disables image saving (default: 0)",
    )
    parser.add_argument(
        "--save_process",
        dest="save_process",
        action="store_true",
        default=True,
        help="Save server-side process artifacts such as trajectory metadata, videos, and RRD files (default: enabled)",
    )
    parser.add_argument(
        "--no_save_process",
        dest="save_process",
        action="store_false",
        help="Disable saving server-side process artifacts",
    )
    parser.add_argument(
        "--step_timeout",
        type=float,
        default=600.0,
        help="Server timeout for /step in seconds (default: 600)",
    )
    parser.add_argument(
        "--reset_timeout",
        type=float,
        default=600.0,
        help="Server timeout for /reset in seconds (default: 600)",
    )
    parser.add_argument(
        "--create_timeout",
        type=float,
        default=600.0,
        help="Server timeout for /create_workers in seconds (default: 600)",
    )
    parser.add_argument(
        "--kill_timeout",
        type=float,
        default=120.0,
        help="Server timeout for /kill in seconds (default: 120)",
    )
    parser.add_argument(
        "--load_config_timeout",
        type=float,
        default=60.0,
        help="Server timeout for /start_new_job in seconds (default: 60)",
    )
    parser.add_argument(
        "--worker_restart_memory_gib",
        type=float,
        default=15.0,
        help="Restart a worker actor during reset when its VmRSS exceeds this threshold in GiB; <=0 disables (default: 15)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print("Starting Ray Eval Server...")
    ray_context = ray.init()
    print(ray.cluster_resources())
    server = EvalServer(
        args.host,
        args.port,
        workers=1,
        step_timeout=args.step_timeout,
        reset_timeout=args.reset_timeout,
        create_timeout=args.create_timeout,
        kill_timeout=args.kill_timeout,
        load_config_timeout=args.load_config_timeout,
    )

    print("Creating Worker Pool...")
    pool = IsaacWorkerPool(
        args, current_dir, world_size=ray.cluster_resources().get("GPU", 0)
    )
    server.register_worker_pool(pool)
    server.run()
