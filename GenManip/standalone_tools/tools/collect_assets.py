"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import argparse
import asyncio

from isaacsim import SimulationApp  # type: ignore[import-untyped]

kit = SimulationApp({"headless": True})
from omni.isaac.core.utils.extensions import enable_extension  # type: ignore

enable_extension("omni.kit.usd.collect")

from omni.kit.usd.collect import Collector, CollectorStatus, FlatCollectionTextureOptions  # type: ignore

VERSION = "0.1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_usd", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    return parser.parse_args()


def collect_usd(usd_path: str, target_dir: str) -> bool:
    collector = Collector(
        usd_path,
        target_dir,
        usd_only=False,
        flat_collection=True,
        skip_existing=True,
        texture_option=FlatCollectionTextureOptions.BY_USD,
    )
    asyncio.ensure_future(collector.collect())
    while (
        collector.get_status() == CollectorStatus.IN_PROGRESS
        or collector.get_status() == CollectorStatus.NOT_STARTED
    ):
        kit.update()
    if collector.get_status() == CollectorStatus.FINISHED:
        return True
    elif collector.get_status() == CollectorStatus.CANCELLED:
        print(f"Collection cancelled or failed: {usd_path}")
        return False
    else:
        print(f"Unknown collector state: {collector.get_status()}")
        return False


def main() -> None:
    args = parse_args()
    collect_usd(args.input_usd, args.output_dir)
    kit.close()


if __name__ == "__main__":
    main()
