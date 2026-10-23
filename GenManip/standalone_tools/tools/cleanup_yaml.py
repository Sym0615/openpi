import argparse
import os
from typing import Any, Dict, List

import yaml


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yaml_path", type=str, required=True)
    parser.add_argument(
        "--preserve-anchors",
        action="store_true",
        help="Detect duplicate dicts and create YAML anchors/aliases",
    )
    return parser.parse_args()


def dict_to_hashable(obj: Any) -> Any:
    """Convert dict to tuple for hashing to detect duplicates"""
    if isinstance(obj, dict):
        return tuple(sorted((k, dict_to_hashable(v)) for k, v in obj.items()))
    elif isinstance(obj, list):
        return tuple(dict_to_hashable(item) for item in obj)
    elif isinstance(obj, (str, int, float, bool, type(None))):
        return obj
    else:
        return str(obj)


def deduplicate_dicts(data: Any, memo: Dict[Any, Any] | None = None) -> Any:
    """Replace duplicate dicts with references to the same object to enable YAML anchors"""
    if memo is None:
        memo = {}

    if isinstance(data, dict):
        # First, recursively process all values
        processed = {}
        for key, value in data.items():
            processed[key] = deduplicate_dicts(value, memo)

        # Check if this dict structure already exists
        hashable = dict_to_hashable(processed)
        if hashable in memo:
            # Return the existing object reference
            return memo[hashable]
        else:
            # Store this as a new unique structure
            memo[hashable] = processed
            return processed

    elif isinstance(data, list):
        return [deduplicate_dicts(item, memo) for item in data]

    else:
        return data


def main() -> None:
    args = parse_args()
    if args.yaml_path.endswith(".yml") or args.yaml_path.endswith(".yaml"):
        try:
            with open(args.yaml_path, "r") as f:
                yaml_data = yaml.load(f, Loader=yaml.FullLoader)

            # Optionally deduplicate to create anchors/aliases
            if args.preserve_anchors:
                yaml_data = deduplicate_dicts(yaml_data)

            with open(args.yaml_path, "w") as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
            print(f"Successfully cleaned: {args.yaml_path}")
        except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
            print(f"Error processing {args.yaml_path}: {e}")
    elif os.path.isdir(args.yaml_path):
        for root, dirs, files in os.walk(args.yaml_path):
            for file in files:
                if file.endswith(".yml") or file.endswith(".yaml"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r") as f:
                            yaml_data = yaml.load(f, Loader=yaml.FullLoader)

                        # Optionally deduplicate to create anchors/aliases
                        if args.preserve_anchors:
                            yaml_data = deduplicate_dicts(yaml_data)

                        with open(file_path, "w") as f:
                            yaml.dump(
                                yaml_data, f, default_flow_style=False, sort_keys=False
                            )
                        print(f"Successfully cleaned: {file_path}")
                    except (OSError, yaml.YAMLError, TypeError, ValueError) as e:
                        print(f"Error processing {file_path}: {e}")


if __name__ == "__main__":
    main()
