"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""

import os


_ASSET_PATTERN_PREFIXES = ("grep:", "regex:", "re:")
_REGEX_SPECIAL_CHARS = set("^$*+?{}[]|()\\")


def strip_asset_pattern_prefix(path_expr: str) -> str:
    stripped_path_expr = path_expr.strip()
    for prefix in _ASSET_PATTERN_PREFIXES:
        if stripped_path_expr.startswith(prefix):
            return stripped_path_expr[len(prefix) :]
    return stripped_path_expr


def extract_asset_search_pattern(path_expr: str) -> str | None:
    stripped_path_expr = path_expr.strip()
    stripped_pattern = strip_asset_pattern_prefix(path_expr)
    if stripped_pattern != stripped_path_expr:
        return stripped_pattern
    if any(char in stripped_pattern for char in _REGEX_SPECIAL_CHARS):
        return stripped_pattern
    return None


def get_asset_search_root(assets_dir: str, path_expr: str) -> tuple[str, str]:
    pattern = strip_asset_pattern_prefix(path_expr)
    search_root = assets_dir
    current_dir = assets_dir

    # Only walk inside the longest leading literal directory prefix.
    for segment in pattern.split("/"):
        if segment in ("", "."):
            break
        if any(char in segment for char in _REGEX_SPECIAL_CHARS):
            break
        next_dir = os.path.join(current_dir, segment)
        if not os.path.isdir(next_dir):
            break
        search_root = next_dir
        current_dir = next_dir

    return pattern, search_root
