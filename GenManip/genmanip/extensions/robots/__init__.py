import importlib
import os

_dir = os.path.dirname(__file__)

for root, dirs, files in os.walk(_dir):
    for file in files:
        if file.endswith(".py") and file != "__init__.py":
            rel_path = os.path.relpath(os.path.join(root, file), _dir)
            module_name = rel_path[:-3].replace(os.path.sep, ".")
            importlib.import_module(f".{module_name}", package=__name__)
