"""
Copyright (c) 2025 Ning Gao, Shanghai Artificial Intelligence Laboratory
All rights reserved.

Licensed under the MIT License.
"""


class Registry:
    def __init__(self):
        self._registry = {}

    def register(self, name):
        """Decorator to register a class."""

        def decorator(cls):
            if name in self._registry:
                raise KeyError(f"{name} is already registered")
            self._registry[name] = cls
            return cls

        return decorator

    def get(self, name):
        if name not in self._registry:
            raise KeyError(f"{name} is not registered")
        return self._registry[name]

    def build(self, name, *args, **kwargs):
        cls = self.get(name)
        return cls(*args, **kwargs)
