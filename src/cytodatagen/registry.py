"""
Provides class registries for deserialization from json.
"""


class Registry:
    def __init__(self):
        self._registry = dict()

    def register(self, key, value):
        if key not in self._registry:
            self._registry[key] = value
        else:
            raise KeyError(f"key '{key}' is already registered")

    def register_class(self, key):
        def register_class(cls):
            self.register(key, cls)
            return cls
        return register_class

    def get(self, key):
        if key in self._registry:
            return self._registry[key]
        else:
            raise KeyError(f"{key} has not been registered")

    def unregister(self, key):
        if key in self._registry:
            del self._registry[key]
        else:
            raise KeyError(f"{key} has not been registered")

    def __getitem__(self, key):
        return self.get(key)

    def from_dict(self, data):
        key = data.pop("_target_")
        cls = self.get(key)
        return cls.from_dict(data)


xform_registry = Registry()
pop_registry = Registry()
mdist_registry = Registry()
dist_registry = Registry()
