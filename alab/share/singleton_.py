def singleton(*args, **kwargs):
    def init(cls):
        instances = {}

        def get_instance():
            if cls not in instances:
                instances[cls] = cls(*args, **kwargs)
            return instances[cls]

        return get_instance()
    return init



