MODEL_BUILDERS = {}
BUILDERS = {}
SAMPLE_METHODS = {}

def register_model(fn):
    MODEL_BUILDERS[fn.__name__] = fn
    return fn

def register_builder(fn):
    BUILDERS[fn.__name__] = fn
    return fn

def register_sample_method(fn):
    SAMPLE_METHODS[fn.__name__] = fn
    return fn
