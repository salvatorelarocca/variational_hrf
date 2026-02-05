import torch as th

def pretty_shape(obj):
    if isinstance(obj, th.Tensor):
        return tuple(obj.shape)
    elif isinstance(obj, (list, tuple)):
        return [pretty_shape(o) for o in obj]
    else:
        return type(obj).__name__


def unet_shape_hook(name):
    def hook(module, inp, out):
        print(f"\n[{name}] {module.__class__.__name__}")
        print("  input :", pretty_shape(inp))
        print("  output:", pretty_shape(out))
    return hook
