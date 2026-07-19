import sys, os, builtins
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
os.chdir(os.path.join(os.path.dirname(__file__)))
_orig = builtins.__import__
def _skip(name, *a, **kw):
    if name.startswith("isre"): raise ImportError()
    return _orig(name, *a, **kw)
builtins.__import__ = _skip
from train_urcm import train_system
train_system()
