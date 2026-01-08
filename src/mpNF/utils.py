import torch.nn as nn
import torch

from mpNF.modules import *

def get_module(module_name) : 
    try:
        module = globals()[module_name]
    except KeyError :
        raise ValueError(f"Invalid module name : {module_name}")
    return module