import torch
from torch.utils.data import DataLoader

import numpy as np
import time
import sys
import os

import catr.caption
import catr.configuration 

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
from typing import Optional, List

import torch
import torch.nn.functional as F
from torch import nn, Tensor


def build_catr_model():
    config = catr.configuration.Config()
    model, _ = catr.caption.build_model(config)
    return model