import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, time, random
import tensorflow as tf
from transformers import *
from torch.utils.data import Dataloader


from transformers import AutoModel, AutoTokenizer


GPTtokenizer = GPT2Tokenizer.from_pretrained('heegyu/gpt2-emotion')


model_name = "textattack/bert-base-uncased-SST-2"
BERTmodel = AutoModel.from_pretrained(model_name)
BERTtokenizer = AutoTokenizer.from_pretrained(model_name)