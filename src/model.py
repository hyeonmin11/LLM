import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os, time, random
import tensorflow as tf
from transformers import *
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from ftfy import fix_text
import io


tokenizer = GPT2Tokenizer.from_pretrained('heegyu/gpt2-emotion')


model_name = "textattack/bert-base-uncased-SST-2"
BERTmodel = AutoModel.from_pretrained(model_name)
BERTtokenizer = AutoTokenizer.from_pretrained(model_name)

# drive mount
from google.colab import drive
drive.mount('/content/drive')
#!cd /content/drive/MyDrive/data
#!tar -xvf /content/drive/MyDrive/aclImdb_v1.tar.gz
path = "/content/drive/MyDrive/data/aclImdb/"

class MovieReviewDataset(Dataset):
  """
pytorch dataset class for loading data. 
This is where the data parsing happens. 
This class is built with reusaility in mind

  """
  def __init__(self, path, use_tokenizer):
    #check if path exists
    if not os.path.isdir(path):
      raise ValueError('Invalid path variable, Needs to be a directory')
    self.texts= []
    self.labels= []

    for label in ['pos', 'neg']:
      sentiment_path = os.path.join(path, label)
      file_names = os.listdir(sentiment_path)
      for file_name in tqdm(files_names, desc=f'{label} files'):
        file_path = os.path.join(sentiment_path, file_name)

        content = io.open(file_path, mode='r', encoding='utf-8').read()
        content = fix_text(content)
        self.texts.append(content)
        self.labels.append(label)
    
    self.n_examples = len(self.labels)
    return

  def __len__(self):
    return self.n_examples
  def __getitem__(self, item):
    return {'text': self.texts[item], 'label':self.labels[item]}
  



print('Dealing with Train')
#Create pytorch dataset
train_dataset = MovieReviewDataset(path = '/content/aclImdb/train',
                                   use_tokenizer = tokenizer,
                                   labels_ids = labels_ids,
                                   max_sequence_len = max_length)

print('Created train dataset with %d examples'%len(train_dataset))

train_dataloadoer = DataLoader(train_dataset, batch_size = batch_size, shuffle=True)
print('Created 'train_dataloader' with %d batches'%len(train_dataloader))
print()
print('Dealing with Val')
test_dataset = MovieReviewDataset(path='/content/aclImdb/test',
                                   use_tokenizer = tokenizer,
                                   labels_ids = labels_ids,
                                   max_sequence_len = max_length)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)



