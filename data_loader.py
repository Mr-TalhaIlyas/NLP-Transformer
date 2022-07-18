# -*- coding: utf-8 -*-
"""
Created on Mon Jul 18 17:09:19 2022

@author: talha
"""

import os, pathlib
import tensorflow as tf
import tensorflow_text

from itertools import islice
from termcolor import colored, cprint
tf.get_logger().setLevel('ERROR')



def build_tf_dataset(path_pt, path_en):
    pt_train = pathlib.Path(path_pt).read_text(encoding="utf-8").splitlines()
    en_train = pathlib.Path(path_en).read_text(encoding="utf-8").splitlines()
    
    # convert the list to tensor
    pt_train = tf.convert_to_tensor(pt_train, dtype=tf.string)
    en_train = tf.convert_to_tensor(en_train, dtype=tf.string)
    
    # make dataset
    ds = tf.data.Dataset.from_tensor_slices((pt_train, en_train))
    return ds


def filter_max_tokens(pt, en):
    '''
    Sequence greater with length greater than MAX_TOKENS will not pass.
    '''
    num_tokens = tf.maximum(tf.shape(pt)[1],tf.shape(en)[1])
    return num_tokens < MAX_TOKENS

def tokenize_pairs(pt, en):
    pt = tokenizers.pt.tokenize(pt)
    # Convert from ragged to dense, padding with zeros.
    pt = pt.to_tensor()

    en = tokenizers.en.tokenize(en)
    # Convert from ragged to dense, padding with zeros.
    en = en.to_tensor()
    return pt, en

def make_batches(ds):
  return (
      ds
      .cache()
      .shuffle(BUFFER_SIZE)
      .batch(BATCH_SIZE)
      # 1st mapping from language to tokens will happen
      .map(tokenize_pairs, num_parallel_calls=tf.data.AUTOTUNE) 
      # 2nd filtering will be done on tokenized data.
      .filter(filter_max_tokens) 
      .prefetch(tf.data.AUTOTUNE))
#%%

PATH = os.getcwd()
MAX_TOKENS = 128
BUFFER_SIZE = 20000
BATCH_SIZE = 64

tokenizer_model = './tokenizer/'
tokenizers = tf.saved_model.load(tokenizer_model)
#[print(item) for item in dir(tokenizers.en) if not item.startswith('_')]

# Load your data files
train_data = build_tf_dataset('./data/pt_train.txt', './data/en_train.txt')
val_data = build_tf_dataset('./data/pt_val.txt', './data/en_val.txt')

'''
Code sanity check
'''
for pt_examples, en_examples in train_data.batch(2).take(1):
    for pt in pt_examples.numpy():
      cprint(f"Raw PT Data :\n {pt.decode('utf-8')}", 'green')
    
    print()
    
    for en in en_examples.numpy():
      cprint(f"Raw EN Data :\n {en.decode('utf-8')}", 'magenta')
    
    pt, en = tokenize_pairs(pt_examples, en_examples)
    cprint(f'Tokenized data PT :\n {pt.numpy()}', 'yellow')
    cprint(f'Tokenized data EN :\n {en.numpy()}', 'cyan')

train_batches = make_batches(train_data)
val_batches = make_batches(val_data)
train_samples = train_data.__len__().numpy()

pt_batch, en_batch = next(islice(train_batches, 7, None))
pt_batch = pt_batch.numpy()
en_batch = en_batch.numpy()
