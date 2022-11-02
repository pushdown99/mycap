import os
import json
import string

from os import path
from os import makedirs
from os import listdir
from numpy import array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import ModelCheckpoint

from .. import utils
from .dataset import Dataset

from itertools  import repeat
from os.path    import isfile
from pickle     import dump,load
from tqdm       import tqdm

class Dataset(Dataset):
  def __init__(self, data_opt, model_opt, verbose = True):
    super().__init__(verbose)

    self.data_opt  = data_opt
    self.model_opt = model_opt
    self.verbose   = verbose

    for dir in data_opt['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

  def get_data (self):
    Data = {
      'dataset_train':      self.train,
      'dataset_valid':      self.valid,
      'dataset_test':       self.test,
      'img_name_train':     self.img_name_train,
      'img_name_valid':     self.img_name_valid,
      'img_name_test':      self.img_name_test,
      'train_descriptions': self.train_descriptions,
      'valid_descriptions': self.valid_descriptions,
      'test_descriptions':  self.test_descriptions,
      'train_features':     self.train_features,
      'valid_features':     self.valid_features,
      'test_features':      self.test_features,
      'vocab_size':         self.vocab_size,
      'max_length':         self.max_length,
      'tokenizer':          self.tokenizer,
    }
    return Data

  def load_data (self):
    self.prepare_mapping_data_flickr8k ()

    ###
    features = 'files/features_{}_{}.pkl'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    mapping  = 'files/mapping_{}_{}.txt'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])

    self.train = self.load_set(self.data_opt['images_train'], self.data_opt['limits_train'])
    self.valid = self.load_set(self.data_opt['images_valid'], self.data_opt['limits_valid'])
    self.test  = self.load_set(self.data_opt['images_test'],  self.data_opt['limits_test'])

    self.img_name_train = [x for item in self.train for x in repeat(item, 5)]
    self.img_name_valid = [x for item in self.valid for x in repeat(item, 5)]
    self.img_name_test  = [x for item in self.test  for x in repeat(item, 5)]

    self.train_descriptions = self.load_clean_descriptions(mapping, self.train)
    self.valid_descriptions = self.load_clean_descriptions(mapping, self.valid)
    self.test_descriptions  = self.load_clean_descriptions(mapping, self.test)

    self.train_features = self.load_photo_features(features, self.train) 
    self.valid_features = self.load_photo_features(features, self.valid) 
    self.test_features  = self.load_photo_features(features, self.test) 

    ###
    filename = 'files/tokenizer_{}_{}.pkl'.format(self.data_opt['dataset_name'],self.model_opt['model_name'])
    # only create tokenizer if it does not exist
    if not isfile(filename):
        self.tokenizer = self.create_tokenizer(self.train_descriptions)
        # save the tokenizer
        dump(self.tokenizer, open(filename, 'wb'))
    else:
        self.tokenizer = load(open(filename, 'rb'))
    # define vocabulary size
    self.vocab_size = len(self.tokenizer.word_index) + 1
    # determine the maximum sequence length
    self.max_length = self.max_len(self.train_descriptions)

    if self.verbose:
      print('  Vocabulary Size: %d' % self.vocab_size)
      print('  Description Length: %d' % self.max_length)
      print(list(self.tokenizer.word_index.items())[:5])
      print(list(self.tokenizer.word_index.items())[-5:])

    return self.get_data()
