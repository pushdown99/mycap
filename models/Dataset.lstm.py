import tensorflow as tf

import os
import gc
import json
import csv
import random
import string
import pandas as pd
import collections
from tqdm import tqdm
from pickle import dump,load
from ..utils import Display

class Dataset:
  def __init__(self, config, verbose = True):
    self.verbose = verbose
    self.config  = config

  # load doc into memory
  def load_doc(self, filename):
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text

  # load a pre-defined list of photo identifiers
  def load_set(self, filename):
    doc = self.load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
      # skip empty lines
      if len(line) < 1:
        continue
      # get the image identifier
      identifier = line.split('.')[0]
      dataset.append(identifier)
    return set(dataset)

  # load clean descriptions into memory
  def load_clean_descriptions(self, filename, dataset):
    # load document
    doc = self.load_doc(filename)
    descriptions = dict()
    for line in tqdm(doc.split('\n')):
      # split line by white space
      tokens = line.split()
      # split id from description
      image_id, image_desc = tokens[0], tokens[1:]
      # skip images not in the set
      if image_id in dataset:
        # create list
        if image_id not in descriptions:
          descriptions[image_id] = list()
        # wrap description in tokens
        desc = 'sos ' + ' '.join(image_desc) + ' eos'
        # store
        descriptions[image_id].append(desc)
    return descriptions

  def load_dataset(self, descriptions, dataset):
    mapping = dict()

    for image_id, desc_list in tqdm(descriptions.items(), total=len(descriptions)):
      if image_id in dataset:
        if image_id not in mapping:
          mapping[image_id] = list()
        for i in range(len(desc_list)):
          caption = 'sos {} eos'.format(desc_list[i])
          mapping[image_id].append(caption)
    return mapping


  # covert a dictionary of clean descriptions to a list of descriptions
  def to_lines(self, descriptions):
    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d) for d in descriptions[key]]
    return all_desc

  # fit a tokenizer given caption descriptions
  def create_tokenizer(self, descriptions):
    lines = self.to_lines(descriptions)
    tokenizer = tf.keras.preprocessing.text.Tokenizer()
    tokenizer.fit_on_texts(lines)
    return tokenizer

  # extract descriptions for images
  def load_descriptions(self, doc):
    mapping = dict()
    # process lines
    for line in doc.split('\n'):
      # split line by white space
      tokens = line.split()
      if len(line) < 2:
        continue
      # take the first token as the image id, the rest as the description
      image_id, image_desc = tokens[0], tokens[1:]
      # remove filename from image id
      image_id = image_id.split('.')[0]
      # convert description tokens back to string
      image_desc = ' '.join(image_desc)
      # create the list if needed
      if image_id not in mapping:
        mapping[image_id] = list()
      # store description
      mapping[image_id].append(image_desc)
    return mapping

  def df_load_descriptions(self, df):
    mapping = dict()
    for i, r in tqdm(df.iterrows(), total=df.shape[0]):
      image_id = r['image_id']
      caption  = r['caption']

      if image_id not in mapping:
        mapping[image_id] = list()

      mapping[image_id].append(caption)
    return mapping


  def clean_descriptions(self, descriptions):
    # prepare translation table for removing punctuation
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
      for i in range(len(desc_list)):
        desc = desc_list[i]
        # tokenize
        desc = desc.split()
        # convert to lower case
        desc = [word.lower() for word in desc]
        # remove punctuation from each token
        desc = [w.translate(table) for w in desc]
        # remove hanging 's' and 'a'
        desc = [word for word in desc if len(word)>1]
        # remove tokens with numbers in them
        desc = [word for word in desc if word.isalpha()]
        # store as string
        desc_list[i] =  ' '.join(desc)

  # convert the loaded descriptions into a vocabulary of words
  def to_vocabulary(self, descriptions):
    # build a list of all description strings
    all_desc = set()
    for key in descriptions.keys():
      [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

  def save_text_data (self, descriptions, dataname):
    lines = list()
    for key, desc_list in descriptions.items():
      for desc in desc_list:
        lines.append('sos ' + desc + ' eos')
    data = '\n'.join(lines)
    file = open('files/{}_text_data.txt'.format(dataname), 'w')
    file.write(data)
    file.close()

  def save_caption_mapping_data (self, descriptions, t, dataname):
    with open('files/{}_{}_caption_mapping.json'.format(dataname, t), 'w') as f:
      json_string = json.dumps(descriptions, default=lambda o: o.__dict__, sort_keys=False, indent=2)
      f.write(json_string)

  # save descriptions to file, one per line
  def save_descriptions(self, descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
      for desc in desc_list:
        lines.append(key + ' ' + desc)
    data = '\n'.join(lines)
    file = open(filename, 'w')
    file.write(data)
    file.close()

class flickr8k (Dataset):
  def __init__(self, config, verbose = True):
    super().__init__ (config, verbose)

    dataname   = config['name']

    doc = self.load_doc(config['caption_file'])
    descriptions = self.load_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))

    self.clean_descriptions(descriptions)

    print(descriptions[list(descriptions.keys())[0]])
    vocabulary = self.to_vocabulary(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))
    self.save_descriptions(descriptions, 'files/{}_descriptions.txt'.format(dataname))
    
    train = list(self.load_set(config['train_file']))
    train = train[:config['train_limit']]
    print('Dataset: train=%d' % len(train))
    train_descriptions = self.load_dataset(descriptions, train)
    print('Descriptions: train=%d' % len(train_descriptions))

    valid = list(self.load_set(config['valid_file']))
    valid = valid[:config['valid_limit']]
    print('Dataset: valid=%d' % len(valid))
    valid_descriptions = self.load_dataset(descriptions, valid)
    print('Descriptions: valid=%d' % len(valid_descriptions))

    test = list(self.load_set(config['test_file']))
    test = test[:config['test_limit']]
    print('Dataset: test=%d' % len(test))
    test_descriptions = self.load_dataset(descriptions, test)
    print('Descriptions: test=%d' % len(test_descriptions))

    tokenizer = self.create_tokenizer(train_descriptions)

    ##
    dump(train, open('files/{}_train.pkl'.format(dataname), 'wb'))
    dump(valid, open('files/{}_valid.pkl'.format(dataname), 'wb'))
    dump(test,  open('files/{}_test.pkl'.format(dataname), 'wb'))
    dump(train_descriptions, open('files/{}_train_descriptions.pkl'.format(dataname), 'wb'))
    dump(valid_descriptions, open('files/{}_valid_descriptions.pkl'.format(dataname), 'wb'))
    dump(test_descriptions, open('files/{}_test_descriptions.pkl'.format(dataname), 'wb'))

    dump(tokenizer, open('files/{}_tokenizer.pkl'.format(dataname), 'wb'))

class nia (Dataset):
  def __init__(self, config, verbose = True):
    super().__init__ (config, verbose)

    dataname   = config['name']

    doc = self.load_doc(config['caption_file'])
    descriptions = self.load_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))
    self.clean_descriptions(descriptions)
    print(list(descriptions.keys())[0])
    print(descriptions[list(descriptions.keys())[0]])
    vocabulary = self.to_vocabulary(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))
    self.save_descriptions(descriptions, 'files/{}_descriptions.txt'.format(dataname))

    train = list(self.load_set(config['train_file']))
    train = train[:config['train_limit']]
    print('Dataset: train=%d' % len(train))
    train_descriptions = self.load_dataset(descriptions, train)
    print('Descriptions: train=%d' % len(train_descriptions))

    valid = list(self.load_set(config['valid_file']))
    valid = valid[:config['valid_limit']]
    print('Dataset: valid=%d' % len(valid))
    valid_descriptions = self.load_dataset(descriptions, valid)
    print('Descriptions: valid=%d' % len(valid_descriptions))

    test = list(self.load_set(config['test_file']))
    test = test[:config['test_limit']]
    print('Dataset: test=%d' % len(test))
    test_descriptions = self.load_dataset(descriptions, test)
    print('Descriptions: test=%d' % len(test_descriptions))

    tokenizer = self.create_tokenizer(train_descriptions)

    ##
    dump(train, open('files/{}_train.pkl'.format(dataname), 'wb'))
    dump(valid, open('files/{}_valid.pkl'.format(dataname), 'wb'))
    dump(test,  open('files/{}_test.pkl'.format(dataname), 'wb'))
    dump(train_descriptions, open('files/{}_train_descriptions.pkl'.format(dataname), 'wb'))
    dump(valid_descriptions, open('files/{}_valid_descriptions.pkl'.format(dataname), 'wb'))
    dump(test_descriptions, open('files/{}_test_descriptions.pkl'.format(dataname), 'wb'))

    dump(tokenizer, open('files/{}_tokenizer.pkl'.format(dataname), 'wb'))

class coco (Dataset):
  def __init__(self, config, verbose = True):
    super().__init__ (config, verbose)

    dataname   = config['name']
    descr_file = 'files/{}_descriptions.txt'.format(dataname)
    train_file = config['train_file']
    valid_file = config['valid_file']

    if not os.path.isfile('files/{}_descriptions.txt'.format(dataname)):

      if dataname == 'coco2014':
        df1 = pd.DataFrame(list(pd.read_json(train_file, lines=True).annotations[0]))
        df1.loc[:,'image_id'] = 'COCO_train2014_'+df1['image_id'].map('{:012d}'.format)
        df2 = pd.DataFrame(list(pd.read_json(valid_file, lines=True).annotations[0]))
        df2.loc[:,'image_id'] = 'COCO_val2014_'+df2['image_id'].map('{:012d}'.format)
      else:
        df1 = pd.DataFrame(list(pd.read_json(train_file, lines=True).annotations[0]))
        df1.loc[:,'image_id'] = 'COCO_train2017_'+df1['image_id'].map('{:012d}'.format)
        df2 = pd.DataFrame(list(pd.read_json(valid_file, lines=True).annotations[0]))
        df2.loc[:,'image_id'] = 'COCO_val2017_'+df2['image_id'].map('{:012d}'.format)


      df = pd.concat([df1, df2])
      descriptions = self.df_load_descriptions (df)
      print('Loaded: %d ' % len(descriptions))
      self.clean_descriptions(descriptions)
      print(list(descriptions.keys())[0])
      print(descriptions[list(descriptions.keys())[0]])
      vocabulary = self.to_vocabulary(descriptions)
      print('Vocabulary Size: %d' % len(vocabulary))
      self.save_descriptions(descriptions, 'files/{}_descriptions.txt'.format(dataname))

    if not os.path.isfile('files/{}_train.pkl'.format(dataname)):
      df = pd.DataFrame(list(pd.read_json(train_file, lines=True).annotations[0]))

      if dataname == 'coco2014':
        df.loc[:,'image_id'] = 'COCO_train2014_'+df['image_id'].map('{:012d}'.format)
      else:
        df.loc[:,'image_id'] = df['image_id'].map('{:012d}'.format)

      train = list(set(df.loc[:,'image_id']))
      train = train[:config['train_limit']]
      print('Dataset: train=%d' % len(train))
      train_descriptions = self.load_dataset (descriptions, train)
      print('Descriptions: train=%d' % len(train_descriptions))

      tokenizer = self.create_tokenizer(train_descriptions)

      dump(train, open('files/{}_train.pkl'.format(dataname), 'wb'))
      dump(train_descriptions, open('files/{}_train_descriptions.pkl'.format(dataname), 'wb'))

    if not os.path.isfile('files/{}_tokenizer.pkl'.format(dataname)):
      dump(tokenizer, open('files/{}_tokenizer.pkl'.format(dataname), 'wb'))

    if not os.path.isfile('files/{}_valid.pkl'.format(dataname)):
      df = pd.DataFrame(list(pd.read_json(valid_file, lines=True).annotations[0]))

      if dataname == 'coco2014':
        df.loc[:,'image_id'] = 'COCO_val2014_'+df['image_id'].map('{:012d}'.format)
      else:
        df.loc[:,'image_id'] = df['image_id'].map('{:012d}'.format)

      df = df.drop_duplicates('image_id')
      df = df.head(config['valid_limit'])

      df_valid = df.sample(frac=config['val_test_split'],random_state=200) #random state is a seed value
      df_test  = df.drop(df_valid.index).sample(frac=1.0)

      valid = list(set(df_valid.image_id))
      print('Dataset: valid=%d' % len(valid))
      valid_descriptions = self.load_dataset (descriptions, valid)
      print('Descriptions: valid=%d' % len(valid_descriptions))

      test  = list(set(df_test.image_id))
      print('Dataset: test=%d' % len(test))
      test_descriptions = self.load_dataset (descriptions, test)
      print('Descriptions: test=%d' % len(test_descriptions))

      dump(valid, open('files/{}_valid.pkl'.format(dataname), 'wb'))
      dump(test,  open('files/{}_test.pkl'.format(dataname), 'wb'))
      dump(valid_descriptions, open('files/{}_valid_descriptions.pkl'.format(dataname), 'wb'))
      dump(test_descriptions, open('files/{}_test_descriptions.pkl'.format(dataname), 'wb'))

