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

  def load_dataset(self, images_dir, descriptions, dataset, title='[title]', number_of_captions = 5):
    map1 = dict()
    map2 = dict()

    for image_id, desc_list in tqdm(descriptions.items(), total=len(descriptions), desc=title):
      if image_id in dataset:
        image_path = '{}/{}.jpg'.format(images_dir, image_id)
        if image_id not in map1:
          map1[image_id]   = list()
          map2[image_path] = list()
        for i in range(len(desc_list)):
          caption = 'sos {} eos'.format(desc_list[i])
          map1[image_id].append(caption)
          map2[image_path].append(caption)
    return map1, map2


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

  def df_load_descriptions(self, df, title='[title]', number_of_captions = 5):
    mapping = dict()
    for i, r in tqdm(df.iterrows(), total=df.shape[0], desc=title):
      image_id = r['image_id']
      caption  = r['caption']

      if image_id not in mapping:
        mapping[image_id] = list()

      if len(mapping[image_id]) >= number_of_captions:
        continue
      mapping[image_id].append(caption)

    # cleasing description, caption data
    self.clean_descriptions (mapping)

    for k, v in mapping.items():
      if not os.path.isfile(k):
        mapping.pop(k)
        print ('pop (not os.path.isfile): ', k)
      elif len(v) != number_of_captions:
        mapping.pop(k)
        print ('pop (number_of_captions): ', k, len(v))

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

  def save_text_data(self, descriptions, filename):
    lines = list()
    for key, desc_list in descriptions.items():
      for desc in desc_list:
        lines.append('sos ' + desc + ' eos')
    with open(filename, mode='w', encoding='utf-8') as f:
      json.dump(lines, f, indent=2)

class flickr8k (Dataset):
  def __init__(self, config, verbose = True):
    super().__init__ (config, verbose)

    dataname   = config['name']
    images_dir = config['images_dir']

    doc = self.load_doc(config['caption_file'])
    descriptions = self.load_descriptions(doc)
    print('Loaded: %d ' % len(descriptions))

    self.clean_descriptions(descriptions)

    print(descriptions[list(descriptions.keys())[0]])
    vocabulary = self.to_vocabulary(descriptions)
    print('Vocabulary Size: %d' % len(vocabulary))
    self.save_descriptions(descriptions, 'files/{}_descriptions.txt'.format(dataname))
    self.save_text_data(descriptions, 'files/{}_text_data.txt'.format(dataname))
    
    train = list(self.load_set(config['train_file']))
    train = train[:config['train_limit']]
    print('Dataset: train=%d' % len(train))
    train_descriptions, train_data = self.load_dataset(images_dir[0], descriptions, train)
    print('Descriptions: train=%d' % len(train_descriptions))

    valid = list(self.load_set(config['valid_file']))
    valid = valid[:config['valid_limit']]
    print('Dataset: valid=%d' % len(valid))
    valid_descriptions, valid_data = self.load_dataset(images_dir[0], descriptions, valid)
    print('Descriptions: valid=%d' % len(valid_descriptions))

    test = list(self.load_set(config['test_file']))
    test = test[:config['test_limit']]
    print('Dataset: test=%d' % len(test))
    test_descriptions, test_data = self.load_dataset(images_dir[0], descriptions, test)
    print('Descriptions: test=%d' % len(test_descriptions))

    #tokenizer = self.create_tokenizer(train_descriptions)

    ##
    dump(train, open('files/{}_train.pkl'.format(dataname), 'wb'))
    dump(valid, open('files/{}_valid.pkl'.format(dataname), 'wb'))
    dump(test,  open('files/{}_test.pkl'.format(dataname), 'wb'))
    dump(train_descriptions, open('files/{}_train_descriptions.pkl'.format(dataname), 'wb'))
    dump(valid_descriptions, open('files/{}_valid_descriptions.pkl'.format(dataname), 'wb'))
    dump(test_descriptions, open('files/{}_test_descriptions.pkl'.format(dataname), 'wb'))

    with open('files/{}_train_data.json'.format(dataname), "w") as f:
      json.dump(train_data, f, indent=2)
    with open('files/{}_valid_data.json'.format(dataname), "w") as f:
      json.dump(valid_data, f, indent=2)
    with open('files/{}_test_data.json'.format(dataname), "w") as f:
      json.dump(test_data, f, indent=2)

    #dump(tokenizer, open('files/{}_tokenizer.pkl'.format(dataname), 'wb'))

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
    self.save_text_data(descriptions, 'files/{}_text_data.txt'.format(dataname))

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

    name   = config['name']
    images = config['images_dir']
    trains = config['train_file']
    valids = config['valid_file']
    limit1 = config['train_limit']
    limit2 = config['valid_limit']
    split  = config['val_test_split']
    number_of_captions    = config['number_of_captions']

    self.get_data (name, images, trains, valids, limit1, limit2, split, True, number_of_captions, 2)

  def get_data (self, name, images, trains, valids, limit1, limit2, split, force = False, number_of_captions = 5, indent = None):
    if not force and os.path.isfile('files/{}_descr.json'.format(name)):
      return

    if name == 'coco2014':
      df1 = pd.DataFrame(list(pd.read_json(trains, lines=True).annotations[0]))
      df1.loc[:,'image_id'] = images[0] + '/COCO_train2014_'+df1['image_id'].map('{:012d}'.format) +'.jpg'
      df2 = pd.DataFrame(list(pd.read_json(valids, lines=True).annotations[0]))
      df2.loc[:,'image_id'] = images[1] + '/COCO_val2014_'+df2['image_id'].map('{:012d}'.format) +'.jpg'
    else:
      df1 = pd.DataFrame(list(pd.read_json(trains, lines=True).annotations[0]))
      df1.loc[:,'image_id'] = images[0] + '/'+df1['image_id'].map('{:012d}'.format) + '.jpg'
      df2 = pd.DataFrame(list(pd.read_json(valids, lines=True).annotations[0]))
      df2.loc[:,'image_id'] = images[1] + '/'+df2['image_id'].map('{:012d}'.format) + '.jpg'

    df  = pd.concat([df1, df2])
    ds  = self.df_load_descriptions (df,  'Descriptions ', number_of_captions)
    vo  = self.to_vocabulary(ds) 
    dx  = list(df['caption'])            # list for caption data

    ds1 = self.df_load_descriptions (df1, 'Train dataset', number_of_captions)
    ds1 = dict(list(ds1.items())[:limit1])

    dst = self.df_load_descriptions (df2, 'Valid dataset', number_of_captions)
    dst = dict(list(dst.items())[:limit2])
    ds2 = dict(list(dst.items())[:int(len(dst)*split)])
    ds3 = dict(list(dst.items())[int(len(dst)*split):])

    print('Loaded: {}/df#{} (train={} valid={}, test={}, vocabulary={})'.format(len(ds), df.shape[0], len(ds1), len(ds2), len(ds3), len(vo)))

    json.dump(ds, open('files/{}_descr.json'.format(name), 'w'), indent=indent)
    json.dump(ds1,open('files/{}_train.json'.format(name), 'w'), indent=indent)
    json.dump(ds2,open('files/{}_valid.json'.format(name), 'w'), indent=indent)
    json.dump(ds3,open('files/{}_test.json'.format(name),  'w'), indent=indent)
    json.dump(dx, open('files/{}_text.json'.format(name),  'w'), indent=indent)

