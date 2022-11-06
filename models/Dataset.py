import tensorflow as tf

import os
import gc
import json
import glob
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

    self.clean_descriptions (mapping)

    drops = list()
    for k, v in mapping.items():
      if not os.path.isfile(k):
        drops.append(k)
        print ('pop (not os.path.isfile): ', k)
      elif len(v) != number_of_captions:
        drops.append(k)
        print ('pop (number_of_captions): ', k, len(v))

    for k in drops:
      mapping.pop(k)

    return mapping

  def clean_descriptions(self, descriptions):
    table = str.maketrans('', '', string.punctuation)
    for key, desc_list in descriptions.items():
      for i in range(len(desc_list)):
        desc = desc_list[i]
        desc = desc.split()
        desc = [word.lower() for word in desc]
        desc = [w.translate(table) for w in desc]
        desc = [word for word in desc if len(word)>1]
        desc = [word for word in desc if word.isalpha()]
        desc_list[i] =  ' '.join(desc)

  def to_vocabulary(self, descriptions):
    all_desc = set()
    for key in descriptions.keys():
      [all_desc.update(d.split()) for d in descriptions[key]]
    return all_desc

class flickr8k (Dataset):
  def __init__(self, config, verbose = True, force = True):
    super().__init__ (config, verbose)

    name    = config['name']
    images  = config['images_dir']
    caption = config['caption_file']
    trains  = config['train_file']
    valids  = config['valid_file']
    tests   = config['test_file']
    limit1  = config['train_limit']
    limit2  = config['valid_limit']
    limit3  = config['test_limit']
    number_of_captions    = config['number_of_captions']

    self.get_data (name, images, caption, trains, valids, tests, limit1, limit2, limit3, force, number_of_captions, 2)

  def get_data (self, name, images, caption, trains, valids, tests, limit1, limit2, limit3, force = False, number_of_captions = 5, indent = None):
    if not force and os.path.isfile('files/{}_descr.json'.format(name)):
      return

    df  = pd.read_csv(caption, lineterminator='\n', names=['image_id', 'caption'], sep='\t')
    df1 = pd.read_csv(trains,  lineterminator='\n', names=['image_id'])
    df2 = pd.read_csv(valids,  lineterminator='\n', names=['image_id'])
    df3 = pd.read_csv(tests,   lineterminator='\n', names=['image_id'])
    df.loc[:,'image_id']  = images[0]+'/'+df.image_id.str.split('#').str[0]
    df1.loc[:,'image_id'] = images[0]+'/'+df1.image_id.str.split('#').str[0]
    df2.loc[:,'image_id'] = images[0]+'/'+df2.image_id.str.split('#').str[0]
    df3.loc[:,'image_id'] = images[0]+'/'+df3.image_id.str.split('#').str[0]
    df1 = df.loc[df['image_id'].isin(list(df1.image_id))]
    df2 = df.loc[df['image_id'].isin(list(df2.image_id))]
    df3 = df.loc[df['image_id'].isin(list(df3.image_id))]

    ds  = self.df_load_descriptions (df,  'Descriptions ', number_of_captions)
    vo  = self.to_vocabulary(ds) 
    dx  = list(df['caption'])            # list for caption data
    ds1 = self.df_load_descriptions (df1, 'Train dataset', number_of_captions)
    ds1 = dict(list(ds1.items())[:limit1])
    ds2 = self.df_load_descriptions (df2, 'Valid dataset', number_of_captions)
    ds2 = dict(list(ds2.items())[:limit2])
    ds3 = self.df_load_descriptions (df3, 'Test dataset ', number_of_captions)
    ds3 = dict(list(ds3.items())[:limit3])

    print('Loaded: {}/df#{} (train={} valid={}, test={}, vocabulary={})'.format(len(ds), df.shape[0], len(ds1), len(ds2), len(ds3), len(vo)))

    json.dump(ds, open('files/{}_descr.json'.format(name), 'w'), indent=indent)
    json.dump(ds1,open('files/{}_train.json'.format(name), 'w'), indent=indent)
    json.dump(ds2,open('files/{}_valid.json'.format(name), 'w'), indent=indent)
    json.dump(ds3,open('files/{}_test.json'.format(name),  'w'), indent=indent)
    json.dump(dx, open('files/{}_text.json'.format(name),  'w'), indent=indent)

class nia (Dataset):
  def __init__(self, config, verbose = True, force = True, indent = None):
    super().__init__ (config, verbose)

    name    = config['name']
    data    = config['data_dir']
    jsons   = config['json_dir']
    images  = config['images_dir']
    caption = config['caption_file']
    trains  = config['train_file']
    valids  = config['valid_file']
    tests   = config['test_file']
    limit1  = config['train_limit']
    limit2  = config['valid_limit']
    limit3  = config['test_limit']
    split1  = config['train_val_split']
    split2  = config['val_test_split']
    number_of_captions    = config['number_of_captions']

    self.get_label (name, data, jsons, images[0], split1, split2)
    self.get_data  (name, images, caption, trains, valids, tests, limit1, limit2, limit3, True, number_of_captions, indent)

  def get_label (self, name, data_dir, json_dir, image_dir, split1, split2, force = False):
    if not force and os.path.isfile('{}/{}.token.kor.txt'.format(data_dir, name)):
      return

    if name == 'nia0404':
      frelation = open('{}/{}.relation.txt'.format(data_dir, name), 'w')
    fetoken   = open('{}/{}.token.eng.txt'.format(data_dir, name),  'w')
    fktoken   = open('{}/{}.token.kor.txt'.format(data_dir, name),  'w')
    fimages   = open('{}/{}.images.txt'.format(data_dir, name),     'w')
    ftrain    = open('{}/{}.trainImages.txt'.format(data_dir, name),'w')
    fvalid    = open('{}/{}.devImages.txt'.format(data_dir, name),  'w')
    ftest     = open('{}/{}.testImages.txt'.format(data_dir, name), 'w')

    images = []

    for file in tqdm(glob.glob(json_dir + '/*.json')):
      with open(file, encoding='utf-8-sig') as f:
        try:
          data = json.load(f)
          filename = data['images'][0]['file_name']
          image = filename.split('.')[0]
          if not os.path.isfile(image_dir + '/' + filename):
            print ('not found: ' + image_dir + filename + ' => ' + file)
            continue

          images.append ('{}.jpg'.format(image))
          for idx,caption in enumerate(data['annotations'][0]['text']):
            if name == 'nia0404':
              frelation.write ("{}.jpg#{}\t{}\t{}\t{}\n".format(image, idx,caption['entity1'],caption['entity2'],caption['relation']))
            fetoken.write ("{}.jpg#{}\t{}\n".format(image, idx,caption['english']))
            fktoken.write ("{}.jpg#{}\t{}\n".format(image, idx,caption['korean']))
          if idx != 9:
            print (idx, file)
        except JSONDecodeError as e:
          print ('Decoding JSON has failed: ', file, e)

    random.shuffle(images)
    fimages.write('\n'.join(images))

    random.shuffle(images)
    nsize  = len(images)
    ntrain = int(nsize*split1)
    nvalid = int((nsize - int(nsize*split1))*split2)
    ntest = nsize - ntrain - nvalid
    print (nsize, ntrain, nvalid, ntest)

    train = images[:ntrain]
    valid = images[ntrain:ntrain+nvalid]
    test  = images[ntrain+nvalid:]

    print(len(train), len(valid), len(test))

    ftrain.write('\n'.join(train))
    fvalid.write('\n'.join(valid))
    ftest.write('\n'.join(test))
    
    if name == 'nia0404':
      frelation.close()  
    fimages.close()  
    fetoken.close()  
    fktoken.close()  
    ftrain.close()
    fvalid.close()
    ftest.close()

  def get_data (self, name, images, caption, trains, valids, tests, limit1, limit2, limit3, force, number_of_captions, indent):
    if not force and os.path.isfile('files/{}_descr.json'.format(name)):
      return

    df  = pd.read_csv(caption, lineterminator='\n', names=['image_id', 'caption'], sep='\t')
    df1 = pd.read_csv(trains,  lineterminator='\n', names=['image_id'])
    df2 = pd.read_csv(valids,  lineterminator='\n', names=['image_id'])
    df3 = pd.read_csv(tests,   lineterminator='\n', names=['image_id'])
    df.loc[:,'image_id']  = images[0]+'/'+df.image_id.str.split('#').str[0]
    df1.loc[:,'image_id'] = images[0]+'/'+df1.image_id.str.split('#').str[0]
    df2.loc[:,'image_id'] = images[0]+'/'+df2.image_id.str.split('#').str[0]
    df3.loc[:,'image_id'] = images[0]+'/'+df3.image_id.str.split('#').str[0]
    df1 = df.loc[df['image_id'].isin(list(df1.image_id))]
    df2 = df.loc[df['image_id'].isin(list(df2.image_id))]
    df3 = df.loc[df['image_id'].isin(list(df3.image_id))]

    print (df.head())
    ds  = self.df_load_descriptions (df,  'Descriptions ', number_of_captions)
    vo  = self.to_vocabulary(ds) 
    dx  = list(df['caption'])            # list for caption data

    ds1 = self.df_load_descriptions (df1, 'Train dataset', number_of_captions)
    ds1 = dict(list(ds1.items())[:limit1])
    ds2 = self.df_load_descriptions (df2, 'Valid dataset', number_of_captions)
    ds2 = dict(list(ds2.items())[:limit2])
    ds3 = self.df_load_descriptions (df3, 'Test dataset ', number_of_captions)
    ds3 = dict(list(ds3.items())[:limit3])

    print('Loaded: {}/df#{} (train={} valid={}, test={}, vocabulary={})'.format(len(ds), df.shape[0], len(ds1), len(ds2), len(ds3), len(vo)))

    json.dump(ds, open('files/{}_descr.json'.format(name), 'w'), indent=indent)
    json.dump(ds1,open('files/{}_train.json'.format(name), 'w'), indent=indent)
    json.dump(ds2,open('files/{}_valid.json'.format(name), 'w'), indent=indent)
    json.dump(ds3,open('files/{}_test.json'.format(name),  'w'), indent=indent)
    json.dump(dx, open('files/{}_text.json'.format(name),  'w'), indent=indent)

class coco (Dataset):

  def __init__(self, config, verbose = True, force = True):
    super().__init__ (config, verbose)

    name   = config['name']
    images = config['images_dir']
    trains = config['train_file']
    valids = config['valid_file']
    limit1 = config['train_limit']
    limit2 = config['valid_limit']
    split  = config['val_test_split']
    number_of_captions    = config['number_of_captions']

    self.get_data (name, images, trains, valids, limit1, limit2, split, force, number_of_captions, 2)

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

