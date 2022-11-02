from os import path
from os.path import isfile
import random

import numpy as np
import matplotlib.pyplot as plt

from os import listdir
from pickle import dump,load
from keras.applications.vgg16 import VGG16
from keras.models import Model, load_model
from keras.callbacks import ModelCheckpoint, EarlyStopping

from IPython.display import Image, display

from .rnn  import rnn1, rnn2
from .misc import get_dataset, preprocess_vgg16_image, extract_vgg16_features
from .misc import data_generator, evaluate_model, evaluate_model_beam_search, generate_captions
from ..utils import plot

class model1 (rnn1):
  def __init__(self, config, data_opt, model_opt, verbose = True):
    vgg16_features (vgg16_model(), config, data_opt, model_opt)

    super().__init__(config, data_opt, model_opt)

  def Fit(self):
    vgg16_fit (self.model, self.model_name, self.config, self.data_opt, self.model_opt, self.data)

  def Evaluate (self):
    vgg16_evaluate_model (self.model_name, self.config, self.data_opt, self.model_opt, self.data)

  def EvaluateBeamSearch (self, beam_index=3):
    vgg16_evaluate_model_beam_search (self.model_name, self.config, self.data_opt, self.model_opt, self.data, beam_index)

  def Generate(self):
    vgg16_generate (self.model_name, self.config, self.data_opt, self.model_opt, self.data)

class model2 (rnn2):
  def __init__(self, config, data_opt, model_opt, verbose = True):
    vgg16_features (vgg16_model(), config, data_opt, model_opt)

    super().__init__(config, data_opt, model_opt)

  def Fit(self):
    vgg16_fit (self.model, self.model_name, self.config, self.data_opt, self.model_opt, self.data)

  def Evaluate (self):
    vgg16_evaluate_model (self.model_name, self.config, self.data_opt, self.model_opt, self.data)

  def EvaluateBeamSearch (self, beam_index=3):
    vgg16_evaluate_model_beam_search (self.model_name, self.config, self.data_opt, self.model_opt, self.data, beam_index)

  def Generate(self):
    vgg16_generate (self.model_name, self.config, self.data_opt, self.model_opt, self.data)

######

def vgg16_model (verbose = True):
  model = VGG16()
  model = Model(inputs=model.inputs, outputs=model.layers[-2].output)

  if verbose:
    print(model.summary())

  return model

def vgg16_features (model, config, data_opt, model_opt):
  filename = 'models/{}.png'.format(model_opt['model_name'])
  plot(model, filename)

  filename = 'files/features_{}_vgg16.pkl'.format(data_opt['dataset_name']) 
  print (filename)
  # only extract if file does not exist
  if not isfile(filename):
    features = dict()
    # extract features from all images
    for directory in data_opt['images_dir']:
      print (directory)
      if path.exists(directory):
        dictionary  = extract_vgg16_features (directory, model, model_opt['image_size'])
        features.update(dictionary)
    # save to file
    dump(features, open(filename, 'wb'))

  example_image = data_opt['example_image']
  display(Image(example_image))
  image = preprocess_vgg16_image(example_image, model_opt['image_size'])
  plt.imshow(np.squeeze(image))

def vgg16_fit (model, model_name, config, data_opt, model_opt, data):
  num_of_epochs, batch_size, tokenizer, vocab_size, max_length = vgg16_get_data_for_generation (config, data)
  train_features, train_descriptions = vgg16_get_train_data_for_generation (data)
  valid_features, valid_descriptions = vgg16_get_valid_data_for_generation (data)
  steps_train, steps_val = vgg16_get_model_steps (train_descriptions, valid_descriptions, batch_size)

  random.seed('1000')
  ids_train = list(train_descriptions.keys())
  random.shuffle(ids_train)
  train_descriptions = {_id: train_descriptions[_id] for _id in ids_train}

  generator_train = data_generator(train_features, train_descriptions, tokenizer, max_length, batch_size, random_seed='1000')
  generator_val   = data_generator(valid_features, valid_descriptions, tokenizer, max_length, batch_size, random_seed='1000')

  filename = 'models/{}_{}_ep{}.h5'.format(model_name, model_opt['model_name'], num_of_epochs)
  checkpoint = ModelCheckpoint(filename, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
  early = EarlyStopping(patience=1, verbose=1)

  history = model.fit(generator_train,
    epochs           = num_of_epochs,
    steps_per_epoch  = steps_train,
    validation_data  = generator_val,
    validation_steps = steps_val,
    callbacks        = [checkpoint, early],
    verbose          = 1)

  vgg16_hist_plot (history, 'models/{}_{}_ep{}_loss.png'.format(model_name, model_opt['model_name'],num_of_epochs))

def vgg16_get_model_steps (train_descriptions, valid_descriptions, batch_size):
  train_length  = len(train_descriptions)
  val_length    = len(valid_descriptions)
  steps_train   = train_length // batch_size

  if train_length % batch_size != 0:
    steps_train = steps_train + 1

  steps_val = val_length // batch_size

  if val_length % batch_size != 0:
    steps_val   = steps_val + 1

  return steps_train, steps_val

def vgg16_get_train_data_for_generation (data):
  return data['train_features'], data['train_descriptions']

def vgg16_get_valid_data_for_generation (data):
  return data['valid_features'], data['valid_descriptions']

def vgg16_get_data_for_generation (config, data):
  return config['num_of_epochs'], config['batch_size'], data['tokenizer'], data['vocab_size'], data['max_length']

def vgg16_hist_plot (history, filename):
  for label in ["loss","val_loss"]:
    plt.plot(history.history[label],label=label)
  plt.legend()
  plt.xlabel("epochs")
  plt.ylabel("loss")
  plt.savefig(filename)

def vgg16_evaluate_model (model_name, config, data_opt, model_opt, data):
  num_of_epochs     = config['num_of_epochs']
  batch_size        = config['batch_size']
  test_descriptions = data['test_descriptions']
  test_features     = data['test_features']
  tokenizer         = data['tokenizer']
  max_length        = data['max_length']

  filename = 'models/{}_{}_ep{}.h5'.format(model_name, model_opt['model_name'], num_of_epochs)
  model = load_model (filename)
  evaluate_model (model, test_descriptions, test_features, tokenizer, max_length)

def vgg16_evaluate_model_beam_search (model_name, config, data_opt, model_opt, data, beam_index=3):
  num_of_epochs     = config['num_of_epochs']
  batch_size        = config['batch_size']
  test_descriptions = data['test_descriptions']
  test_features     = data['test_features']
  tokenizer         = data['tokenizer']
  max_length        = data['max_length']

  filename = 'models/{}_{}_ep{}.h5'.format(model_name, model_opt['model_name'], num_of_epochs)
  model = load_model(filename)
  evaluate_model_beam_search(model, test_descriptions, test_features, tokenizer, max_length, beam_index)


def vgg16_generate (model_name, config, data_opt, model_opt, data):
  num_of_epochs     = config['num_of_epochs']
  test_descriptions = data['test_descriptions']
  test_features     = data['test_features']
  max_length        = data['max_length']

  filename = 'files/tokenizer_{}_{}.pkl'.format(data_opt['dataset_name'],model_opt['model_name'])
  tokenizer = load(open(filename, 'rb'))

  filename = 'models/{}_{}_ep{}.h5'.format(model_name, model_opt['model_name'], num_of_epochs)
  model = load_model(filename)
  generate_captions (model, data_opt['tests_dir'], test_descriptions, test_features, tokenizer, max_length, model_opt['image_size'], 10)

