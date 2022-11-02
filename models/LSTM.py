import tensorflow as tf
import numpy as np
import os
import time
import glob
import random
import collections
import matplotlib.pyplot as plt

from tqdm import tqdm
from pickle import dump,load
from PIL import Image
from pathlib import Path

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB0
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import EarlyStopping
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
from tqdm.keras import TqdmCallback

from ..utils import plot, Display, History

class LSTM:
  def __init__ (self, config, verbose=True):
    self.config   = config
    self.verbose  = verbose
    #self.strategy = tf.distribute.MirroredStrategy()
    self.strategy = tf.distribute.MultiWorkerMirroredStrategy()
    print('Number of devices: {}'.format(self.strategy.num_replicas_in_sync))

  # preprocess the image for the model
  def preprocess_vgg16_image (self, filename, image_size = 224):
    print (filename)
    Display(filename)

    image = tf.keras.utils.load_img(filename, target_size=(image_size, image_size))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.vgg16.preprocess_input(image)

    plt.imshow(np.squeeze(image))
    return image

  def preprocess_inceptionv3_image (self, filename, image_size = 299):
    print (filename)
    Display(filename)

    image = tf.keras.utils.load_img(filename, target_size=(image_size, image_size))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.inception_v3.preprocess_input(image)

    plt.imshow(np.squeeze(image))
    return image

  def preprocess_efficientnetb0_image (self, filename, image_size = 299):
    print (filename)
    Display(filename)

    image = tf.keras.utils.load_img(filename, target_size=(image_size, image_size))
    image = tf.keras.utils.img_to_array(image)
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
    image = tf.keras.applications.efficientnet.preprocess_input(image)

    plt.imshow(np.squeeze(image))
    return image

  # extract features from each photo in the directory
  def extract_vgg16_features(self, directory, image_size = 224):
    # load the model
    model = VGG16()
    # re-structure the model
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    # summarize
    print(model.summary())
    plot(model, 'files/vgg16_model.png')
    # extract features from each photo
    features = dict()
    #for name in tqdm(os.listdir(directory)):
    for filename in tqdm(glob.glob(directory + '/*.jpg')):
      #if not filname.endswith(".jpg"):
      #  continue
      image = tf.keras.utils.load_img(filename, target_size=(image_size, image_size))
      # convert the image pixels to a numpy array
      image = tf.keras.utils.img_to_array(image)
      # reshape data for the model
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # prepare the image for the VGG model
      image = tf.keras.applications.vgg16.preprocess_input(image)
      # get features
      feature = model.predict(image, verbose=0)
      # get image id
      name = Path(filename).name
      image_id = name.split('.')[0]
      # store feature
      features[image_id] = feature
      #print('>%s' % name)
    return features

  def extract_inceptionv3_features(self, directory, image_size = 299):
    model = InceptionV3(weights='imagenet')
    model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
    print(model.summary())
    plot(model, 'files/inceptionv3_model.png')
    features = dict()
    for filename in tqdm(glob.glob(directory + '/*.jpg')):
      image = tf.keras.utils.load_img(filename, target_size=(image_size, image_size))
      image = tf.keras.utils.img_to_array(image)
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      image = tf.keras.applications.inception_v3.preprocess_input(image)
      feature = model.predict(image, verbose=0)
      name = Path(filename).name
      image_id = name.split('.')[0]
      features[image_id] = feature
    return features

  def extract_efficientnetb0_features(self, directory, image_size = 224):
    with self.strategy.scope():
      model = EfficientNetB0(weights='imagenet')
      model = Model(inputs=model.inputs, outputs=model.layers[-2].output)
      model.compile(loss='categorical_crossentropy', optimizer='adam')

    print(model.summary())
    plot(model, 'files/efficientnetb0_model.png')

    features = dict()
    for filename in tqdm(glob.glob(directory + '/*.jpg')):
      image = tf.keras.utils.load_img(filename, target_size=(image_size, image_size))
      image = tf.keras.utils.img_to_array(image)
      image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      image = tf.keras.applications.efficientnet.preprocess_input(image)

      feature = model.predict(image, verbose=0)
      name = Path(filename).name
      image_id = name.split('.')[0]
      features[image_id] = feature

    return features

  # load photo features
  def load_photo_features(self, filename, dataset):
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features

  # covert a dictionary of clean descriptions to a list of descriptions
  def to_lines(self, descriptions):
    all_desc = list()
    for key in descriptions.keys():
      [all_desc.append(d) for d in descriptions[key]]
    return all_desc

  # calculate the length of the description with the most words
  def max_length(self, descriptions):
    lines = self.to_lines(descriptions)
    return max(len(d.split()) for d in lines)

  # create sequences of images, input sequences and output words for an image
  def create_sequences(self, tokenizer, max_length, desc_list, photo, vocab_size):
    X1, X2, y = list(), list(), list()
    # walk through each description for the image
    for desc in desc_list:
      # encode the sequence
      seq = tokenizer.texts_to_sequences([desc])[0]
      # split one sequence into multiple X,y pairs
      for i in range(1, len(seq)):
        # split into input and output pair
        in_seq, out_seq = seq[:i], seq[i]
        # pad input sequence
        in_seq = tf.keras.utils.pad_sequences([in_seq], maxlen=max_length)[0]
        # encode output sequence
        out_seq = tf.keras.utils.to_categorical([out_seq], num_classes=vocab_size)[0]
        # store
        X1.append(photo)
        X2.append(in_seq)
        y.append(out_seq)
    return np.array(X1), np.array(X2), np.array(y)

  # define the captioning model
  def rnn_model1(self, vocab_size, max_length, embedding_size, units, input_size):
    # feature extractor model
    inputs1  = tf.keras.Input(shape=(input_size,))
    fe1      = tf.keras.layers.Dropout(0.5)(inputs1)
    fe2      = tf.keras.layers.Dense(embedding_size, activation='relu')(fe1)

    # sequence model
    inputs2  = tf.keras.Input(shape=(max_length,))
    se1      = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
    se2      = tf.keras.layers.Dropout(0.5)(se1)
    se3      = tf.keras.layers.LSTM(units)(se2)

    # decoder model
    decoder1 = tf.keras.layers.Concatenate()([fe2, se3])
    decoder2 = tf.keras.layers.Dense(units, activation='relu')(decoder1)
    outputs  = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    model    = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    plot(model, 'files/rnn_model1_{}_{}_{}.png'.format(embedding_size,units,input_size))
    return model

  # define the captioning model
  def rnn_model2(self, vocab_size, max_length, embedding_size, units, input_size):
    image_input     = tf.keras.Input(shape=(input_size,))
    image_model_1   = tf.keras.layers.Dense(embedding_size, activation='relu')(image_input)
    image_model     = tf.keras.layers.RepeatVector(max_length)(image_model_1)

    caption_input   = tf.keras.Input(shape=(max_length,))
    # mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs
    caption_model_1 = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)(caption_input)
    # Since we are going to predict the next word using the previous words, we have to set return_sequences = True.
    caption_model_2 = tf.keras.layers.LSTM(units, return_sequences=True)(caption_model_1)
    caption_model   = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(embedding_size))(caption_model_2)

    # Merging the models and creating a softmax classifier
    final_model_1   = tf.keras.layers.concatenate([image_model, caption_model])
    final_model_2   = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units, return_sequences=False))(final_model_1)
    final_model_3   = tf.keras.layers.Dense(units, activation='relu')(final_model_2)
    final_model     = tf.keras.layers.Dense(vocab_size, activation='softmax')(final_model_3)

    model = tf.keras.models.Model(inputs=[image_input, caption_input], outputs=final_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    plot(model, 'files/rnn_model2_{}_{}_{}.png'.format(embedding_size,units,input_size))
    return model

  # define the captioning model
  def define_model(self, vocab_size, max_length, embedding_size, units, input_size):
    # feature extractor model
    inputs1  = tf.keras.Input(shape=(input_size,))
    fe1      = tf.keras.layers.Dropout(0.5)(inputs1)
    fe2      = tf.keras.layers.Dense(embedding_size, activation='relu')(fe1)

    # sequence model
    inputs2  = tf.keras.Input(shape=(max_length,))
    se1      = tf.keras.layers.Embedding(vocab_size, embedding_size, mask_zero=True)(inputs2)
    se2      = tf.keras.layers.Dropout(0.5)(se1)
    se3      = tf.keras.layers.LSTM(units)(se2)

    # decoder model
    decoder1 = tf.keras.layers.Concatenate()([fe2, se3])
    decoder2 = tf.keras.layers.Dense(units, activation='relu')(decoder1)
    outputs  = tf.keras.layers.Dense(vocab_size, activation='softmax')(decoder2)

    model    = tf.keras.models.Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())
    plot(model, 'files/rnn_model.png')
    return model

  # data generator, intended to be used in a call to model.fit_generator()
  def data_generator(self, descriptions, photos, tokenizer, max_length, vocab_size):
    # loop for ever over images
    while 1:
      for key, desc_list in descriptions.items():
        # retrieve the photo feature
        photo = photos[key][0]
        in_img, in_seq, out_word = self.create_sequences(tokenizer, max_length, desc_list, photo, vocab_size)
        yield ([in_img, in_seq], out_word)

  def fit (self, num_of_epochs = 5):
    train = load(open('files/{}_train.pkl'.format(self.dataname), 'rb'))
    valid = load(open('files/{}_valid.pkl'.format(self.dataname), 'rb'))
    train_descriptions = load(open('files/{}_train_descriptions.pkl'.format(self.dataname), 'rb'))
    valid_descriptions = load(open('files/{}_valid_descriptions.pkl'.format(self.dataname), 'rb'))
    tokenizer = load(open('files/{}_tokenizer.pkl'.format(self.dataname), 'rb'))

    train_features = self.load_photo_features('files/rnn_vgg16_{}.pkl'.format(self.dataname), train)
    print('Photos: train=%d' % len(train_features))
    vocab_size = len(tokenizer.word_index) + 1
    print('Vocabulary Size: %d' % vocab_size)
    max_length = self.max_length(train_descriptions)
    print('Description Length: %d' % max_length)

    valid_features = self.load_photo_features('files/rnn_vgg16_{}.pkl'.format(self.dataname), valid)
    print('Photos: val=%d' % len(valid_features))

    # define the model
    BUFFER_SIZE = 10000
    BATCH_SIZE_PER_REPLICA = 256

    model_name      = self.config['model']
    embedding_size  = self.config['embedding_size']
    units           = self.config['units']
    input_size      = self.config['input_size']
    BATCH_SIZE      = BATCH_SIZE_PER_REPLICA * self.strategy.num_replicas_in_sync

    with self.strategy.scope():
      if model_name   == 'rnn_model1':
        model         = self.rnn_model1 (vocab_size, max_length, embedding_size, units, input_size)
      elif model_name == 'rnn_model2':
        model         = self.rnn_model2 (vocab_size, max_length, embedding_size, units, input_size)
      else:
        return

    train_steps     = len(train_descriptions)
    valid_steps     = len(valid_descriptions)

    generator       = self.data_generator(train_descriptions, train_features, tokenizer, max_length, vocab_size)
    valid_generator = self.data_generator(valid_descriptions, valid_features, tokenizer, max_length, vocab_size)

    filepath        = 'files/model_' + self.config['name'] + '_' + model_name + '_' + str(embedding_size) + '_' + str(units) + '_' + str(input_size) + '_' + self.dataname + '_ep{epoch:03d}_loss{loss:.3f}_val_loss{val_loss:.3f}.h5'
    checkpoint      = ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    early           = EarlyStopping(patience=1, verbose=1)

    #generator = generator.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    #valid_generator = valid_generator.batch(BATCH_SIZE)

    history = model.fit(generator, 
      epochs = num_of_epochs, 
      validation_data = valid_generator, 
      validation_steps = valid_steps, 
      steps_per_epoch = train_steps, 
      callbacks = [checkpoint, early],      
      #callbacks = [checkpoint, early, TqdmCallback(verbose=1)],      
      #batch_size = BATCH_SIZE,
      verbose = 1)

    History (history)
    #model.save('files/lstm_{}_{}_model_{}.h5'.format(self.config['name'], self.dataname, str(i))

  # generate a description for an image
  def generate_desc(self, model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = '<start>'
    # iterate over the whole length of the sequence
    for i in range(max_length):
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([in_text])[0]
      # pad input
      sequence = tf.keras.utils.pad_sequences([sequence], maxlen=max_length)
      # predict next word
      yhat = model.predict([photo, sequence], verbose=0)
      # convert probability to integer
      yhat = np.argmax(yhat)
      # map integer to word
      word = tokenizer.index_word[yhat]
      # stop if we cannot map the word
      if word is None:
        break
      # append as input for generating the next word
      in_text += ' ' + word
      # stop if we predict the end of the sequence
      if word == '<end>':
        break
    return in_text

  # calculate BLEU score
  def calculate_scores(self, actual, predicted):
    smooth = SmoothingFunction().method4
    bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0),           smoothing_function=smooth)*100
    bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0),         smoothing_function=smooth)*100
    bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0),       smoothing_function=smooth)*100
    bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)*100
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)

  # evaluate the skill of the model
  def evaluate_model(self, model, descriptions, features, tokenizer, max_length):
    actual, predicted = list(), list()
    # step over the whole set
    for key, desc_list in tqdm(descriptions.items(), position=0, leave=True):
      # generate description
      yhat = self.generate_desc(model, tokenizer, features[key], max_length)
      # store actual and predicted
      references = [d.split() for d in desc_list]
      actual.append(references)
      predicted.append(yhat.split())
    print('Sampling:')
    self.calculate_scores(actual, predicted)

  def evaluate(self, filename, max_length):
    test = load(open('files/{}_test.pkl'.format(self.dataname), 'rb'))
    test_descriptions = load(open('files/{}_test_descriptions.pkl'.format(self.dataname), 'rb'))
    tokenizer = load(open('files/{}_tokenizer.pkl'.format(self.dataname), 'rb'))

    test_features = self.load_photo_features('files/rnn_vgg16_{}.pkl'.format(self.dataname), test)
    print('Photos: test=%d' % len(test_features))

    # load the model
    model = tf.keras.models.load_model(filename)
    self.evaluate_model(model, test_descriptions, test_features, tokenizer, max_length)

class vgg16(LSTM):
  def __init__ (self, config, data, verbose=True):
    super().__init__(config, verbose)
    self.dataname = data['name']
    self.images   = data['images_dir']
    self.samsple  = data['example_image']

    print('#')
    print('# vgg16 model {}/{}'.format(self.dataname, self.images))
    print('#')

    self.preprocess_vgg16_image (self.samsple)
    filename = 'files/rnn_vgg16_{}.pkl'.format(self.dataname)

    if not os.path.isfile(filename):
      print('File not found: {}'.format(filename))
      features = dict()
      for images in (self.images):
        print (images)
        feature  = self.extract_vgg16_features(images)
        features.update(feature)
      print('Extracted Features: %d' % len(features))
      dump(features, open(filename, 'wb'))
    else:
      print('File found: {}'.format(filename))
      Display('files/vgg16_model.png')

class inceptionv3(LSTM):
  def __init__ (self, config, data, verbose=True):
    super().__init__(config, verbose)
    self.dataname = data['name']
    self.images   = data['images_dir']
    self.samsple  = data['example_image']

    print('#')
    print('# inceptionv3 model {}/{}'.format(self.dataname, self.images))
    print('#')

    self.preprocess_inceptionv3_image (self.samsple)
    filename = 'files/rnn_inceptionv3_{}.pkl'.format(self.dataname)
    
    if not os.path.isfile(filename):
      print('File not found: {}'.format(filename))
      features = dict()
      for images in (self.images):
        print (images)
        feature  = self.extract_inceptionv3_features(images)
        features.update(feature)
      print('Extracted Features: %d' % len(features))
      dump(features, open(filename, 'wb'))
    else:
      print('File found: {}'.format(filename))
      Display('files/vgg16_model.png')

class efficientnetb0 (LSTM):
  def __init__ (self, config, data, verbose=True):
    super().__init__(config, verbose)
    self.dataname = data['name']
    self.images   = data['images_dir']
    self.samsple  = data['example_image']

    print('#')
    print('# inceptionv3 model {}/{}'.format(self.dataname, self.images))
    print('#')

    self.preprocess_efficientnetb0_image (self.samsple)
    filename = 'files/rnn_efficientnetb0_{}.pkl'.format(self.dataname)
    
    if not os.path.isfile(filename):
      print('File not found: {}'.format(filename))
      features = dict()
      for images in (self.images):
        print (images)
        feature  = self.extract_efficientnetb0_features(images)
        features.update(feature)
      print('Extracted Features: %d' % len(features))
      dump(features, open(filename, 'wb'))
    else:
      print('File found: {}'.format(filename))
      Display('files/vgg16_model.png')



