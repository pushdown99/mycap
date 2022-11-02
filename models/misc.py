import tensorflow as tf
import random

from os import listdir
from tqdm import tqdm

import numpy as np
from keras.utils import pad_sequences, to_categorical
from keras.utils import plot_model, load_img, img_to_array
from keras.applications import vgg16, inception_v3

# Evaluate Model
from numpy import argmax, argsort
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction

# Generate Captions
from IPython.display import Image, display

from . import flickr8k, coco2014

# Create sequences of images, input sequences and output words for an image
def create_sequences(tokenizer, max_length, captions_list, image):
  # X1 : input for image features
  # X2 : input for text features
  # y  : output word
  X1, X2, y = list(), list(), list()
  vocab_size = len(tokenizer.word_index) + 1
  # Walk through each caption for the image
  for caption in captions_list:
    # Encode the sequence
    seq = tokenizer.texts_to_sequences([caption])[0]
    # Split one sequence into multiple X,y pairs
    for i in range(1, len(seq)):
      # Split into input and output pair
      in_seq, out_seq = seq[:i], seq[i]
      # Pad input sequence
      in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
      # Encode output sequence
      out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
      # Store
      X1.append(image)
      X2.append(in_seq)
      y.append(out_seq)
  return X1, X2, y

# Data generator, intended to be used in a call to model.fit()
def data_generator(images, captions, tokenizer, max_length, batch_size, random_seed):
  # Setting random seed for reproducibility of results
  random.seed(random_seed)
  # Image ids
  image_ids = list(captions.keys())
  _count=0
  while True:
    if _count >= len(image_ids):
      # Generator exceeded or reached the end so restart it
      _count = 0
    # Batch list to store data
    input_img_batch, input_sequence_batch, output_word_batch = list(), list(), list()
    for i in range(_count, min(len(image_ids), _count+batch_size)):
      # Retrieve the image id
      image_id = image_ids[i]
      # Retrieve the image features
      image = images[image_id][0]
      # Retrieve the captions list
      captions_list = captions[image_id]
      # Shuffle captions list
      random.shuffle(captions_list)
      input_img, input_sequence, output_word = create_sequences(tokenizer, max_length, captions_list, image)
      # Add to batch
      for j in range(len(input_img)):
        input_img_batch.append(input_img[j])
        input_sequence_batch.append(input_sequence[j])
        output_word_batch.append(output_word[j])
    _count = _count + batch_size
    yield ([np.array(input_img_batch), np.array(input_sequence_batch)], np.array(output_word_batch))

# generate a description for an image
def generate_desc(model, tokenizer, photo, max_length):
  # seed the generation process
  in_text = '<start>'
  # iterate over the whole length of the sequence
  for i in range(max_length):
    # integer encode input sequence
    sequence = tokenizer.texts_to_sequences([in_text])[0]
    # pad input
    sequence = pad_sequences([sequence], maxlen=max_length)
    # predict next word
    yhat = model.predict([photo,sequence], verbose=0)
    # convert probability to integer
    yhat = argmax(yhat)
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

# generate a description for an image using beam search
def generate_desc_beam_search(model, tokenizer, photo, max_length, beam_index=3):
  # seed the generation process
  in_text = [['<start>', 0.0]]
  # iterate over the whole length of the sequence
  for i in range(max_length):
    temp = []
    for s in in_text:
      # integer encode input sequence
      sequence = tokenizer.texts_to_sequences([s[0]])[0]
      # pad input
      sequence = pad_sequences([sequence], maxlen=max_length)
      # predict next words
      preds = model.predict([photo,sequence], verbose=0)
      word_preds = argsort(preds[0])[-beam_index:]
      # get top predictions
      for w in word_preds:
        next_cap, prob = s[0][:], s[1]
        # map integer to word
        word = tokenizer.index_word[w]
        next_cap += ' ' + word
        prob += preds[0][w]
        temp.append([next_cap, prob])

    in_text = temp
    # sorting according to the probabilities
    in_text = sorted(in_text, reverse=False, key=lambda l: l[1])
    # getting the top words
    in_text = in_text[-beam_index:]

  # get last (best) caption text
  in_text = in_text[-1][0]
  caption_list = []
  # remove leftover <end> 
  for w in in_text.split():
    caption_list.append(w)
    if w == '<end>':
      break
  # convert list to string
  caption = ' '.join(caption_list)
  return caption

def calculate_scores(actual, predicted):
  # calculate BLEU score
  smooth = SmoothingFunction().method4
  bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
  bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth)*100
  bleu3 = corpus_bleu(actual, predicted, weights=(0.3, 0.3, 0.3, 0), smoothing_function=smooth)*100
  bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth)*100
  print('  BLEU-1: %f' % bleu1)
  print('  BLEU-2: %f' % bleu2)
  print('  BLEU-3: %f' % bleu3)
  print('  BLEU-4: %f' % bleu4)

# evaluate the skill of the model
def evaluate_model(model, descriptions, features, tokenizer, max_length):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in tqdm(descriptions.items(), position=0, leave=True):
    # generate description
    yhat = generate_desc(model, tokenizer, features[key], max_length)
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat.split())
  print('  Sampling:')
  calculate_scores(actual, predicted)

# evaluate the skill of the model
def evaluate_model_beam_search(model, descriptions, features, tokenizer, max_length, beam_index=3):
  actual, predicted = list(), list()
  # step over the whole set
  for key, desc_list in tqdm(descriptions.items(), position=0, leave=True):
    # generate description beam search
    yhat = generate_desc_beam_search(model, tokenizer, features[key], max_length, beam_index)
    # store actual and predicted
    references = [d.split() for d in desc_list]
    actual.append(references)
    predicted.append(yhat.split())
  print('  Beam Search k=%d:' % beam_index)
  calculate_scores(actual, predicted)

def clean_caption(caption):
  # split caption words
  caption_list = caption.split()
  # remove <start> and <end>
  caption_list = caption_list[1:len(caption_list)-1]
  # convert list to string
  caption = ' '.join(caption_list)
  return caption

def generate_captions(model, directory, descriptions, features, tokenizer, max_length, image_size, count):
  c = 0
  for key, desc_list in descriptions.items():
    # load an image from file
    filename = directory + '/' + key + '.jpg'
    #diplay image
    display(Image(filename))
    # print original descriptions
    for i, desc in enumerate(desc_list):
      print('  Original ' + str(i+1) + ': ' + clean_caption(desc_list[i]))
    # generate descriptions
    desc = generate_desc(model, tokenizer, features[key], max_length)
    desc_beam_3 = generate_desc_beam_search(model, tokenizer, features[key], max_length, beam_index=3)
    desc_beam_5 = generate_desc_beam_search(model, tokenizer, features[key], max_length, beam_index=5)
    # calculate BLEU-1 scores
    references = [d.split() for d in desc_list]
    smooth = SmoothingFunction().method4
    desc_bleu = sentence_bleu(references, desc.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
    desc_beam_3_bleu = sentence_bleu(references, desc_beam_3.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
    desc_beam_5_bleu = sentence_bleu(references, desc_beam_5.split(), weights=(1.0, 0, 0, 0), smoothing_function=smooth)*100
    # print descriptions with scores
    print('  Sampling (BLEU-1: %f): %s' % (desc_bleu, clean_caption(desc)))
    print('  Beam Search k=3 (BLEU-1: %f): %s' % (desc_beam_3_bleu, clean_caption(desc_beam_3)))
    print('  Beam Search k=5 (BLEU-1: %f): %s' % (desc_beam_5_bleu, clean_caption(desc_beam_5)))
    c += 1
    if c == count:
      break

def get_dataset(data_opt, model_opt, verbose):
  if data_opt['dataset_name'] == 'flickr8k':
    dataset = flickr8k.Dataset(data_opt, model_opt, verbose)
  elif data_opt['dataset_name'] == 'coco2014':
    dataset = coco2014.Dataset(data_opt, model_opt, verbose)
  data    = dataset.load_data ()
  return data

def preprocess_vgg16_image (filename, image_size):
  image = load_img(filename, target_size=(image_size, image_size))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = vgg16.preprocess_input(image)
  return image

def preprocess_inception_v3_image (filename, image_size):
  image = load_img(filename, target_size=(image_size, image_size))
  image = img_to_array(image)
  image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
  image = inception_v3.preprocess_input(image)
  return image

def extract_vgg16_features (directory, model, image_size, extension = ".jpg"):
  features = dict()

  for name in tqdm(listdir(directory), position=0, leave=True):
    if not name.endswith(extension):
      continue

    filename = directory + '/' + name
    image = preprocess_vgg16_image(filename, image_size)
    feature = model.predict(image, verbose=0)
    image_id = name.split('.')[0]
    features[image_id] = feature
  return features

def extract_inception_v3_features (directory, model, image_size, extension = ".jpg"):
  features = dict()

  for name in tqdm(listdir(directory), position=0, leave=True):
    if not name.endswith(extension):
      continue

    filename = directory + '/' + name
    image = preprocess_inception_v3_image(filename, image_size)
    feature = model.predict(image, verbose=0)
    image_id = name.split('.')[0]
    features[image_id] = feature
  return features
