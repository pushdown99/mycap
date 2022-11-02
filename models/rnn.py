import tensorflow as tf
from os import path
from os.path import isfile

from keras.models import Model
from keras.layers import Input, Dense, LSTM, Embedding, Dropout, Concatenate
from keras.layers import RepeatVector, TimeDistributed, concatenate, Bidirectional

from .misc import get_dataset

from ..utils import plot

class rnn1 (tf.keras.Model):
  def __init__(self, config, data_opt, model_opt, verbose = True):
    super().__init__()

    self.verbose = verbose

    for dir in data_opt['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

    self.config    = config
    self.data_opt  = data_opt
    self.model_opt = model_opt
    self.data      = get_dataset(data_opt, model_opt, verbose)
    embedding_dim  = self.config['embedding_dim']
    units          = self.config['units']
    input_size     = self.model_opt['input_size']
    vocab_size     = self.data['vocab_size']
    max_length     = self.data['max_length']
 
    # feature extractor model
    inputs1 = Input(shape=(input_size,))
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(embedding_dim, activation='relu')(fe1)
    # sequence model
    inputs2 = Input(shape=(max_length,))
    se1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(units)(se2)
    # decoder model
    decoder1 = Concatenate()([fe2, se3])
    decoder2 = Dense(units, activation='relu')(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder2)
    # tie it together [image, seq] [word]
    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    # summarize model
    print(model.summary())

    filename = 'models/rnn_model_1_{}.png'.format(self.model_opt['model_name'])
    plot(model, filename)

    self.model = model
    self.model_name  = 'rnn1'

class rnn2 (tf.keras.Model):
  def __init__(self, config, data_opt, model_opt, verbose = True):
    super().__init__()

    self.verbose = verbose

    for dir in data_opt['dirs']:
      if not path.exists(dir):
        makedirs(dir, exist_ok=True)

    self.config    = config
    self.data_opt  = data_opt
    self.model_opt = model_opt
    self.data      = get_dataset(data_opt, model_opt, verbose)
    embedding_dim  = self.config['embedding_dim']
    units          = self.config['units']
    input_size     = self.model_opt['input_size']
    vocab_size     = self.data['vocab_size']
    max_length     = self.data['max_length']

    image_input     = Input(shape=(input_size,))
    image_model_1   = Dense(embedding_dim, activation='relu')(image_input)
    image_model     = RepeatVector(max_length)(image_model_1)

    caption_input   = Input(shape=(max_length,))
    # mask_zero: We zero pad inputs to the same length, the zero mask ignores those inputs
    caption_model_1 = Embedding(vocab_size, embedding_dim, mask_zero=True)(caption_input)
    # Since we are going to predict the next word using the previous words, we have to set return_sequences = True.
    caption_model_2 = LSTM(units, return_sequences=True)(caption_model_1)
    caption_model   = TimeDistributed(Dense(embedding_dim))(caption_model_2)

    # Merging the models and creating a softmax classifier
    final_model_1   = concatenate([image_model, caption_model])
    final_model_2   = Bidirectional(LSTM(units, return_sequences=False))(final_model_1)
    final_model_3   = Dense(units, activation='relu')(final_model_2)
    final_model     = Dense(vocab_size, activation='softmax')(final_model_3)

    model = Model(inputs=[image_input, caption_input], outputs=final_model)
    model.compile(loss='categorical_crossentropy', optimizer='adam')

    # summarize model
    print(model.summary())

    filename = 'models/rnn_model_1_{}.png'.format(self.model_opt['model_name'])
    plot(model, filename)

    self.model = model
    self.model_name  = 'rnn1'

