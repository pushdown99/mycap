import tensorflow as tf
import numpy as np
import os
import time
import random
import collections
import matplotlib.pyplot as plt

from tqdm import tqdm
from pickle import dump,load
from PIL import Image

from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.efficientnet import EfficientNetB0
from keras.models import Model

from ..utils import plot, Display

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

class Attention:
  def __init__ (self, config, verbose=True):
    self.config  = config
    self.verbose = verbose

  def preprocessing(self):
    self.image_features_extract_model.summary()
    plot(self.image_features_extract_model, 'files/model_{}.png'.format(self.config['name']))

  def load_image(self, image_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (self.config['image_size'], self.config['image_size']))
    if self.config['name'] == 'vgg16':
      img = tf.keras.applications.vgg16.preprocess_input(img)
    else:
      img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path

  # Find the maximum length of any caption in our dataset
  def calc_max_length(self, tensor):
    return max(len(t) for t in tensor)

  # Load the numpy files
  def map_func(self, img_name, cap):
    #img_tensor = np.load(img_name.decode('utf-8')+'.npy')
    img_tensor = self.image_features[img_name.decode('utf-8')]
    return img_tensor, cap

  def fit(self):
    # Choose the top 5000 words from the vocabulary
    top_k = self.config['top_k']
    self.tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')
    self.tokenizer.fit_on_texts(self.train_captions)
    train_seqs = self.tokenizer.texts_to_sequences(self.train_captions)

    self.tokenizer.word_index['<pad>'] = 0
    self.tokenizer.index_word[0] = '<pad>'

    # Create the tokenized vectors
    train_seqs = self.tokenizer.texts_to_sequences(self.train_captions)

    # Pad each vector to the max_length of the captions
    # If you do not provide a max_length value, pad_sequences calculates it automatically
    cap_vector = tf.keras.preprocessing.sequence.pad_sequences(train_seqs, padding='post')

    # Calculates the max_length, which is used to store the attention weights
    self.max_length = self.calc_max_length(train_seqs)

    img_to_cap_vector = collections.defaultdict(list)
    for img, cap in zip(self.img_name_vector, cap_vector):
      img_to_cap_vector[img].append(cap)

    # Create training and validation sets using an 80-20 split randomly.
    img_keys = list(img_to_cap_vector.keys())
    random.shuffle(img_keys)

    slice_index = int(len(img_keys)*0.8)
    img_name_train_keys, img_name_val_keys = img_keys[:slice_index], img_keys[slice_index:]

    self.img_name_train = []
    self.cap_train = []
    for imgt in img_name_train_keys:
      capt_len = len(img_to_cap_vector[imgt])
      self.img_name_train.extend([imgt] * capt_len)
      self.cap_train.extend(img_to_cap_vector[imgt])

    self.img_name_val = []
    self.cap_val = []
    for imgv in img_name_val_keys:
      capv_len = len(img_to_cap_vector[imgv])
      self.img_name_val.extend([imgv] * capv_len)
      self.cap_val.extend(img_to_cap_vector[imgv])

    if self.verbose:
      print ('img_name_train: ',len(self.img_name_train))
      print ('cap_train     : ',len(self.cap_train))
      print ('img_name_val  : ',len(self.img_name_val))
      print ('cap_val       : ',len(self.cap_val))

    # Feel free to change these parameters according to your system's configuration

    BATCH_SIZE    = self.config['batch_size']
    BUFFER_SIZE   = self.config['buffer_size']
    embedding_dim = self.config['embedding_dim']
    units         = self.config['units']
    vocab_size    = top_k + 1
    num_steps     = len(self.img_name_train) // BATCH_SIZE
    # Shape of the vector extracted from InceptionV3 is (64, 2048)
    # These two variables represent that vector shape
    self.features_shape           = self.config['features_shape']
    self.attention_features_shape = self.config['attention_features_shape']

    dataset = tf.data.Dataset.from_tensor_slices((self.img_name_train, self.cap_train))

    # Use map to load the numpy files in parallel
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(
          self.map_func, [item1, item2], [tf.float32, tf.int32]),
          num_parallel_calls=tf.data.experimental.AUTOTUNE)

    # Shuffle and batch
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    self.encoder = CNN_Encoder(embedding_dim)
    self.decoder = RNN_Decoder(embedding_dim, units, vocab_size)

    self.optimizer = tf.keras.optimizers.Adam()
    self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    checkpoint_path = "checkpoints/train"
    ckpt = tf.train.Checkpoint(encoder=self.encoder, decoder=self.decoder, optimizer = self.optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

    start_epoch = 0
    if ckpt_manager.latest_checkpoint:
      start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
      # restoring the latest checkpoint in checkpoint_path
      ckpt.restore(ckpt_manager.latest_checkpoint)

    # adding this in a separate cell because if you run the training cell
    # many times, the loss_plot array will be reset
    loss_plot = []

    EPOCHS = self.config['num_of_epochs']

    for epoch in range(start_epoch, EPOCHS):
      start = time.time()
      total_loss = 0

      for (batch, (img_tensor, target)) in enumerate(dataset):
        print (tf.shape(img_tensor).numpy())
        batch_loss, t_loss = self.train_step(img_tensor, target)
        total_loss += t_loss

        if batch % 100 == 0:
            print ('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1, batch, batch_loss.numpy() / int(target.shape[1])))

      # storing the epoch end loss value to plot later
      loss_plot.append(total_loss / num_steps)

      if epoch % 5 == 0:
        ckpt_manager.save()

      print ('Epoch {} Loss {:.6f}'.format(epoch + 1, total_loss/num_steps))
      print ('Time taken for 1 epoch {} sec\n'.format(time.time() - start))

    plt.plot(loss_plot)
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss Plot')
    plt.show()

  @tf.function
  def train_step(self, img_tensor, target):
    loss = 0

    # initializing the hidden state for each batch
    # because the captions are not related from image to image
    hidden = self.decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
      features = self.encoder(img_tensor)

      for i in range(1, target.shape[1]):
        # passing the features through the decoder
        predictions, hidden, _ = self.decoder(dec_input, features, hidden)
        loss += self.loss_function(target[:, i], predictions)

        # using teacher forcing
        dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = self.encoder.trainable_variables + self.decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

  def loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = self.loss_object(real, pred)

    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask

    return tf.reduce_mean(loss_)

  def evaluate(self, image):
    attention_plot = np.zeros((self.max_length, self.attention_features_shape))

    hidden = self.decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(self.load_image(image)[0], 0)
    img_tensor_val = self.image_features_extract_model(temp_input)
    if self.config['name'] == 'vgg16':
      img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, 2048))
    else:
      img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0], -1, img_tensor_val.shape[3]))

    features = self.encoder(img_tensor_val)

    dec_input = tf.expand_dims([self.tokenizer.word_index['<start>']], 0)
    result = []

    for i in range(self.max_length):
      predictions, hidden, attention_weights = self.decoder(dec_input, features, hidden)

      attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

      predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
      result.append(self.tokenizer.index_word[predicted_id])

      if self.tokenizer.index_word[predicted_id] == '<end>':
        return result, attention_plot

      dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot

  def plot_attention(self, image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(10, 10))

    len_result = len(result)
    for l in range(len_result):
      temp_att = np.resize(attention_plot[l], (8, 8))
      ax = fig.add_subplot(len_result//2, len_result//2, l+1)
      ax.set_title(result[l])
      img = ax.imshow(temp_image)
      ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

  def validation(self):
    # captions on the validation set
    rid = np.random.randint(0, len(self.img_name_val))
    image = self.img_name_val[rid]
    real_caption = ' '.join([self.tokenizer.index_word[i] for i in self.cap_val[rid] if i not in [0]])
    result, attention_plot = self.evaluate(image)

    print ('Real Caption:', real_caption)
    print ('Prediction Caption:', ' '.join(result))
    self.plot_attention(image, result, attention_plot)

  def prediction(self):
    image_url = 'https://tensorflow.org/images/surf.jpg'
    image_extension = image_url[-4:]
    image_path = tf.keras.utils.get_file('image'+image_extension, origin=image_url)

    result, attention_plot = self.evaluate(image_path)
    print ('Prediction Caption:', ' '.join(result))
    self.plot_attention(image_path, result, attention_plot)
    # opening the image
    Display(image_path)

  def put_data(self, data):
    self.train_captions  = data['train_captions']
    self.img_name_vector = data['img_name_vector']

    features_file = 'files/{}_{}_features.pkl'.format(self.config['name'], data['name'])

    self.image_features = dict()
    if not os.path.isfile(features_file):
      # Get unique images
      encode_train = sorted(set(self.img_name_vector))

      # Feel free to change batch_size according to your system configuration
      image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
      image_dataset = image_dataset.map(
        self.load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(16)

      for img, path in tqdm(image_dataset):
        batch_features = self.image_features_extract_model(img)
        print (tf.shape(batch_features).numpy())
        if self.config['name'] == 'vgg16':
          batch_features = tf.reshape(batch_features, (batch_features.shape[0], 64, -1))
        elif self.config['name'] == 'inceptionv3':
          batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        else:
          batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, 2048))
          #batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        print (tf.shape(batch_features).numpy())
  
        for bf, p in zip(batch_features, path):
          path_of_feature = p.numpy().decode("utf-8")
          self.image_features[path_of_feature] = bf.numpy()
      dump(self.image_features, open(features_file, 'wb'))
    else:
      self.image_features = load(open(features_file, 'rb'))

class vgg16(Attention):
  def __init__ (self, config, verbose=True):
    super().__init__(config, verbose)

    image_model = VGG16()
    new_input = image_model.input
    hidden_layer = image_model.layers[-2].output
    self.image_features_extract_model =  tf.keras.Model(new_input, hidden_layer)

    self.preprocessing()

class inceptionv3(Attention):
  def __init__ (self, config, verbose=True):
    super().__init__(config, verbose)

    image_model = InceptionV3(include_top=False, weights='imagenet')
    new_input = image_model.input
    hidden_layer = image_model.layers[-1].output
    self.image_features_extract_model =  tf.keras.Model(new_input, hidden_layer)

    self.preprocessing()

class efficientb0(Attention):
  def __init__ (self, config, verbose=True):
    super().__init__(config, verbose)

    #image_model = EfficientNetB0(input_shape=(*(299, 299), 3), include_top=False, weights='imagenet')
    image_model = EfficientNetB0(include_top=False, weights='imagenet')
    #image_model.trainable = False
    new_input = image_model.input
    hidden_layer = image_model.output
    #hidden_layer = image_model.layers[-1].output
    self.image_features_extract_model =  tf.keras.Model(new_input, hidden_layer)

    self.preprocessing()


