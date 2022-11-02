#
# CNN Transformer in PyTorch and TensorFlow 2 w/ Keras
# tf2/ImageCaption/__main__.py
# Copyright 2022 Haeyeon, Hwang
#
# Main module for the TensorFlow/Keras implementation of Image Captioninh. Run this
# from the root directory, e.g.:
#
# python -m tf2.ImageCaption --help
#

#
# TODO
# ----
# -
#

import platform
from .config import Config, Flickr8kOpts, InceptionV3Opts

from .models    import Flickr8k
from .models    import InceptionV3
from .models    import Encoder
from .models    import Decoder
from .models    import BahdanauAttention
from . import utils

print('')
print('________                               _______________')
print('___  __/__________________________________  ____/__  /________      __')
print('__  /  _  _ \_  __ \_  ___/  __ \_  ___/_  /_   __  /_  __ \_ | /| / /')
print('_  /   /  __/  / / /(__  )/ /_/ /  /   _  __/   _  / / /_/ /_ |/ |/ / ')
print('/_/    \___//_/ /_//____/ \____//_/    /_/      /_/  \____/____/|__/  ')
print('')
print('----------------------------------------------------------------------')
print('')

import logging
logging.getLogger('tensorflow').disabled = True

import tensorflow as tf

if __name__ == "__main__":
  cuda_available 	= tf.test.is_built_with_cuda()
  gpu_available 	= len(tf.config.list_physical_devices('GPU'))

  print("Version        : %s" % (platform.version()))
  print("Python         : %s" % (platform.python_version()))
  print("Tensorflow     : %s" % (tf.__version__))
  print("CUDA Available : %s" % ("yes" if cuda_available else "no"))
  print("GPU Available  : %s" % ("yes" if gpu_available else "no"))
  print("Eager Execution: %s" % ("yes" if tf.executing_eagerly() else "no"))

#if utils.is_notebook():
#  import matplotlib.pyplot as plt
#else:
#  import plotext as plt
#
#plt.plot([1, 2, 3, 4], [1, 4, 9, 16])
#plt.plotsize(100, 30)
#plt.show()

dataset = Flickr8k.Dataset(Flickr8kOpts, True)
data  = dataset.LoadData ()

model = InceptionV3.CNN(Config, InceptionV3Opts, data)
#model.Fit ()
model.Evaluate ()
model.Generate (10)

