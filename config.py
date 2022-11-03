parameters = {
}

coco2014_cfg = {
  'name':               'coco2014',
  'dataset_dir':        'datasets/COCO',
  'images_dir':        ['datasets/COCO/train2014', 'datasets/COCO/val2014' ],
  'train_file':         'datasets/COCO/annotations/captions_train2014.json',
  'valid_file':         'datasets/COCO/annotations/captions_val2014.json',
  'train_limit':        3000000,
  'valid_limit':        3000000,
  'val_test_split':     0.9,
  'number_of_captions': 5,
  'example_image':      'datasets/COCO/train2014/COCO_train2014_000000247789.jpg',
  'example_id':         'COCO_train2014_000000247789',
}

coco2017_cfg = {
  'name':               'coco2017',
  'dataset_dir':        'datasets/COCO',
  'images_dir':        ['datasets/COCO/train2017', 'datasets/COCO/val2017'],
  'train_file':         'datasets/COCO/annotations/captions_train2017.json',
  'valid_file':         'datasets/COCO/annotations/captions_val2017.json',
  'train_limit':        3000000,
  'valid_limit':        3000000,
  'val_test_split':     0.9,
  'number_of_captions': 5,
  'example_image':      'datasets/COCO/train2017/COCO_train2017_000000247789.jpg',
  'example_id':         'COCO_train2017_000000247789',
}


flickr8k_cfg = {
  'name':           'flickr8k',
  'dataset_dir':    'datasets/Flickr8k',
  'images_dir':    ['datasets/Flickr8k/Flicker8k_Dataset'],
  'caption_file':   'datasets/Flickr8k/Flickr8k.token.txt',
  'train_file':     'datasets/Flickr8k/Flickr_8k.trainImages.txt',
  'valid_file':     'datasets/Flickr8k/Flickr_8k.devImages.txt',
  'test_file':      'datasets/Flickr8k/Flickr_8k.testImages.txt',
  'train_limit':    3000000,
  'valid_limit':    3000000,
  'test_limit':     3000000,
  'val_test_split': 1.0,
  'example_image':  'datasets/Flickr8k/Flicker8k_Dataset/667626_18933d713e.jpg',
  'example_id':     '667626_18933d713e',
}

nia0403_cfg = {
  'name':           'nia0403',
  'dataset_dir':    'datasets/NIA/annotations',
  'images_dir':    ['datasets/NIA/images'],
  'caption_file':   'datasets/NIA/annotations/4-3.token.eng.txt',
  'train_file':     'datasets/NIA/annotations/4-3.trainImages.txt',
  'valid_file':     'datasets/NIA/annotations/4-3.devImages.txt',
  'test_file':      'datasets/NIA/annotations/4-3.testImages.txt',
  'train_limit':    3000000,
  'valid_limit':    10000,
  'test_limit':     10,
  'val_test_split': 1.0,
  'example_image':  'datasets/NIA/images/IMG_0250492_pan(pan).jpg',
  'example_id':     'IMG_0250492_pan(pan)',
}

nia0404_cfg = {
  'name':           'nia0404',
  'dataset_dir':    'datasets/NIA/annotations',
  'images_dir':    ['datasets/NIA/images'],
  'caption_file':   'datasets/NIA/annotations/4-4.token.eng.txt',
  'train_file':     'datasets/NIA/annotations/4-4.trainImages.txt',
  'valid_file':     'datasets/NIA/annotations/4-4.devImages.txt',
  'test_file':      'datasets/NIA/annotations/4-4.testImages.txt',
  'train_limit':    3000000,
  'valid_limit':    10000,
  'test_limit':     10,
  'val_test_split': 1.0,
  'example_image':  'datasets/NIA/images/IMG_0250492_pan(pan).jpg',
  'example_id':     'IMG_0250492_pan(pan)',
}

vgg16_cfg = {
  'name':                       'vgg16',
  'model':                      'rnn_model1',

  'verbose':                    True,
  'image_size':                 224,
  'num_of_epochs':              10,
  'batch_size':                 64,
  'buffer_size':                1000,
  'embedding_size':             256,
  'units':                      256,
  'input_size':                 4096,
  'top_k':                      5000,
  'features_shape':             512,
  'attention_features_shape':   64,
}

inceptionv3_cfg = {
  'name':                       'inceptionv3',
  'model':                      'rnn_model1',

  'verbose':                    True,
  'image_size':                 299,
  'num_of_epochs':              10,
  'batch_size':                 64,
  'buffer_size':                1000,
  'embedding_size':             256,
  'units':                      256,
  'input_size':                 4096,
  'top_k':                      5000,
  'features_shape':             512,
  'attention_features_shape':   64,
}

efficientb0_cfg = {
  'name':                       'efficientb0',
  'model':                      'rnn_model1',

  'verbose':                    True,
  'image_size':                 224,
  'num_of_epochs':              10,
  'batch_size':                 64,
  'buffer_size':                1000,
  'embedding_size':             256,
  'units':                      256,
  'input_size':                 4096,
  'top_k':                      5000,
  'features_shape':             512,
  'attention_features_shape':   64,

  'IMAGE_SHAPE':  			    (299,299),
  'MAX_VOCAB_SIZE':  			2000000,
  'SEQ_LENGTH':      			25,
  'BATCH_SIZE':      			64,
  'SHUFFLE_DIM':     			512,
  'EMBED_DIM':       			512,
  'FF_DIM':          			1024,
  'NUM_HEADS':       			6,
}

