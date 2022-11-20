from dataclasses import dataclass
import numpy as np
import os
from pathlib import Path
import random
import json
import codecs
import xml.etree.ElementTree as ET
from  glob import glob
from tqdm import tqdm
from typing import List
from typing import Tuple
from munch import DefaultMunch

from .training_sample import Box
from .training_sample import TrainingSample
from . import image
from cap.models import anchors

class Dataset:
  num_classes = 80
  class_index_to_name = {
  "1": "person",
  "2": "bicycle",
  "3": "car",
  "4": "motorcycle",
  "5": "airplane",
  "6": "bus",
  "7": "train",
  "8": "truck",
  "9": "boat",
  "10": "traffic light",
  "11": "fire hydrant",
  "13": "stop sign",
  "14": "parking meter",
  "15": "bench",
  "16": "bird",
  "17": "cat",
  "18": "dog",
  "19": "horse",
  "20": "sheep",
  "21": "cow",
  "22": "elephant",
  "23": "bear",
  "24": "zebra",
  "25": "giraffe",
  "27": "backpack",
  "28": "umbrella",
  "31": "handbag",
  "32": "tie",
  "33": "suitcase",
  "34": "frisbee",
  "35": "skis",
  "36": "snowboard",
  "37": "sports ball",
  "38": "kite",
  "39": "baseball bat",
  "40": "baseball glove",
  "41": "skateboard",
  "42": "surfboard",
  "43": "tennis racket",
  "44": "bottle",
  "46": "wine glass",
  "47": "cup",
  "48": "fork",
  "49": "knife",
  "50": "spoon",
  "51": "bowl",
  "52": "banana",
  "53": "apple",
  "54": "sandwich",
  "55": "orange",
  "56": "broccoli",
  "57": "carrot",
  "58": "hot dog",
  "59": "pizza",
  "60": "donut",
  "61": "cake",
  "62": "chair",
  "63": "couch",
  "64": "potted plant",
  "65": "bed",
  "67": "dining table",
  "70": "toilet",
  "72": "tv",
  "73": "laptop",
  "74": "mouse",
  "75": "remote",
  "76": "keyboard",
  "77": "cell phone",
  "78": "microwave",
  "79": "oven",
  "80": "toaster",
  "81": "sink",
  "82": "refrigerator",
  "84": "book",
  "85": "clock",
  "86": "vase",
  "87": "scissors",
  "88": "teddy bear",
  "89": "hair drier",
  "90": "toothbrush"
  }

  def __init__(self, split, dir = "datasets/NIA", feature_pixels = 16, augment = True, shuffle = True, allow_difficult = False, cache = True):
    if not os.path.exists(dir):
      raise FileNotFoundError("Dataset directory does not exist: %s" % dir)
    self.split = split
    self._dir  = dir
    self.class_index_to_name = self._get_classes()
    self.class_name_to_index = { class_name: class_index for (class_index, class_name) in self.class_index_to_name.items() }
    self.num_classes = len(self.class_index_to_name)
    assert self.num_classes == Dataset.num_classes, "Dataset does not have the expected number of classes (found %d but expected %d)" % (self.num_classes, Dataset.num_classes)
    assert self.class_index_to_name == Dataset.class_index_to_name, "Dataset does not have the expected class mapping"
    self._filepaths = self._get_filepaths()
    self.num_samples = len(self._filepaths)
    self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths = self._filepaths, allow_difficult = allow_difficult) 
    self._i = 0
    self._iterable_filepaths = self._filepaths.copy()
    self._feature_pixels = feature_pixels
    self._augment = augment
    self._shuffle = shuffle
    self._cache = cache
    self._unaugmented_cached_sample_by_filepath = {}
    self._augmented_cached_sample_by_filepath = {}

  def __iter__(self):
    self._i = 0
    if self._shuffle:
      random.shuffle(self._iterable_filepaths)
    return self

  def __next__(self):
    if self._i >= len(self._iterable_filepaths):
      raise StopIteration

    # Next file to load
    filepath = self._iterable_filepaths[self._i]
    self._i += 1

    # Augment?
    flip = random.randint(0, 1) != 0 if self._augment else 0
    cached_sample_by_filepath = self._augmented_cached_sample_by_filepath if flip else self._unaugmented_cached_sample_by_filepath
  
    # Load and, if caching, write back to cache
    if filepath in cached_sample_by_filepath:
      sample = cached_sample_by_filepath[filepath]
    else:
      sample = self._generate_training_sample(filepath = filepath, flip = flip)
    if self._cache:
      cached_sample_by_filepath[filepath] = sample

    # Return the sample
    return sample

  def _generate_training_sample(self, filepath, flip):
    # Load and preprocess the image
    prefix = filepath.split('_')[1]
    filepath = os.path.join(self._dir, prefix, filepath)
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape

    # Scale ground truth boxes to new image size
    scaled_gt_boxes = []
    #print (list(self._gt_boxes_by_filepath.keys())[0], filepath)
    for box in self._gt_boxes_by_filepath[filepath]:
      if flip:
        corners = np.array([
          box.corners[0],
          original_width - 1 - box.corners[3],
          box.corners[2],
          original_width - 1 - box.corners[1]
        ]) 
      else:
        corners = box.corners
      scaled_box = Box(
        class_index = box.class_index,
        class_name = box.class_name,
        corners = corners * scale_factor 
      )
      scaled_gt_boxes.append(scaled_box)

    # Generate anchor maps and RPN truth map
    anchor_map, anchor_valid_map = anchors.generate_anchor_maps(image_shape = scaled_image_data.shape, feature_pixels = self._feature_pixels)
    gt_rpn_map, gt_rpn_object_indices, gt_rpn_background_indices = anchors.generate_rpn_map(anchor_map = anchor_map, anchor_valid_map = anchor_valid_map, gt_boxes = scaled_gt_boxes)

    # Return sample
    return TrainingSample(
      anchor_map = anchor_map,
      anchor_valid_map = anchor_valid_map,
      gt_rpn_map = gt_rpn_map,
      gt_rpn_object_indices = gt_rpn_object_indices,
      gt_rpn_background_indices = gt_rpn_background_indices,
      gt_boxes = scaled_gt_boxes,
      image_data = scaled_image_data,
      image = scaled_image,
      filepath = filepath
    )

  def _get_classes(self):
    print ('_get_classes: ', self._dir + '/annotations/ImageSets/class_id_to_name.json')
    object = json.load(codecs.open(self._dir + '/annotations/ImageSets/class_id_to_name.json', 'r', 'utf-8-sig'))
    return object

  def _get_filepaths(self):
    #image_paths = [Path(f).stem for f in tqdm(glob(self._dir + '/images/*.jpg', recursive=True), desc='_get_filepaths') ]
    output = os.path.join(self._dir, 'annotations', 'ImageSets')
    image_paths = json.load(codecs.open(output + '/' + self.split + '.json', 'r', 'utf-8-sig'))
    #image_paths = [f for f in tqdm(glob(self._dir + '/images/*.jpg', recursive=True), desc='_get_filepaths') ]
    return image_paths

  def _get_ground_truth_boxes(self, filepaths, allow_difficult):
    gt_boxes_by_filepath = {}
    object = json.load(codecs.open(self._dir + '/annotations/ImageSets/bound_box.json', 'r', 'utf-8-sig'))
    o = DefaultMunch.fromDict(object)
    for k, v in tqdm(o.items(), desc='_get_ground_truth_boxes'):
      prefix = k.split('_')[1]
      f = self._dir + '/' + prefix + '/' + k
      if not os.path.exists(f):
        print ('File {} not found!'.format(f))
        continue

      boxes = []
      for c in v:
        x_min = int(c.bbox[0])
        y_min = int(c.bbox[1])
        x_max = int(c.bbox[2])+x_min
        y_max = int(c.bbox[3])+y_min
        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = c.id, class_name = c.name, corners = corners)
        boxes.append(box)
      gt_boxes_by_filepath[f] = boxes
    return gt_boxes_by_filepath 
