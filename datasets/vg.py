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
  num_classes = 150
  class_index_to_name = {
  "1": "airplane",
  "2": "animal",
  "3": "arm",
  "4": "background",
  "5": "bag",
  "6": "banana",
  "7": "bathroom",
  "8": "beach",
  "9": "bear",
  "10": "bed",
  "11": "bench",
  "12": "bike",
  "13": "bird",
  "14": "board",
  "15": "boat",
  "16": "book",
  "17": "bottle",
  "18": "bowl",
  "19": "box",
  "20": "boy",
  "21": "branch",
  "22": "building",
  "23": "bus",
  "24": "bush",
  "25": "cake",
  "26": "car",
  "27": "cat",
  "28": "chair",
  "29": "child",
  "30": "clock",
  "31": "cloud",
  "32": "couch",
  "33": "counter",
  "34": "cow",
  "35": "cup",
  "36": "desk",
  "37": "dirt",
  "38": "dog",
  "39": "donut",
  "40": "door",
  "41": "ear",
  "42": "elephant",
  "43": "eye",
  "44": "face",
  "45": "fence",
  "46": "field",
  "47": "floor",
  "48": "flower",
  "49": "food",
  "50": "foot",
  "51": "fork",
  "52": "giraffe",
  "53": "girl",
  "54": "glass",
  "55": "glove",
  "56": "grass",
  "57": "ground",
  "58": "hair",
  "59": "hand",
  "60": "handle",
  "61": "hat",
  "62": "head",
  "63": "helmet",
  "64": "hill",
  "65": "horse",
  "66": "house",
  "67": "jacket",
  "68": "jean",
  "69": "kite",
  "70": "lady",
  "71": "lamp",
  "72": "laptop",
  "73": "leaf",
  "74": "leg",
  "75": "letter",
  "76": "light",
  "77": "line",
  "78": "logo",
  "79": "man",
  "80": "mirror",
  "81": "motorcycle",
  "82": "mountain",
  "83": "neck",
  "84": "nose",
  "85": "number",
  "86": "ocean",
  "87": "orange",
  "88": "pant",
  "89": "paper",
  "90": "part",
  "91": "people",
  "92": "person",
  "93": "phone",
  "94": "picture",
  "95": "pillow",
  "96": "pizza",
  "97": "plane",
  "98": "plant",
  "99": "plate",
  "100": "player",
  "101": "pole",
  "102": "post",
  "103": "racket",
  "104": "reflection",
  "105": "road",
  "106": "rock",
  "107": "roof",
  "108": "sand",
  "109": "sandwich",
  "110": "shadow",
  "111": "sheep",
  "112": "shelf",
  "113": "shirt",
  "114": "shoe",
  "115": "short",
  "116": "sidewalk",
  "117": "sign",
  "118": "sink",
  "119": "skateboard",
  "120": "ski",
  "121": "skier",
  "122": "sky",
  "123": "snow",
  "124": "street",
  "125": "stripe",
  "126": "surfboard",
  "127": "surfer",
  "128": "table",
  "129": "tail",
  "130": "tile",
  "131": "tire",
  "132": "toilet",
  "133": "top",
  "134": "tower",
  "135": "track",
  "136": "train",
  "137": "tree",
  "138": "truck",
  "139": "trunk",
  "140": "umbrella",
  "141": "vase",
  "142": "vehicle",
  "143": "wall",
  "144": "water",
  "145": "wave",
  "146": "wheel",
  "147": "window",
  "148": "wing",
  "149": "woman",
  "150": "zebra"
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
    print ('Sample files: ', self.num_samples)
    self._gt_boxes_by_filepath = self._get_ground_truth_boxes(filepaths = self._filepaths, allow_difficult = allow_difficult) 
    print ('Bound boxes : ', len(self._gt_boxes_by_filepath))
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
    #filepath = os.path.join(self._dir, filepath)
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = os.path.join(self._dir, filepath), min_dimension_pixels = 600, horizontal_flip = flip)
    _, original_height, original_width = original_shape

    # Scale ground truth boxes to new image size
    scaled_gt_boxes = []

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
      #f = self._dir + '/' + k
      f = k
      if not os.path.exists(self._dir + '/' + f):
        print ('File {} not found!'.format(self._dir + '/' + f))
        continue

      boxes = []
      for c in v:
        x_min = int(c.bbox[0])
        y_min = int(c.bbox[1])
        x_max = int(c.bbox[2])
        y_max = int(c.bbox[3])
        corners = np.array([ y_min, x_min, y_max, x_max ]).astype(np.float32)
        box = Box(class_index = c.id, class_name = c.name, corners = corners)
        boxes.append(box)
      gt_boxes_by_filepath[f] = boxes
    return gt_boxes_by_filepath 
