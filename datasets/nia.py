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
  num_classes = 158
  class_index_to_name = {
  "1": "person",
  "2": "bicycle",
  "3": "car",
  "4": "motorcycle",
  "5": "scooter",
  "6": "bus",
  "7": "truck",
  "8": "traffic light",
  "9": "fire hydrant",
  "10": "fire extinguisher",
  "11": "sign",
  "12": "trash bin",
  "13": "bench",
  "14": "roof",
  "15": "bird",
  "16": "cat",
  "17": "dog",
  "18": "chicken",
  "19": "backpack",
  "20": "umbrella",
  "21": "handbag",
  "22": "tie",
  "23": "suitcase",
  "24": "muffler",
  "25": "hat",
  "26": "ball",
  "27": "poles",
  "28": "plate(skis)",
  "29": "board",
  "30": "drone",
  "31": "pilates equipment",
  "32": "treadmill",
  "33": "dumbbell",
  "34": "golf club",
  "35": "Billiards cue",
  "36": "skating shoes",
  "37": "tennis racket",
  "38": "badminton racket",
  "39": "goalpost",
  "40": "basketball hoop",
  "41": "carabiner",
  "42": "table tennis racket",
  "43": "rice cooker",
  "44": "gas stove",
  "45": "pot",
  "46": "pan",
  "47": "microwave",
  "48": "toaster",
  "49": "knives",
  "50": "chopping boards",
  "51": "ladle",
  "52": "silicon spatula",
  "53": "rice spatula",
  "54": "vegetable peeler",
  "55": "box grater",
  "56": "scissors",
  "57": "bowl",
  "58": "cutlery",
  "59": "plate",
  "60": "side dish",
  "61": "tray",
  "62": "mug",
  "63": "refrigerator",
  "64": "dish washer",
  "65": "espresso machine",
  "66": "purifier",
  "67": "banana",
  "68": "apple",
  "69": "grape",
  "70": "pear",
  "71": "melon",
  "72": "cucumber",
  "73": "watermelon",
  "74": "orange",
  "75": "peach",
  "76": "strawberry",
  "77": "plum",
  "78": "persimmon",
  "79": "lettuce",
  "80": "cabbage",
  "81": "radish",
  "82": "perilla leaf",
  "83": "garlic",
  "84": "onion",
  "85": "spring onion",
  "86": "carrot",
  "87": "corn",
  "88": "potato",
  "89": "sweet potato",
  "90": "egg plant",
  "91": "tomato",
  "92": "pumpkin",
  "93": "squash",
  "94": "chili",
  "95": "pimento",
  "96": "sandwich",
  "97": "hamburger",
  "98": "hotdog",
  "99": "pizza",
  "100": "donut",
  "101": "cake",
  "102": "white bread",
  "103": "icecream",
  "104": "ttoke",
  "105": "tteokbokki",
  "106": "kimchi",
  "107": "gimbap",
  "108": "sushi",
  "109": "mandu",
  "110": "gonggibap",
  "111": "couch",
  "112": "mirror",
  "113": "window",
  "114": "table",
  "115": "lamp",
  "116": "door",
  "117": "chair",
  "118": "bed",
  "119": "toilet bowl",
  "120": "washstand",
  "121": "book",
  "122": "clock",
  "123": "doll",
  "124": "hair drier",
  "125": "toothbrush",
  "126": "hair brush",
  "127": "tv",
  "128": "laptop",
  "129": "mouse",
  "130": "keyboard",
  "131": "cell phone",
  "132": "watch",
  "133": "camera",
  "134": "speaker",
  "135": "fan",
  "136": "air conditioner",
  "137": "piano",
  "138": "Tambourine",
  "139": "Castanets",
  "140": "guitar",
  "141": "violin",
  "142": "flute",
  "143": "recorder",
  "144": "xylophone",
  "145": "ocarina",
  "146": "thermometer",
  "147": "sphygmomanometer",
  "148": "blood glucose meter",
  "149": "defibrillator",
  "150": "massage gun",
  "151": "ceiling",
  "152": "floor",
  "153": "wall",
  "154": "pillar",
  "156": "road",
  "158": "tree",
  "160": "building",
  "162": "background",
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
    filepath = os.path.join(self._dir, 'images', filepath)
    scaled_image_data, scaled_image, scale_factor, original_shape = image.load_image(url = filepath, min_dimension_pixels = 600, horizontal_flip = flip)
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
      f = self._dir + '/images/' + k
      if not os.path.exists(f):
        #print ('File {} not found!'.format(f))
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
