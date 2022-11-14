import numpy as np
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras import Model
from tensorflow.keras.layers import Lambda

from . import vgg16
from . import rpn
from . import detector
from . import math_utils


class FasterRCNNModel (tf.keras.Model):
  def __init__(self, num_classes, allow_edge_proposals, custom_roi_pool, activate_class_outputs, l2 = 0, dropout_probability = 0):
    super().__init__()

    print('FasterRCNNModel: ', num_classes, allow_edge_proposals, custom_roi_pool, activate_class_outputs, l2, dropout_probability)
    self.num_classes = num_classes
    self.activate_class_outputs = activate_class_outputs
    self.stage1_feature_extractor = vgg16.FeatureExtractor(l2 = l2)
    print ('self.stage1_feature_extractor: ', self.stage1_feature_extractor)
    self.stage2_region_proposal_network = rpn.RegionProposalNetwork(
      max_proposals_pre_nms_train  = 12000,
      max_proposals_post_nms_train = 2000,
      max_proposals_pre_nms_infer  = 6000,
      max_proposals_post_nms_infer = 300,
      l2                           = l2,
      allow_edge_proposals         = allow_edge_proposals
    )
    self.stage3_detector_network = detector.DetectorNetwork(
      num_classes            = num_classes,
      custom_roi_pool        = custom_roi_pool,
      activate_class_outputs = activate_class_outputs,
      l2                     = l2,
      dropout_probability    = dropout_probability
    )

  def call(self, inputs, training = False):
    print ('FasterRCNNModel:call 0', inputs[0])
    print ('FasterRCNNModel:call 1', inputs[1])
    print ('FasterRCNNModel:call 2', inputs[2])
    print ('FasterRCNNModel:call 3', inputs[3])
    print ('FasterRCNNModel:call 4', inputs[4])
    print ('FasterRCNNModel:call 5', inputs[5])
    input_image = inputs[0]             # (1, height_pixels, width_pixels, 3)
    anchor_map = inputs[1]              # (1, height, width, num_anchors * 4)
    anchor_valid_map = inputs[2]        # (1, height, width, num_anchors)
    if training:
      gt_rpn_map = inputs[3]            # (1, height, width, num_anchors, 6)
      gt_box_class_idxs_map = inputs[4] # (1, num_gt_boxes)
      gt_box_corners_map = inputs[5]    # (1, num_gt_boxes, 4)

    # Stage 1: Extract features
    print (input_image, training)
    feature_map = self.stage1_feature_extractor(input_image = input_image, training = training)
    print (feature_map)

    # Stage 2: Generate object proposals using RPN
    print ('feature_map.shape: ', len(feature_map.shape))
    rpn_scores, rpn_box_deltas, proposals = self.stage2_region_proposal_network(
      inputs = [
        input_image,
        feature_map,
        anchor_map,
        anchor_valid_map
      ],
      training = training
    )

    # If training, we must generate ground truth data for the detector stage
    # from RPN outputs
    if training:
      # Assign labels to proposals and take random sample (for detector training)
      proposals, gt_classes, gt_box_deltas = self.label_proposals(
        proposals = proposals,
        gt_box_class_idxs = gt_box_class_idxs_map[0], # for now, batch size of 1
        gt_box_corners = gt_box_corners_map[0],
        min_background_iou_threshold = 0.0,
        min_object_iou_threshold = 0.5
      )
      proposals, gt_classes, gt_box_deltas = self.sample_proposals(
        proposals = proposals,
        gt_classes = gt_classes,
        gt_box_deltas = gt_box_deltas,
        max_proposals = 128,
        positive_fraction = 0.25
      )
      gt_classes = tf.expand_dims(gt_classes, axis = 0)       # (N,num_classes) -> (1,N,num_classes) (as expected by loss function)
      gt_box_deltas = tf.expand_dims(gt_box_deltas, axis = 0) # (N,2,(num_classes-1)*4) -> (1,N,2,(num_classes-1)*4)

      # Ensure proposals are treated as constants and do not propagate gradients
      proposals = tf.stop_gradient(proposals)
      gt_classes = tf.stop_gradient(gt_classes)
      gt_box_deltas = tf.stop_gradient(gt_box_deltas)

    # Stage 3: Detector
    detector_classes, detector_box_deltas = self.stage3_detector_network(
      inputs = [
        input_image,
        feature_map,
        proposals
      ],
      training = training
    )

    # Losses
    if training:
      rpn_class_loss = self.stage2_region_proposal_network.class_loss(y_predicted = rpn_scores, gt_rpn_map = gt_rpn_map)
      rpn_regression_loss = self.stage2_region_proposal_network.regression_loss(y_predicted = rpn_box_deltas, gt_rpn_map = gt_rpn_map)
      detector_class_loss = self.stage3_detector_network.class_loss(y_predicted = detector_classes, y_true = gt_classes, from_logits = not self.activate_class_outputs)
      detector_regression_loss = self.stage3_detector_network.regression_loss(y_predicted = detector_box_deltas, y_true = gt_box_deltas)
      self.add_loss(rpn_class_loss)
      self.add_loss(rpn_regression_loss)
      self.add_loss(detector_class_loss)
      self.add_loss(detector_regression_loss)
      self.add_metric(rpn_class_loss, name = "rpn_class_loss")
      self.add_metric(rpn_regression_loss, name = "rpn_regression_loss")
      self.add_metric(detector_class_loss, name = "detector_class_loss")
      self.add_metric(detector_regression_loss, name = "detector_regression_loss")
    else:
      # Losses cannot be computed during inference and should be ignored
      rpn_class_loss = float("inf")
      rpn_regression_loss = float("inf")
      detector_class_loss = float("inf")
      detector_regression_loss = float("inf")

    # Return outputs
    return [
      rpn_scores,
      rpn_box_deltas,
      detector_classes,
      detector_box_deltas,
      proposals,
      rpn_class_loss,
      rpn_regression_loss,
      detector_class_loss,
      detector_regression_loss
   ]

  def predict_on_batch(self, x, score_threshold):
    _, _, detector_classes, detector_box_deltas, proposals, _, _, _, _ = super().predict_on_batch(x = x)
    scored_boxes_by_class_index = self.predictions_to_scored_boxes(
      input_image = x[0],
      classes = detector_classes,
      box_deltas = detector_box_deltas,
      proposals = proposals,
      score_threshold = score_threshold
    )
    return scored_boxes_by_class_index

  def load_imagenet_weights(self):
    keras_model = tf.keras.applications.VGG16(weights = "imagenet")
    for keras_layer in keras_model.layers:
      weights = keras_layer.get_weights()
      if len(weights) > 0:
        vgg16_layers = self.stage1_feature_extractor.layers + self.stage3_detector_network.layers
        our_layer = [ layer for layer in vgg16_layers if layer.name == keras_layer.name ]
        if len(our_layer) > 0:
          print("Loading VGG-16 ImageNet weights into layer: %s" % our_layer[0].name)
          our_layer[0].set_weights(weights)

  def predictions_to_scored_boxes(self, input_image, classes, box_deltas, proposals, score_threshold):
    # Eliminate batch dimension
    input_image = np.squeeze(input_image, axis = 0)
    classes = np.squeeze(classes, axis = 0)
    box_deltas = np.squeeze(box_deltas, axis = 0)

    # Convert logits to probability distribution if using logits mode
    if not self.activate_class_outputs:
      classes = tf.nn.softmax(classes, axis = 1).numpy()

    # Convert proposal boxes -> center point and size
    proposal_anchors = np.empty(proposals.shape)
    proposal_anchors[:,0] = 0.5 * (proposals[:,0] + proposals[:,2]) # center_y
    proposal_anchors[:,1] = 0.5 * (proposals[:,1] + proposals[:,3]) # center_x
    proposal_anchors[:,2:4] = proposals[:,2:4] - proposals[:,0:2]   # height, width

    # Separate out results per class: class_idx -> (y1, x1, y2, x2, score)
    boxes_and_scores_by_class_idx = {}
    for class_idx in range(1, classes.shape[1]):  # skip class 0 (background)
      # Get the regression parameters (ty, tx, th, tw) corresponding to this
      # class, for all proposals
      box_delta_idx = (class_idx - 1) * 4
      box_delta_params = box_deltas[:, (box_delta_idx + 0) : (box_delta_idx + 4)] # (N, 4)
      proposal_boxes_this_class = math_utils.convert_deltas_to_boxes(
        box_deltas = box_delta_params,
        anchors = proposal_anchors,
        box_delta_means = [0.0, 0.0, 0.0, 0.0],
        box_delta_stds = [0.1, 0.1, 0.2, 0.2]
      )

      # Clip to image boundaries
      proposal_boxes_this_class[:,0::2] = np.clip(proposal_boxes_this_class[:,0::2], 0, input_image.shape[0] - 1) # clip y1 and y2 to [0,height)
      proposal_boxes_this_class[:,1::2] = np.clip(proposal_boxes_this_class[:,1::2], 0, input_image.shape[1] - 1) # clip x1 and x2 to [0,width)

      # Get the scores for this class. The class scores are returned in
      # normalized categorical form. Each row corresponds to a class.
      scores_this_class = classes[:,class_idx]

      # Keep only those scoring high enough
      sufficiently_scoring_idxs = np.where(scores_this_class > score_threshold)[0]
      proposal_boxes_this_class = proposal_boxes_this_class[sufficiently_scoring_idxs]
      scores_this_class = scores_this_class[sufficiently_scoring_idxs]
      boxes_and_scores_by_class_idx[class_idx] = (proposal_boxes_this_class, scores_this_class)

    # Perform NMS per class
    scored_boxes_by_class_idx = {}
    for class_idx, (boxes, scores) in boxes_and_scores_by_class_idx.items():
      idxs = tf.image.non_max_suppression(
        boxes = boxes,
        scores = scores,
        max_output_size = proposals.shape[0],
        iou_threshold = 0.3
      )
      idxs = idxs.numpy()
      boxes = boxes[idxs]
      scores = np.expand_dims(scores[idxs], axis = 0) # (N,) -> (N,1)
      scored_boxes = np.hstack([ boxes, scores.T ])   # (N,5), with each row: (y1, x1, y2, x2, score)
      scored_boxes_by_class_idx[class_idx] = scored_boxes

    return scored_boxes_by_class_idx

  def label_proposals(self, proposals, gt_box_class_idxs, gt_box_corners, min_background_iou_threshold, min_object_iou_threshold):
    proposals = tf.concat([ proposals, gt_box_corners ], axis = 0)

    ious = math_utils.tf_intersection_over_union(boxes1 = proposals, boxes2 = gt_box_corners)

    best_ious = tf.math.reduce_max(ious, axis = 1)  # (N,) of maximum IoUs for each of the N proposals
    box_idxs = tf.math.argmax(ious, axis = 1)       # (N,) of ground truth box index for each proposal
    gt_box_class_idxs = tf.gather(gt_box_class_idxs, indices = box_idxs)  # (N,) of class indices of highest-IoU box for each proposal
    gt_box_corners = tf.gather(gt_box_corners, indices = box_idxs)        # (N,4) of box corners of highest-IoU box for each proposal

    idxs = tf.where(best_ious >= min_background_iou_threshold)  # keep proposals w/ sufficiently high IoU
    proposals = tf.gather_nd(proposals, indices = idxs)
    best_ious = tf.gather_nd(best_ious, indices = idxs)
    gt_box_class_idxs = tf.gather_nd(gt_box_class_idxs, indices = idxs)
    gt_box_corners = tf.gather_nd(gt_box_corners, indices = idxs)

    retain_mask = tf.cast(best_ious >= min_object_iou_threshold, dtype = gt_box_class_idxs.dtype) # (N,), with 0 wherever best_iou < threshold, else 1
    gt_box_class_idxs = gt_box_class_idxs * retain_mask

    num_classes = self.num_classes
    gt_classes = tf.one_hot(indices = gt_box_class_idxs, depth = num_classes) # (N,num_classes)

    proposal_centers = 0.5 * (proposals[:,0:2] + proposals[:,2:4])          # center_y, center_x
    proposal_sides = proposals[:,2:4] - proposals[:,0:2]                    # height, width
    gt_box_centers = 0.5 * (gt_box_corners[:,0:2] + gt_box_corners[:,2:4])  # center_y, center_x
    gt_box_sides = gt_box_corners[:,2:4] - gt_box_corners[:,0:2]            # height, width

    detector_box_delta_means = tf.constant([0, 0, 0, 0], dtype = tf.float32)
    detector_box_delta_stds = tf.constant([0.1, 0.1, 0.2, 0.2], dtype = tf.float32)
    tyx = (gt_box_centers - proposal_centers) / proposal_sides  # ty = (gt_center_y - proposal_center_y) / proposal_height, tx = (gt_center_x - proposal_center_x) / proposal_width
    thw = tf.math.log(gt_box_sides / proposal_sides)            # th = log(gt_height / proposal_height), tw = (gt_width / proposal_width)
    box_delta_targets = tf.concat([ tyx, thw ], axis = 1)       # (N,4) box delta regression targets tensor
    box_delta_targets = (box_delta_targets - detector_box_delta_means) / detector_box_delta_stds  # mean and standard deviation adjustment

    gt_box_deltas_mask = tf.repeat(gt_classes, repeats = 4, axis = 1)[:,4:]             # create masks using interleaved repetition, remembering to discard class 0
    gt_box_deltas_values = tf.tile(box_delta_targets, multiples = [1, num_classes - 1]) # populate regression targets with straightforward repetition of each row (only those columns corresponding to class will be masked on)
    gt_box_deltas_mask = tf.expand_dims(gt_box_deltas_mask, axis = 0)     # (N,4*(C-1)) -> (1,N,4*(C-1))
    gt_box_deltas_values = tf.expand_dims(gt_box_deltas_values, axis = 0) # (N,4*(C-1)) -> (1,N,4*(C-1))
    gt_box_deltas = tf.concat([ gt_box_deltas_mask, gt_box_deltas_values ], axis = 0) # (2,N,4*(C-1))
    gt_box_deltas = tf.transpose(gt_box_deltas, perm = [ 1, 0, 2])        # (N,2,4*(C-1))

    return proposals, gt_classes, gt_box_deltas

  def sample_proposals(self, proposals, gt_classes, gt_box_deltas, max_proposals, positive_fraction):
    if max_proposals <= 0:
      return proposals, gt_classes, gt_box_deltas

    class_indices = tf.argmax(gt_classes, axis = 1) # (N,num_classes) -> (N,), where each element is the class index (highest score from its row)
    positive_indices = tf.squeeze(tf.where(class_indices > 0), axis = 1)  # (P,), tensor of P indices (the positive, non-background classes in class_indices)
    negative_indices = tf.squeeze(tf.where(class_indices <= 0), axis = 1) # (N,), tensor of N indices (the negative, background classes in class_indices)
    num_positive_proposals = tf.size(positive_indices)
    num_negative_proposals = tf.size(negative_indices)

    num_samples = tf.minimum(max_proposals, tf.size(class_indices))
    num_positive_samples = tf.minimum(tf.cast(tf.math.round(tf.cast(num_samples, dtype = float) * positive_fraction), dtype = num_samples.dtype), num_positive_proposals)
    num_negative_samples = tf.minimum(num_samples - num_positive_samples, num_negative_proposals)

    positive_sample_indices = tf.random.shuffle(positive_indices)[:num_positive_samples]
    negative_sample_indices = tf.random.shuffle(negative_indices)[:num_negative_samples]
    indices = tf.concat([ positive_sample_indices, negative_sample_indices ], axis = 0)

    return tf.gather(proposals, indices = indices), tf.gather(gt_classes, indices = indices), tf.gather(gt_box_deltas, indices = indices)

def create_optimizer(optimizer = 'sgd', clipnorm = 0.0, learning_rate = 1e-3, momentum = 0.9, beta_1 = 0.9, beta_2 = 0.999):
  kwargs = {}
  if clipnorm > 0:
    kwargs = { 'clipnorm': clipnorm }
  if optimizer   == 'sgd':
    optimizer = SGD(learning_rate = learning_rate, momentum = momentum, **kwargs)
  elif optimizer == 'adam':
    optimizer = Adam(learning_rate = learning_rate, beta_1 = beta1, beta_2 = beta2, **kwargs)
  else:
    raise ValueError("Optimizer must be 'sgd' for stochastic gradient descent or 'adam' for Adam")
  return optimizer

class rcnn:
  def __init__(self, num_classes = 21, allow_edge_proposals = True, custom_roi_pool = False, weight_decay = 5e-4, detector_logits = True, dropout = 0.0):

    # Construct model and load initial weights

    model = FasterRCNNModel(
      num_classes            = num_classes,
      allow_edge_proposals   = allow_edge_proposals,
      custom_roi_pool        = custom_roi_pool,
      activate_class_outputs = detector_logits,
      l2                     = 0.5 * weight_decay,
      dropout_probability    = dropout
    )
    model.build(
      input_shape = [
        (1, None, None, 3),     # input_image: (1, height_pixels, width_pixels, 3)
        (1, None, None, 9 * 4), # anchor_map: (1, height, width, num_anchors * 4)
        (1, None, None, 9),     # anchor_valid_map: (1, height, width, num_anchors)
        (1, None, None, 9, 6),  # gt_rpn_map: (1, height, width, num_anchors, 6)
        (1, None),              # gt_box_class_idxs_map: (1, num_gt_boxes)
        (1, None, 4)            # gt_box_corners_map: (1, num_gt_boxes, 4)
      ]
    )
    model.compile(optimizer = create_optimizer()) # losses not needed here because they were baked in at model construction

