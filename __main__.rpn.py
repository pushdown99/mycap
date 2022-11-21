import argparse
from . import FasterRCNN, vocFasterRCNN, niaFasterRCNN, cocoFasterRCNN


if __name__ == "__main__":
  parser = argparse.ArgumentParser("FasterRCNN")
  group = parser.add_mutually_exclusive_group()
  group.add_argument ("--train", action = "store_true", help = "Train model")
  group.add_argument ("--eval", action = "store_true", help = "Evaluate model")
  group.add_argument ("--predict", metavar = "url", action = "store", type = str, help = "Run inference on image and display detected boxes")
  group.add_argument ("--predict-to-file", metavar = "url", action = "store", type = str, help = "Run inference on image and render detected boxes to 'predictions.png'")
  group.add_argument ("--predict-all", metavar = "name", action = "store", type = str, help = "Run inference on all images in the specified dataset split and write to directory 'predictions_${split}'")
  parser.add_argument("--load-from", metavar = "file", action = "store", help = "Load initial model weights from file")
  parser.add_argument("--save-to", metavar = "file", action = "store", help = "Save final trained weights to file")
  parser.add_argument("--save-best-to", metavar = "file", action = "store", help = "Save best weights (highest mean average precision) to file")
  parser.add_argument("--dataset-dir", metavar = "dir", action = "store", default = "datasets/VOCdevkit/VOC2007", help = "VOC dataset directory")
  parser.add_argument("--train-split", metavar = "name", action = "store", default = "trainval", help = "Dataset split to use for training")
  parser.add_argument("--eval-split", metavar = "name", action = "store", default = "test", help = "Dataset split to use for evaluation")
  parser.add_argument("--cache-images", action = "store_true", help = "Cache images during training (requires ample CPU memory)")
  parser.add_argument("--periodic-eval-samples", metavar = "count", action = "store", default = 1000, help = "Number of samples to use during evaluation after each epoch")
  parser.add_argument("--checkpoint-dir", metavar = "dir", action = "store", help = "Save checkpoints after each epoch to the given directory")
  parser.add_argument("--plot", action = "store_true", help = "Plots the average precision of each class after evaluation (use with --train or --eval)")
  parser.add_argument("--log-csv", metavar = "file", action = "store", help = "Log training metrics to CSV file")
  parser.add_argument("--epochs", metavar = "count", type = int, action = "store", default = 1, help = "Number of epochs to train for")
  parser.add_argument("--optimizer", metavar = "name", type = str, action = "store", default = "sgd", help = "Optimizer to use (\"sgd\" or \"adam\")")
  parser.add_argument("--learning-rate", metavar = "value", type = float, action = "store", default = 1e-3, help = "Learning rate")
  parser.add_argument("--clipnorm", metavar = "value", type = float, action = "store", default = 0.0, help = "Gradient norm clipping (use 0 for none)")
  parser.add_argument("--momentum", metavar = "value", type = float, action = "store", default = 0.9, help = "SGD momentum")
  parser.add_argument("--beta1", metavar = "value", type = float, action = "store", default = 0.9, help = "Adam beta1 parameter (decay rate for 1st moment estimates)")
  parser.add_argument("--beta2", metavar = "value", type = float, action = "store", default = 0.999, help = "Adam beta2 parameter (decay rate for 2nd moment estimates)")
  parser.add_argument("--weight-decay", metavar = "value", type = float, action = "store", default = 5e-4, help = "Weight decay")
  parser.add_argument("--dropout", metavar = "probability", type = float, action = "store", default = 0.0, help = "Dropout probability after each of the two fully-connected detector layers")
  parser.add_argument("--custom-roi-pool", action = "store_true", help = "Use custom RoI pool implementation instead of TensorFlow crop-and-resize with max-pool (much slower)")
  parser.add_argument("--detector-logits", action = "store_true", help = "Do not apply softmax to detector class output and compute loss from logits directly")
  parser.add_argument("--no-augment", action = "store_true", help = "Disable image augmentation (random horizontal flips) during training")
  parser.add_argument("--exclude-edge-proposals", action = "store_true", help = "Exclude proposals generated at anchors spanning image edges from being passed to detector stage")
  parser.add_argument("--dump-anchors", metavar = "dir", action = "store", help = "Render out all object anchors and ground truth boxes from the training set to a directory")
  parser.add_argument("--debug-dir", metavar = "dir", action = "store", help = "Enable full TensorFlow Debugger V2 logging to specified directory")
  options = parser.parse_args()

  cocoFasterRCNN.Model (options)