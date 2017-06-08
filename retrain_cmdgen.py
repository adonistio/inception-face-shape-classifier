#This script generates the Windows CMD command to re-train the Inception v3 model that tees CMD line prompts into a text file

import subprocess
from PIL import Image
from PIL import ImageFilter, ImageEnhance, ImageOps
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
import time
import tensorflow as tf, sys
import os
import numpy as np
import cv2

#where to find python.exe
py_path = "python"

#where to store generated files?
newmodel_path = "\"C:/Users/Adonis Tio/Jupyter/face_shape_celebs3_aug_500"
newmodel_path1 = "C:/Users/Adonis Tio/Jupyter/face_shape_celebs3_aug_500"

#where to find the retrain.py script?
#v2 also prints validation and train size
retrain_path = "\"C:/Users/Adonis Tio/Anaconda3_x64/Lib/site-packages/tensorflow/examples/image_retraining/retrain_v2.py\""

#where to cache tensorflow values per image before last layer of inception graph? (no. of files generated = no. of training images)
outbottle_path = "\"C:/Users/Adonis Tio/Jupyter/face_shape_celebs3_aug" + "/bottlenecks\""

#where can you find the original inception graph folder?
inception_path = "\"C:/Users/Adonis Tio/Jupyter/inception\""

#where to create output graph?
outgraph_path = newmodel_path + "/retrained_graph.pb\""

#where to create output list of labels, retrain logs, CMD logs, and CMM commands?
outlabel_path = newmodel_path + "/retrained_labels.txt\""

sum_path = newmodel_path + "/retrain_logs\""

console_path = newmodel_path + "/console_logs.txt\""

cmd_path = newmodel_path1 + "/console_commands.txt"

#where to save list of training images
trainset_path = newmodel_path + "/trainset_files.txt\""
testset_path = newmodel_path + "/testset_files.txt\""
validset_path = newmodel_path + "/validset_files.txt\""
xtestset_path = newmodel_path + "/xtestset_files.txt\""
pred_path = newmodel_path + "/predictions.txt\""

#directory of training images; folder must contain sub-folders named correct labels
#at least two folders/classes needed
imagetrain_path = "\"C:/Users/Adonis Tio/Jupyter/Google Images/celebs3_augmented\""

cmd = ("python "+ retrain_path
       + " --image_dir=" + imagetrain_path
       + " --output_graph=" + outgraph_path 
       + " --output_labels=" + outlabel_path 
       + " --summaries_dir=" + sum_path
       + " --trainset_dir=" + trainset_path
       + " --testset_dir=" + testset_path
       + " --validset_dir=" + validset_path
       + " --xtestset_dir=" + xtestset_path
       + " --prediction_dir=" + pred_path
       + " --how_many_training_steps " + str(4000)
       + " --learning_rate " + str(0.01) 
       + " --testing_percentage " + str(10)
       + " --validation_percentage " + str(10)
       + " --eval_step_interval " + str(10)
       + " --train_batch_size " + str(500)
       + " --test_batch_size " + str(-1)
       + " --validation_batch_size " + str(-1) 
       + " --print_misclassified_test_images " + str(False)
       + " --model_dir=" + inception_path
       + " --bottleneck_dir=" + outbottle_path
       + " | tee " + console_path)

commands = []

print(cmd)
commands.append(cmd)

cmd = "tensorboard --logdir " + sum_path
print(cmd)

commands.append("\n")
commands.append(cmd)
np.savetxt(cmd_path,commands,fmt = '%s')