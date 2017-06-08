This repository contains the scripts used in retraining the Inception v3 model to classify images of human faces into five basic shapes: heart, oblong, oval, round, and square. The repository also contains the scripts used to benchmark it to traditional classifiers using features derived from facial landmark coordinates generated using OpenCV and DLIB pre-trained models.

Training images used are downloaded via Google image search and may have copyright constraints; I can give you a copy of the images, bottleneck files, and/or feature files by request if you promise not to redistribute and to only use for academic purposes. You can send me an e-mail at adonis@eee.upd.edu.ph if you are interested.

CLASSIFY_FACE.PY
This script runs the re-trained Inception model to classify a single or a batch of images

CLASSIFY_FACE_CONFUSION.PY
Similar to classify_face.py but generates a text file of results and a confusion matrix

EXTRACT_FEATURES.PY
This script detects the face(s) in the image, specifies the bounding box, detects the facial landmarks, and extracts the features for training

PROCESS_IMAGE.PY
This script contains a couple of image pre-processing and augmentation functions like squaring an image, filters, blurs, zoom, rotate, flip, and recolor, etc

RETRAIN_CMDGEN.PY
This script generates the Windows CMD command to re-train the Inception v3 model that tees CMD line prompts into a text file; Set up the needed files and directories then run in the CMD line to retrain the model.

RETRAIN_v2.PY
#Slight modifications in defining the overall test set to include all images, resolved issue of "doubling-counting" of validation and test images, added CMD line arguments on where to save useful info as txt file

TRAIN_CLASSIFIERS.PY
This script trains the LDA, SVM-LIN, SVM-RBF, MLP, and KNN classifiers for a set of training set sizes

PAPER.PDF
A short paper describing the methodology and experimental results
