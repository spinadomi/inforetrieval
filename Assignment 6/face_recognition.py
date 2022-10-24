# You received the dataset fr_dataset.zip.
# Extract it in a folder, possibly the same in which you are working.
# The folder fr_dataset includes training and test set. 
# The training folder includes 10 subfolders, named from 00, 01, ... , 09, one for each known person. 
# Each of these folders contains 10 images, so you have 10 faces for each known person.  
# The test folder includes 11 subfolders, named from 00, 01, ... , 09, 10, one for each known person and the last one with faces of unknown people. 
# Each of the first 10 folders contains 5 images, so you have 5 test faces for each known person, while the last contains 50 faces of unknown people.
# The goal of the assignment is the realization of a face recognition system, which correctly identify known people and reject unknown people.
# To complete the assignment, follow the instructions and complete the parts tagged with # YOUR CODE HERE 

# To execute the code you need to install the requirements with this command
# pip install opencv-python scipy keras_vggface tensorflow keras_applications scikit-learn
# In case of ModuleNotFoundError: No module named 'keras.engine.topology'
# Change from keras.engine.topology import get_source_inputs 
# in from keras.utils.layer_utils import get_source_inputs
# in file keras_vggface/models.py

import cv2
import numpy as np

from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
from scipy.spatial.distance import cosine
from sklearn.metrics import accuracy_score, confusion_matrix

import os
from glob import glob

# This method takes as input the face recognition model and the filename of the image and returns
# the feature vector
def extract_features(face_reco_model, filename):
    faceim = cv2.imread(filename)
    faceim = cv2.resize(faceim, (224,224))
    faceim = preprocess_input([faceim.astype(np.float32)], version=2)
    feature_vector = (face_reco_model.predict(faceim)).flatten()
    return feature_vector

# Number of subjects in the training set
number_of_known_people = 10
# Number of images stored for a known person
number_of_training_images_per_person = 10
# Maximum distance for considering a test sample as a face of a known person
rejection_threshold = 1.00
# Dataset path - Folder in which you extracted fr_dataset.zip, you can use relative path
dataset_path = ''

# Load the VGG-Face model based on ResNet-50
face_reco_model = VGGFace(model='resnet50', include_top=False, pooling='avg')

# Create the database of known people
database = []
training_path = os.path.join(dataset_path, 'fr_dataset', 'training')
for i in range(number_of_known_people):
    person_path = os.path.join(training_path, str(i).zfill(2))
    count = 0
    person = []
    for filename in glob(os.path.join(person_path,'*.jpg')):
        if count < number_of_training_images_per_person:
            feature_vector = extract_features(face_reco_model, filename)
            person.append({"id": i, "feature_vector": feature_vector, "filename": filename})
            count += 1
            print("Loading %d - %d"%(i, count))
    database.append(person)

# Print information about the database of known people
for i in range(number_of_known_people):
    for j in range(number_of_training_images_per_person):
        print("%d %s"%(database[i][j]['id'], database[i][j]['filename'])) 

# For each test sample, compute the feature vector and the cosine distance 
# (https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.cosine.html) 
# with all the known people and: 
# (1) if the minimum distance is less than the rejection threshold, associate the more similar person; 
# (2) otherwise, the face belongs to an unknown person.
# Here (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html) 
# and here (https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html) 
# you find scikit-learn accuracy and confusion matrix documentation
# In groundtruth you must insert for each sample the correct label, while in the predictions the predicted label. 
groundtruth = []
predictions = []
test_path = os.path.join(dataset_path, 'fr_dataset', 'test')
for i in range(11):
    person_path = os.path.join(test_path, str(i).zfill(2))
    for filename in glob(os.path.join(person_path,'*.jpg')):
        # TODO: implement code to compute the feature vector and the cosine distance with all the known people
        # Compute the feature vector of the test sample
        feature_vector = extract_features(face_reco_model, filename)
        # Compute the cosine distance with all the known people
        distances = []
        for j in range(number_of_known_people):
            for k in range(number_of_training_images_per_person):
                distances.append(cosine(database[j][k]['feature_vector'], feature_vector))
        # for groundtruth insert the correct label, while for predictions the predicted label
        groundtruth.append(i)
        if min(distances) < rejection_threshold:
            predictions.append(np.argmin(distances))
        else:
            predictions.append(10)
        # NOTE: was just trying things, I don't think this is correct

# 1) Try different values between 1 and 10 for number_of_known_people
#   Report accuracies (with a chart if you prefer) and confusion matrices and discuss the results
# 2) Try different values between 1 and 10 number_of_training_images_per_person
#   Report accuracies (with a chart if you prefer) and confusion matrices and discuss the results
# 3) Try different values between 0.1 and 1.0 for the rejection_threshold
#   Report accuracies (with a chart if you prefer) and confusion matrices and discuss the results
print("Accuracy score: %.3f" % (accuracy_score(groundtruth, predictions)))
print("Normalized confusion matrix\n %s" % (confusion_matrix(groundtruth, predictions, normalize='true')))

# YOUR (OPTIONAL) CODE HERE FOR REPORTING OTHER RESULTS
#################
print("YOUR OPTIONAL CODE HERE")
#################