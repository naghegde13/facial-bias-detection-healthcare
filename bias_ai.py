import glob
import random
from sklearn.utils import compute_class_weight
import classify
import cv2
import numpy as np
from sklearn.metrics import classification_report
from os.path import dirname, join
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

image_path = join(dirname(__file__), r"..\..\data\images")
imagePaths = sorted(glob.glob(image_path + "\*.jpg"))
# Sort out the data based on gender
# number of images of males: 4372
# number of images of females: 5407
males = []
females = []
for imagePath in imagePaths:
    if imagePath.split("_")[3] == "0":
        males.append(imagePath)
    if imagePath.split("_")[3] == "1":
        females.append(imagePath)
# Select random images from the sorted data. Number will be taken from the user.
print(len(females))

random_females = random.sample(females, 2500)
random_males = random.sample(males, 2500)
imagePaths = np.concatenate((random_males, random_females))

target, label = classify.classify(imagePaths)
targetNames = np.unique(target)
print(classification_report(target, label, target_names=targetNames,))
sns.heatmap(pd.DataFrame(classification_report(target, label, target_names=targetNames,output_dict=True)).iloc[:-1, :].T, annot=True)
plt.show()