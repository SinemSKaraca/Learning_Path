import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# PREPARE DATA
input_dir = r"C:\Users\user\Desktop\deneme\computerVision\Image_Classification\clf-data"
categories  =["empty", "not_empty"]

data = []
labels = []
for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        data.append(img.flatten())   # flatten() -> converts matrix into 1D array
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)


# TRAIN / TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(data,
                                                    labels,
                                                    test_size=0.2,
                                                    shuffle=True,
                                                    stratify=labels)


# TRAIN CLASSIFIER
classifier = SVC()

params = [{"gamma": [0.01, 0.001, 0.0001], "C": [1, 10, 100, 1000]}]

grid_search = GridSearchCV(classifier, params).fit(X_train, y_train)


# TEST PERFORMANCE
best_estimator = grid_search.best_estimator_

y_pred = best_estimator.predict(X_test)

score = accuracy_score(y_pred, y_test)

print("{}% of the samples were correctly classified".format(str(score * 100)))

pickle.dump(best_estimator, open("./model.p", "wb"))