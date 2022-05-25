import numpy as np
from skimage import io, color, img_as_ubyte
import os
import pandas as pd
from skimage.feature import greycomatrix, greycoprops

aa = ['Bacterialblight','Blast','Brownspot','Leaf smut','Tungro','Not a rice disease image']
for i in aa:

    print(i)

alllist = []
features = []
labels = []

# ===========================================================

for myfolders in aa:
    path = 'C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\RiceLeafDiseaseImages\\'+str(myfolders)


    a = os.listdir(path)
    for ii in a:
        rgbImg = io.imread(str(path+"\\"+str(ii)))
        grayImg = img_as_ubyte(color.rgb2gray(rgbImg))

        distances = [1, 2, 3]
        angles = [0, np.pi / 4, np.pi / 2, 3 * np.pi / 4]
        properties = ['energy', 'homogeneity',
                      'dissimilarity', 'correlation', 'contrast']

        glcm = greycomatrix(grayImg,
                            distances=distances,
                            angles=angles,
                            symmetric=True,
                            normed=True)

        feats = np.hstack([greycoprops(glcm, 'homogeneity').ravel()
                          for prop in properties])
        feats1 = np.hstack([greycoprops(glcm, 'energy').ravel()
                           for prop in properties])
        feats2 = np.hstack(
            [greycoprops(glcm, 'dissimilarity').ravel() for prop in properties])
        feats3 = np.hstack(
            [greycoprops(glcm, 'correlation').ravel() for prop in properties])
        feats4 = np.hstack([greycoprops(glcm, 'contrast').ravel()
                           for prop in properties])

        k = np.mean(feats)
        l = np.mean(feats1)
        m = np.mean(feats2)
        n = np.mean(feats3)
        o = np.mean(feats4)


        features.append([k, l, m, n, o])
        labels.append(myfolders)
        disease = myfolders
        aa = [k, l, m, n, o, disease]
        alllist.append(aa)
        data = pd.DataFrame(
            alllist, columns=['f1', 'f2', 'f3', 'f4', 'f5', 'Disease'])
        print("======================================",ii)

        data.to_csv(
            'C:\\Users\\user\\PycharmProjects\\riceplantdisase\\static\\image_detection1.csv')

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.1, random_state=0)

from sklearn.ensemble import RandomForestClassifier
a = RandomForestClassifier(n_estimators=100)

a.fit(features, labels)

m = a.predict(X_test)

from sklearn.metrics import accuracy_score

s = accuracy_score(y_test,m)

print(s, "acccuracy")

