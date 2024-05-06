# -*- coding: utf-8 -*-
"""Attribute_noise_classification.ipynb
"""
import numpy as np
import pandas as pd
from lsa import lsa_
import time

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import nltk
import warnings
import json
warnings.filterwarnings('ignore')
nltk.download('punkt')

file = pd.read_csv('clean_data.csv')



mets = ["euclidean","cosine","jaccard","mahalanobis","pearson"]
# mets = ["euclidean"]

vectors_b,labels = lsa_(file)


vectors_b = np.array(vectors_b)
labels = np.array(labels)


vectors_b = np.array(vectors_b[:,:50])




feature_size_b = len(vectors_b[0])


print(feature_size_b)

feature_dict = { "LSA": [vectors_b, feature_size_b]}

def pearson_distance(v1, v2):
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Calculate means
    mean1 = np.mean(v1)
    mean2 = np.mean(v2)

    # Calculate covariance
    covariance = np.sum((v1 - mean1) * (v2 - mean2))

    # Calculate denominators
    denominator1 = np.sqrt(np.sum((v1 - mean1)**2))
    denominator2 = np.sqrt(np.sum((v2 - mean2)**2))

    # Check for division by zero
    if denominator1 * denominator2 == 0:
        return 0

    # Calculate Pearson distance
    return 1.0 - covariance / (denominator1 * denominator2)

def jaccard_distance(u, v):
    dot_product = np.dot(u, v)
    magnitude_u = np.dot(u, u)
    magnitude_v = np.dot(v, v)

    return 1 - dot_product / (magnitude_u + magnitude_v - dot_product)

def get_metric(met):
    if met == 'cosine':
        metric = 'cosine'
    elif met == 'euclidean':
        metric = 'euclidean'
    elif met == 'pearson':
        metric = pearson_distance
    elif met == 'jaccard':
        metric = jaccard_distance

    return metric

n_neig = [10,150,60,5,150] 

scor_att = {}

for feature in feature_dict:
    scor_att[feature] = {}

    for met in mets:
        scor_att[feature][met] = []


feature_vec = feature_dict[feature][0]

feature_size = feature_dict[feature][1]

x_train, x_test, y_train, y_test = train_test_split(feature_vec,labels,test_size=0.3)
doc_count = len(x_test)



for feature in feature_dict:
     #document index which we want to make changes

    for ind,distance in enumerate(mets):
        
        
    
        s_a = []
        start_time = time.time()
        for i in range(10):
            print(feature," ",distance," beginning iter: ",i)

            if distance == "mahalanobis":
                covariance_matrix = np.cov(feature_vec.T, bias=True)
                epsilon=1e-6
                # Add a small positive constant to the diagonal for regularization
                covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])
            

                model1 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]), metric= distance ,metric_params={'VI': np.cov(np.linalg.inv(covariance_matrix), rowvar=False)})
                model1.fit(x_train,y_train)
                s_a.append(model1.score(x_test,y_test))


            else:
                
                model1 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]),  metric= get_metric(distance))
                model1.fit(x_train,y_train)
                s_a.append(model1.score(x_test,y_test))
        end_time = time.time()
        execution_time = end_time - start_time
        scor_att[feature][distance].append([np.mean(s_a),execution_time])
       


file_path = 'Accuracy(LSA).json'
Noise = {"accuracy":scor_att}
with open(file_path, 'w') as f:
    json.dump(Noise, f, indent=3)

import matplotlib.pyplot as plt

for feature in feature_dict:

    plt.title("pero (" + feature + ")")
    plt.xlabel("executiontime")
    plt.ylabel("Avg accuracy")

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    marker = ['+','o','*','x','.']

    for i, met in enumerate(mets):
        color = colors[i % len(colors)]


        plt.scatter(scor_att[feature][met][1], scor_att[feature][met][0], label=met, color=color, marker = marker[i])
        # plt.plot(scor_att[feature][met][1], scor_att[feature][met][0], color=color)
    plt.legend()
    plt.savefig("accuracy (" + feature + ").png")




