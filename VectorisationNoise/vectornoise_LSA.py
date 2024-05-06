# -*- coding: utf-8 -*-
"""Attribute_noise_classification.ipynb
"""
import numpy as np
import pandas as pd


from lsa import lsa_
from cmath import sqrt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

import nltk
import warnings
import json
warnings.filterwarnings('ignore')
nltk.download('punkt')

file = pd.read_csv('clean_data.csv')

noise_levels = np.arange(0.0,0.7,0.1)

mets = ["Euclidean","Cosine","Jaccard","Mahalanobis"]
# mets = ["euclidean","cosine","jaccard","mahalanobis","pearson"]
# mets = ["euclidean"]

vectors_b,labels = lsa_(file)


vectors_b = np.array(vectors_b)
labels = np.array(labels)






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
    if met == 'Cosine':
        metric = 'cosine'
    elif met == 'Euclidean':
        metric = 'euclidean'
    elif met == 'pearson':
        metric = pearson_distance
    elif met == 'Jaccard':
        metric = jaccard_distance
   

    return metric

def Attribute_noise(vector, size, index):

    vec = vector.copy()
    for col in range(size):

        avg = vec[:,col].mean()            # generating mean and standard deviation
        standard_dev = vec[:,col].std()    # for current column

        for i in index:

            vec[i][col] = np.random.normal(avg,standard_dev)
    return vec

def class_noise(y,noise_level):

    size = len(y)
    z = y.copy()
    random_samples = np.random.uniform(0,1,size)

    for i in range(size):

        if(random_samples[i] < noise_level):
            old_label = z[i]
            new_label = np.random.randint(0,20)

            while(new_label == old_label):
                 new_label = np.random.randint(0,20)


            z[i] = new_label
    return z

n_neig = [10,150,60,5,10]

scor_att = {}
scor_clas = {}
for feature in feature_dict:
    scor_att[feature] = {}
    scor_clas[feature] = {}
    for met in mets:
        scor_att[feature][met] = []
        scor_clas[feature][met] = []

feature_vec = feature_dict[feature][0]

feature_size = feature_dict[feature][1]

x_train, x_test, y_train, y_test = train_test_split(feature_vec,labels,test_size=0.3)
doc_count = len(x_test)


for noise_level in noise_levels:

    for feature in feature_dict:

        a = int(noise_level*doc_count)

         #document index which we want to make changes


        for ind,distance in enumerate(mets):
          
          
        
          s_a = []
          s_c = []
          for i in range(10):
            print(noise_level," ",feature," ",distance," beginning iter: ",i)
            doc_index = np.random.randint(0,doc_count,a)
            x = Attribute_noise(x_test, feature_size, doc_index) #calling attribute noise

            y = class_noise(y_train,noise_level) #calling class noise

            if distance == "Mahalanobis":
                covariance_matrix = np.cov(feature_vec.T, bias=True)
                epsilon=1e-6
                # Add a small positive constant to the diagonal for regularization
                covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])
            

                model1 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]), metric= "mahalanobis" ,metric_params={'VI': np.cov(np.linalg.inv(covariance_matrix), rowvar=False)})
                model1.fit(x_train,y_train)
                s_a.append(model1.score(x,y_test))


                model2 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]), metric= "mahalanobis" ,metric_params={'VI': np.cov(np.linalg.inv(covariance_matrix), rowvar=False)})
                model2.fit(x_train,y)
                s_c.append(model2.score(x_test,y_test))

            else:
                
                model1 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]),  metric= get_metric(distance))
                model1.fit(x_train,y_train)
                s_a.append(model1.score(x,y_test))


                model2 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]), metric= get_metric(distance))
                model2.fit(x_train,y)
                s_c.append(model2.score(x_test,y_test))

          scor_att[feature][distance].append(np.mean(s_a))
          scor_clas[feature][distance].append(np.mean(s_c))


import matplotlib.pyplot as plt

for feature in feature_dict:

    plt.title("Attribute Noise (" + feature + ")")
    plt.xlabel("noise level(fn)")
    plt.ylabel("Avg accuracy")

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    marker = ['+','o','*','x','.']

    for i, met in enumerate(mets):
        color = colors[i % len(colors)]


        plt.scatter(noise_levels, scor_att[feature][met], label=met, color=color, marker = marker[i])
        plt.plot(noise_levels, scor_att[feature][met], color=color)
    plt.legend()
    plt.savefig("vectornoise_LSA.png")

plt.clf()

for feature in feature_dict:

    plt.title("Class Noise (" + feature + ")")
    plt.xlabel("noise level(fn)")
    plt.ylabel("Avg score")

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    marker = ['+','o','*','x','.']
    for i, met in enumerate(mets):
        color = colors[i % len(colors)]

        # plt.scatter(fn, scor[feature][met], label="augmented data", color=colors)
        # plt.plot(fn, scor[feature][met], color=color[0])
        plt.scatter(noise_levels, scor_clas[feature][met], label=met, color=color, marker = marker[i])
        plt.plot(noise_levels, scor_clas[feature][met], color=color)
    plt.legend()
    plt.savefig("Classnoise_LSA.png")

file_path = 'Vectornoise(LSA).json'
Noise = {"Attribute":scor_att, "Class":scor_clas}
with open(file_path, 'w') as f:
    json.dump(Noise, f, indent=3)

