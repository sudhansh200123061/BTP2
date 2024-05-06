# -*- coding: utf-8 -*-
"""Noise_text.ipynb

"""

import numpy as np
import pandas as pd

from lsa import lsa_


from augment import gen_eda
from cmath import sqrt
from sklearn.neighbors import KNeighborsClassifier

import nltk
import warnings
import json
warnings.filterwarnings('ignore')
nltk.download('punkt')
nltk.download('wordnet')

file = pd.read_csv('clean_data.csv')

Noise_levels = np.arange(0.0,0.7,0.1) #initialising aplpha values
n_neig = [5,5,30,5,10]
mets = ["Euclidean","Cosine","Jaccard","Mahalanobis"]

features = [ "LSA"]


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


scor_att = {}

for feature in features:
    scor_att[feature] = {}

    for met in mets:
        scor_att[feature][met] = []
  

lenght = len(file)
test_split = int((lenght*3)/10)

val_dict = {}
for Noise in Noise_levels:
    val_dict[Noise] = {}

    for i in range(10):
        val_dict[Noise][i] = {}

        index = np.random.randint(0,lenght,test_split)
        test_df = file.iloc[index]
        train_df = file.drop(index)
        train_df = train_df[["doc_text", "number_label"]]
        test_len = len(test_df)
        a = int(Noise*test_len)
        indd = np.random.randint(0,test_len,a)
        aug_df = gen_eda(test_df,0.1,4,indd)

        aug_len = len(aug_df)
        
        print(aug_len," ",test_len)
        df = pd.concat([aug_df,test_df,train_df]).reset_index(drop=True) #combining these dataframe, because we need same feature space.

        vectors_b,labels = lsa_(df)

        vectors_b = np.array(vectors_b)
        
     
        labels = np.array(labels)
     
        labels = labels.astype(int)

        

        feature_dict = { "LSA": [vectors_b]}

        for feature in feature_dict:
            val_dict[Noise][i][feature] = {}

            vectors = feature_dict[feature][0]
            
            # below is the structure of our vectors matrix and label. since we did (concat([aug_df,test_df,train_df]))
            #  * <- represents rows of augmented data (x_aug,y_aug)
            #  + <- represents rows of original data before augmentation (x_test, y_test)
            #  - <- represents rows of remaining data, when we did split, one part was for augmentation and other part is this one (x_train, y_train)
            # we are adding class noise in y_train and making y which contains y_train with noise.
            # * * * * * * * *   *
            # * * * * * * * *   *
            # * * * * * * * *   *
            # * * * * * * * *   *
            # + + + + + + + +   +
            # + + + + + + + +   +
            # - - - - - - - -   -
            # - - - - - - - -   -
            # - - - - - - - -   -

            x_aug = vectors[:aug_len,: ]
            x_test = vectors[aug_len :test_len + aug_len,: ]
            x_train = vectors[test_len + aug_len : , :]

            y_aug = labels[:aug_len]
            y_test = labels[aug_len :test_len + aug_len]
            y_train = labels[test_len + aug_len : ]

     

            print(len(x_train.T))
            for ind,distance in enumerate(mets):

                print(Noise," ",i," ",feature," ",distance)

                val_dict[Noise][i][feature][distance] = []

                if distance == 'Mahalanobis':
                    epsilon=1e-6
                    covariance_matrix = np.cov(x_train.T, bias=True)
                    covariance_matrix += epsilon * np.eye(covariance_matrix.shape[0])

                    model1 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]), metric= 'mahalanobis' ,metric_params={'VI': np.cov(np.linalg.inv(covariance_matrix), rowvar=False)})
                    model1.fit(x_train,y_train)
                    val_dict[Noise][i][feature][distance].append(model1.score(x_aug,y_aug))

                else:

                    

                    model1 = KNeighborsClassifier(n_neighbors=int(n_neig[ind]),  metric= get_metric(distance))
                    model1.fit(x_train,y_train)
                    val_dict[Noise][i][feature][distance].append(model1.score(x_aug,y_aug))


for Noise in Noise_levels:

    for feature in features:

        for distance in mets:

            a = []
   
            for i in range(10):

                a.append(val_dict[Noise][i][feature][distance][0])
             
            scor_att[feature][distance].append(np.mean(a))
          

import matplotlib.pyplot as plt

for feature in features:

    plt.title("Attribute Noise (" + feature + ")")
    plt.xlabel("Noise)")
    plt.ylabel("Avg accuracy")

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    marker = ['+','o','*','x','.']

    for i, met in enumerate(mets):
        color = colors[i % len(colors)]


        plt.scatter(Noise_levels, scor_att[feature][met], label=met, color=color, marker = marker[i])
        plt.plot(Noise_levels, scor_att[feature][met], color=color)
    plt.legend()
    plt.savefig("samplenoise_LSA.png")
    # plt.show()


file_path = 'samplenoise(LSA).json'
Noise = {"Attribute":scor_att}
with open(file_path, 'w') as f:
    json.dump(Noise, f, indent = 3)