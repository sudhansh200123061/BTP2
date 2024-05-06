
import pandas as pd
import numpy as np
from eda import *

def gen_eda(train_orig, alpha, num_aug,ind):

    output_file = "output.txt"
    train_orig = train_orig[["number_label","doc_text"]]
    train_orig = train_orig.reset_index(drop=True)
    df = train_orig.copy()

    df = df.iloc[ind]
    
    # with open("clean.txt",'w') as f:
    #     for _ in range(len(df)):
    #         line = str(df.iloc[_]["number_label"]) + "\t" + str(df.iloc[_]["doc_text"]) + "\n"
    #         f.write(line)
    # f.close()

    # writer = open(output_file, 'w')
    # lines = open("clean.txt", 'r').readlines()
    data = {}
    data['number_label'] = []
    data['doc_text'] = []
    for i in range(len(df)):
        line = df.iloc[[i]]
        
        label = int(line['number_label'].iloc[0])
        sentence = str(line['doc_text'].iloc[0])
        aug_sentences = eda(sentence, alpha, num_aug=num_aug)

        for aug_sentence in aug_sentences:
            data['number_label'].append(label)
            data['doc_text'].append(aug_sentence)

    dt = pd.DataFrame(data)
    dt = pd.concat([dt,train_orig.drop(ind)])
    # dt.to_csv("aug_data.csv", index=False)
    print("augment complete")

    return dt.reset_index(drop=True)
    
