from __future__ import division
from cmath import sqrt
import nltk
import numpy
import pandas as pd
from collections import defaultdict
import json

from sklearn.decomposition import TruncatedSVD

def text_generator(data):
    for doc in data:
        yield doc
def cnt(x,dict1):
    return dict1[x]

def TFIDF(document,newcollection,unique_terms):
    word_tfidf = []
    for word in unique_terms[:300]:
        word_tfidf.append(newcollection.tf_idf(word, document))
    return word_tfidf

def lsa_(file):
    final_df = file


    documents = text_generator(final_df['doc_text'].apply(str).tolist())
    labels = final_df['number_label']

    texts = [nltk.Text(nltk.word_tokenize(raw)) for raw in documents]

    # Empty list to hold text documents.
    documents = []

    # Iterate through the directory and build the collection of texts for NLTK.
    dict1 = {}
    dict1 = defaultdict(lambda: 0, dict1)
    for i, text in enumerate(texts):
        tokens = nltk.word_tokenize(str(text))
        stemmed_tokens = nltk.Text(tokens)
        for x in tokens:
            dict1[x] += 1
        documents.append(stemmed_tokens)  # Update the texts list with the modified text

    print("Prepared ", len(documents), " documents...")
    print("They can be accessed using texts[0] - texts[" + str(len(documents)-1) + "]")

    # Load the list of texts into a TextCollection object.
    collection = nltk.TextCollection(documents)
    print("Created a collection of", len(collection), "terms.")

    # Get a list of unique terms
    unique_terms = list(set(collection))

    unique_terms.sort(key=lambda x:cnt(x,dict1), reverse=True)
    print("Unique terms found: ", len(unique_terms))
    newlist = []
    for x in collection:
        if x in unique_terms[:300]:
            newlist.append(x)

    newcollection = nltk.TextCollection(newlist)

    # Function to create a TF*IDF vector for one document.


    # And here we actually call the function and create our array of vectors.
    document_term_matrix = [numpy.array(TFIDF(f,newcollection,unique_terms)) for f in texts if len(f) != 0]

    # Apply Latent Semantic Analysis (LSA) using Truncated SVD
    n_topics = 50  # You can choose the number of topics
    lsa_model = TruncatedSVD(n_components=n_topics, random_state=42)
    lsa_topic_matrix = lsa_model.fit_transform(document_term_matrix)

    # Display the document-topic matrix details
    print("Document-Term Matrix for lsa created ..")

    return lsa_topic_matrix,labels