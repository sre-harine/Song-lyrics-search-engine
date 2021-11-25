from django.shortcuts import render
import pickle
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
nltk.download('punkt')
nltk.download('stopwords')
import re
from nltk.corpus import stopwords
stop_words=set(stopwords.words('english'))

def homepage(request):
    return render(request,'index.html')

def get_tokenized_list(doc_text):
    tokens= nltk.word_tokenize(doc_text)
    return tokens

def remove_stopwords(doc_text):
    cleaned_text = []
    for words in doc_text:
        if words not in stop_words:
            cleaned_text.append(words) 
    return cleaned_text

def word_stemmer(token_list):
    ps = nltk.stem.PorterStemmer()
    stemmed = []
    for words in token_list:
        stemmed.append(ps.stem(words))
    return stemmed

def func1(request):
    if(request.method=='POST'):
        queries = str(request.POST.get('search'))
        file = open("../music/lyrics_scraped.txt")
        read = file.read()
        file.seek(0)
        line = 1
        for word in read:
            if word == '\n':
                line += 1
        #print("Number of lines in file is: ", line)
        array = []
        for i in range(line):
            array.append(file.readline())

        vectorizerX =TfidfVectorizer()
        vectorizerX.fit(array)
        doc_vector =vectorizerX.transform(array)     

        queries_proccessed =[]
        tokens = get_tokenized_list(queries)
        doc_text= remove_stopwords(tokens)
        doc_text = word_stemmer(doc_text)
        doc_text = ' '.join(doc_text)
        queries_proccessed.append(doc_text)
        query_vectors =[]
        for query in queries_proccessed:
            query_vectors.append(vectorizerX.transform([query]))
        lyric={}
        for query_vector in query_vectors:
            cosinesimilarities =cosine_similarity(doc_vector, query_vector).flatten()
            related_docs_indices = cosinesimilarities.argsort()[:-11:-1]
            for i in range(0,10):
                names=re.findall('"([^"]*)"', array[related_docs_indices[i]][:50])
                lyr=array[related_docs_indices[i]][:3000]
                string=" "
                n=string.join(names)
                lyric[n]=lyr
        context={'songs':lyric}
    return render(request,'lyrics.html',context)
 

def func2(request):
    if(request.method=='POST'):
        lyrics=request.POST.get('song')
        n=re.findall('"([^"]*)"', lyrics)
        names=n[0]
        context={'names':names,'lyrics':lyrics}
    return render(request,'songlyrics.html',context)    