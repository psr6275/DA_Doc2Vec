from gensim.models.doc2vec import *
from gensim.models.word2vec import Word2Vec
import pickle
import re
import numpy as np
import pandas as pd
import nltk.corpus as nc
import random
data_path = '../data/'
stops = set(nc.stopwords.words("english"))

src = "kitchen"
tgt = "books"

f = open(data_path + 'amazon_domain_adaptation_dictionary_data.pickle','rb')
Dics = pickle.load(f)
f.close()
src_text = Dics[src]['trainx']
src_label = Dics[src]['trainy']
tgt_text = Dics[tgt]['trainx']
tgt_label = Dics[tgt]['trainy']
total_text = src_text + tgt_text
total_st_label = [1]*len(src_text) + [0]*len(tgt_text)
total_true_label = src_label + tgt_label
print(len(total_true_label))
print(len(total_text))
total_text = np.array(total_text)
print(total_text.shape)
total_st_label = np.array(total_st_label)
total_true_label = np.array(total_true_label)
aa =list(range(len(total_text)))
random.shuffle(aa)


total_text = total_text[aa].tolist()
total_st_label =total_st_label[aa]
total_true_label = total_true_label[aa]
#print(total_text)
f =open(data_path+'amazon_source_'+src+'_target_'+tgt+'_shuffled.pickle','wb') 
pickle.dump({'total_text':total_text,'st_label':total_st_label,'true_label':total_true_label},f)
f.close()

documents = []
for document in total_text:
    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n','.').strip()
    documents.append(temp.split('.'))
sentences2 = []
for doc in documents:
    for sen in doc:
        sen = sen.lower()
        sen = re.sub("[^a-zA-Z]"," ",sen)
        sentence = sen.split()
        sentences2.append(sentence)
sentences = []
for uid, doc in enumerate(documents):
    for sen in doc:
        sen = sen.lower()
        sen = re.sub("[^a-zA-Z]"," ",sen)
        sentence = TaggedDocument(words = sen.split(),tags = ['DOC_%s_st_%d_sent_%d'%(uid,total_st_label[uid],total_true_label[uid])])
        sentences.append(sentence)

print("length of sentences = ",len(sentences))
print(sentences[0])

del documents

d_size = 200

print("start to train the word vectors using word2vec")

model_word2vec = Word2Vec(sentences2,size = d_size, window=3, min_count=10, workers = 30, sg=1, iter=30)
model_word2vec.save(data_path+'word2vec_source_'+src+'_target_'+tgt+'.sg')
print("start to train document vectors using fixed word vectors")


model_dbow = Doc2Vec(sentences,size = d_size,dbow_words =1, window = 3, min_count = 10, workers = 30, dm=0,iter=30)
file_name = data_path+'doc2vec_source_'+src+'_target_'+tgt
model_dbow.save(file_name+'.dbow')
doctag = model_dbow.docvecs.doctag_syn0
doc2vec = {'st_label':total_st_label,'true_label':total_true_label,'docvec':doctag}


f = open(file_name+'_data.pickle','wb')
pickle.dump(doc2vec,f)
f.close()
