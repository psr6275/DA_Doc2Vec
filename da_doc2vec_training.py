#from gensim.models.doc2vec import *
from DA_doc2vec.doc2vec_DA import *
import pickle
import re
import numpy as np
import pandas as pd
import nltk.corpus as nc
import random
from IPython import embed
data_path = '../data/'
stops = set(nc.stopwords.words("english"))

src = "kitchen"
tgt = "books"
"""
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
"""
f = open(data_path+'amazon_source_'+src+'_target_'+tgt+'_shuffled.pickle','rb')
aa = pickle.load(f)
f.close()
total_text = aa['total_text']
total_true_label = aa['true_label']
total_st_label = aa['st_label']
documents = []
for document in total_text:
    temp = document.replace('!','.').replace('?','.').replace(';','.').replace(':','.').replace('\n','.').strip()
    documents.append(temp.split('.'))

sentences = []
for uid, doc in enumerate(documents):
    for sen in doc:
        sen.lower()
        sen = re.sub("[^a-zA-Z]"," ",sen)
        sentence = TaggedDocument(words = sen.split(),tags = ['DOC_%s_st_%d_sent_%d'%(uid,total_st_label[uid],total_true_label[uid])])
        sentences.append(sentence)

print("length of sentences = ",len(sentences))
print(sentences[0])

del documents

d_size = 200

model_da_dbow = Doc2Vec(sentences,st_label = total_st_label,dbow_words = 1,comment=1,size = d_size, window = 3, min_count = 5, workers = 30, dm=0,iter=20)
file_name = data_path+'doc2vec_source_'+src+'_target_'+tgt
model_da_dbow.save(file_name+'.da_dbow')
doctag = model_da_dbow.docvecs.doctag_syn1
doctag2 = model_da_dbow.docvecs.doctag_syn0

if model_da_dbow.negative:
    wordtag = model_da_dbow.syn1neg
else:
    wordtag = model_da_dbow.syn1

word_s = model_da_dbow.wv.syn0_s
word_t = model_da_dbow.wv.syn0_t
doc2vec = {'wordtag':wordtag,'st_label':total_st_label,'docvec0':doctag2,'src_word':word_s,'tgt_word':word_t,'true_label':total_true_label,'docvec':doctag}
embed()

f = open(file_name+'_da_dbow_data.pickle','wb')
pickle.dump(doc2vec,f)
f.close()

