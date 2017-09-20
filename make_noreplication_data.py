import pickle
import numpy as np
import pandas as pd

data_path = "../data/"
category = ['books','dvd','electronics','kitchen']
#category = ["kitchen"]
texts = {}
labels = {}
for cate in category:
    f = open(data_path+cate+'_processed_unlabeled.txt','rb')
    text = pickle.load(f)
    f.close()
    f = open(data_path+cate+'_processed_unlabeled_label.txt','rb')
    label = pickle.load(f)
    f.close()
    temp_text = []
    for i,tx in enumerate(text):
        tx += str(int(label[i]))
        temp_text.append(tx)
    print(len(temp_text))
    temp_text = list(set(temp_text))
    print(len(temp_text))
    print(temp_text[-1])
    text = []
    label = []
    for tt in temp_text:
        text.append(tt[:-1])
        label.append(int(tt[-1]))
    print(text[-1])
    print(type(label[-1]))
    f = open(data_path+cate+'_noreplication_text.pickle','wb')
    pickle.dump(text,f)
    f.close()
    f = open(data_path+cate+'_noreplication_label.pickle','wb')
    pickle.dump(label,f)
    f.close()
