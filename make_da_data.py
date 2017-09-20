import pickle
import numpy as np
import pandas as pd

data_path = "../data/"
category = ['books','dvd','electronics','kitchen']
#category = ["kitchen"]
Dics = {}
#labels = {}
selc_num = 3000
for cate in category:
    f = open(data_path+cate+'_noreplication_text.pickle','rb')
    text = pickle.load(f)
    f.close()
    f = open(data_path+cate+'_noreplication_label.pickle','rb')
    label = pickle.load(f)
    f.close()
    label = np.array(label)
    text = np.array(text)
    posi_idx = label>3
    nega_idx = label<3
    posi_text = list(text[posi_idx])
    nega_text = list(text[nega_idx])
    min_len = min(len(posi_text),len(nega_text))
    print(len(posi_text),len(nega_text))
    ttexts = posi_text[:min_len]+nega_text[:min_len]
    print(ttexts[-1])
    ttext =[]
    for txt in ttexts:
       ttext.append(txt.lower())
    print(ttext[-1])
    llabel = [1]*min_len + [-1]*min_len
    ttext = np.array(ttext)
    llabel = np.array(llabel)
    aa = np.random.choice(2*min_len,2*selc_num)
    trainx = list(ttext[aa[:selc_num]])
    trainy = list(llabel[aa[:selc_num]])
    testx = list(ttext[aa[selc_num:]])
    testy = list(llabel[aa[selc_num:]])
    dic = {'trainx':trainx,'trainy':trainy,'testx':testx,'testy':testy}
    Dics[cate] = dic
    #print(df)

f = open(data_path+'amazon_domain_adaptation_dictionary_data.pickle','wb')
pickle.dump(Dics,f)
f.close()
