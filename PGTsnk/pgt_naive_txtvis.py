
import pandas as pd
import re
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import random
import copy
import warnings
warnings.filterwarnings("ignore")



# In[11]:


# model

class TxtVisNet(nn.Module):
    def __init__(self):
        super(TxtVisNet, self).__init__()
        
        self.fc1 = nn.Linear(31, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), -1)
        return x
    

# model
class TxtNet(nn.Module):
    def __init__(self):
        super(TxtNet, self).__init__()
        
        self.fc1 = nn.Linear(11, 100)
        self.fc2 = nn.Linear(100, 2)
        
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), -1)
        return x
    
# model

class VisNet(nn.Module):
    def __init__(self):
        super(VisNet, self).__init__()
        
        self.fc1 = nn.Linear(20, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), -1)
        return x
    

# In[14]:


def runExp(exp_cate, model_arch, tr_data, tr_lb, te_data, te_lb, disp = False):
    if(disp):
        print(exp_cate)
    rslts = {}
    rslts['train'] = []
    rslts['test'] = []
    rslts['valid'] = []
    rslts['test_cm'] = []
    rslts['train_loss'] = []
    rslts['valid_loss'] = []
    rslts['model'] = None

    training_reach = False
    exp = exp_cate
    epoch = 1
    harm_f1 = np.nan
    t_f1 = np.nan
    f1 = np.nan
    patience=5
    min_delta=0
    counter = 0
    best_loss = 100000.0
    batch_size = 18
    while True:


        nb_batch = len(tr_data)//batch_size
        
        clf = model_arch()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(clf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        if(disp):
            if epoch % 10 == 0:
                print('current epoch', epoch)
#             print(f1)
        t_preds = np.empty(0)
        t_lbs = np.empty(0)
        epoch_loss = 0

        # training set shuffling 

        shf_id = list(range(len(tr_data)))
        random.Random(1).shuffle(shf_id)
        sf_tr_data = []
        sf_tr_lbs = []

        for idx in shf_id:
            sf_tr_data.append(tr_data[idx])
            sf_tr_lbs.append(tr_lb[idx])
        for i in range(nb_batch):
            if i == nb_batch - 1:
                data = torch.from_numpy(np.array(sf_tr_data[i*batch_size:])).type(torch.float)
                labels = torch.from_numpy(np.array(sf_tr_lbs[i*batch_size:])).type(torch.long)
            else:
                data = torch.from_numpy(np.array(sf_tr_data[i*batch_size:(i+1)*batch_size])).type(torch.float)
                labels = torch.from_numpy(np.array(sf_tr_lbs[i*batch_size:(i+1)*batch_size])).type(torch.long)
            optimizer.zero_grad()
            outputs = clf(data)
            loss = criterion(outputs.float(), labels)
            loss.backward()
            optimizer.step()
            predicted = torch.argmax(outputs, axis=-1)
            epoch_loss += loss.detach().numpy().squeeze()
            t_preds = np.hstack((t_preds, predicted.detach().numpy().squeeze()))
            t_lbs = np.hstack((t_lbs, labels))

        # verify
        with torch.no_grad():
            t_cm = confusion_matrix(t_lbs, t_preds)
            t_tn, t_fp, t_fn, t_tp = t_cm.ravel()
            t_acc = (t_tn + t_tp) / (t_tn + t_fp + t_fn + t_tp)
            t_prec = t_tp / (t_tp + t_fp)
            t_rec = t_tp / (t_tp + t_fn)
            t_f1 = 2 * (t_prec * t_rec) / (t_prec + t_rec)
            
            preds = np.empty(0)
            lbs = np.empty(0)

            data = torch.from_numpy(np.array(te_data)).type(torch.float)
            labels = torch.from_numpy(np.array(te_lb)).type(torch.long)
            outputs = clf(data)
            predicted = torch.argmax(outputs, axis=-1)

            preds = np.hstack((preds, predicted.detach().numpy().squeeze()))
            lbs = np.hstack((lbs, labels))

            val_loss = criterion(outputs.float(), labels)
            
            cm = confusion_matrix(lbs, preds)
            tn, fp, fn, tp = cm.ravel()
            acc = (tn + tp) / (tn + fp + fn + tp)
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2 * (prec * rec) / (prec + rec)
            
            harm_f1 = 2 * (t_f1 * f1) / (t_f1 + f1)
    
            if(disp):
                print(f1)
            if not training_reach:
                if f1 > 0.7 and t_f1 > 0.7:
                    training_reach = True
                else:
                    counter = 0
            if best_loss - val_loss > min_delta:
                best_loss = val_loss
                counter = 0
                rslts['test_cm'] = cm
                rslts['test'] = [acc, prec, rec, f1]
                rslts['train'] = [t_acc, t_prec, t_rec, t_f1]
                rslts['model'] = clf

            elif best_loss - val_loss < min_delta:
                if harm_f1 > 0.7:
                    counter += 1
                    if counter >= patience:
                        break
        epoch += 1
    
    return rslts

# In[12]:

def get_naive_model(train_csv, test_csv, save_name, var_save_name, disp=False):
    # tr_fnames
    # te_fnames
    # tr_lbs
    # te_lbs
    # new_vis_feats
    # txt_feats

    tr_fnames = []
    te_fnames = []
    tr_lbs = []
    te_lbs = []
    with open(train_csv, 'r') as f:
        for l in f:
            info = l.strip().split(',')
            fp = info[0]
            fp = fp.replace('/','\\')
            fname = fp[fp.rfind('\\')+1:]
            tr_fnames.append(fname)
            tr_lbs.append(int(info[1]))

    with open(test_csv, 'r') as f:
        for l in f:
            info = l.strip().split(',')
            fp = info[0]
            fp = fp.replace('/','\\')
            fname = fp[fp.rfind('\\')+1:]
            te_fnames.append(fname)
            te_lbs.append(int(info[1]))
            
    if(disp):
        print(len(tr_fnames))
        print(te_fnames[0])
        print(te_lbs[0])


    # In[9]:


    new_vis_feats = {}
    with open('visual/false_visual_feats.arff', 'r') as f:
        for l in f:
            fts = l.strip().split(',')
            new_vis_feats[fts[0]] = [float(it) for it in fts[1:-1]]
            
    with open('visual/true_visual_feats.arff', 'r') as f:
        for l in f:
            fts = l.strip().split(',')
            new_vis_feats[fts[0]] = [float(it) for it in fts[1:-1]]
            
    if(disp):
        print(len(new_vis_feats))
    for fn in new_vis_feats:
        if len(new_vis_feats[fn]) != 20:
            print('nb_feat wrong -- get_naive_model', fn)


    # In[10]:


    txt_feats = {}
    with open('aida-17k-ms-textfeat.csv', 'r') as f:
        for l in f:
            fts = l.strip().split(',')
            fn = fts[0][fts[0].rfind('/')+1:]
            txt_feats[fn] = [float(it) for it in fts[1:]]

    if(disp):
        print(txt_feats[txt_feats.keys()[-1]])



    # In[13]:


    test_txt_ft = []
    test_vis_ft = []
    test_txtvis_ft = []
    for fn in te_fnames:
        test_txt_ft.append(txt_feats[fn])
        test_vis_ft.append(new_vis_feats[fn])
        test_txtvis_ft.append(txt_feats[fn]+new_vis_feats[fn])

    train_txt_ft = []
    train_vis_ft = []
    train_txtvis_ft = []

    for fn in tr_fnames:
        train_txt_ft.append(txt_feats[fn])
        train_vis_ft.append(new_vis_feats[fn])
        train_txtvis_ft.append(txt_feats[fn]+new_vis_feats[fn])

        
    if(disp):
        print(len(train_txt_ft), len(train_vis_ft), len(train_txtvis_ft))
        print(len(test_txt_ft), len(test_vis_ft), len(test_txtvis_ft))

    # In[15]:


    outcomes = {}
    exp_names = ['TextBase', 'VisBase', 'TextVisBase']
    exp_cates = ['txt', 'vis', 'txtvis']
    model_archs = [TxtNet, VisNet, TxtVisNet]
    tr_data_sets = [train_txt_ft, train_vis_ft, train_txtvis_ft]
    te_data_sets = [test_txt_ft, test_vis_ft, test_txtvis_ft]





    # In[19]:


    for exp_name, exp_cate, model_arch, tr_data, te_data in zip(exp_names, exp_cates, model_archs, tr_data_sets, te_data_sets):
        outcomes[exp_name] = runExp(exp_cate, model_arch, tr_data, tr_lbs, te_data, te_lbs)


    # In[20]:

    if(disp):
        for exp_name in exp_names:
            print(exp_name)
            print('train')
            print('acc,prec,rec,f1')
            print(','.join([str(it) for it in outcomes[exp_name]['train']]))
            print(','.join([str(it) for it in outcomes[exp_name]['test']]))




    # In[24]:


    for exp_name in exp_names:
        if(disp):
            print(exp_name)
        save_path = exp_name + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        torch.save(outcomes[exp_name]['model'].state_dict(), save_path)


# In[ ]:




