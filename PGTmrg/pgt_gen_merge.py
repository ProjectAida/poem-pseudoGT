import pandas as pd
import re
import time
import random as r
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
import random
import copy

from pgt_naive_txtvis_merge import TxtVisNet, TxtNet, VisNet

# In[25]:


def runVerify(exp_cate, model, te_data, te_lb, disp=False):

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

    exp = exp_cate
    counter = 0
    batch_size = 100
    criterion = nn.CrossEntropyLoss()
    nb_batch = len(te_lb) // batch_size
    preds = np.empty(0)
    lbs = np.empty(0)
    for i in range(nb_batch):

        # verify
        with torch.no_grad():
            if i == nb_batch - 1:
                data = torch.from_numpy(np.array(te_data[i*batch_size:])).type(torch.float)
                labels = torch.from_numpy(np.array(te_lb[i*batch_size:])).type(torch.long)
            else:
                data = torch.from_numpy(np.array(te_data[i*batch_size:(i+1)*batch_size])).type(torch.float)
                labels = torch.from_numpy(np.array(te_lb[i*batch_size:(i+1)*batch_size])).type(torch.long)
            
            outputs = model(data)
            outputs = F.softmax(outputs, -1)
            predicted = []
            for o in outputs:
                o = o.numpy()
                if max(o) < 0.5:
                    predicted.append(2)
                else:
                    predicted.append(np.argmax(o))
            predicted = np.array(predicted)
            
            preds = np.hstack((preds, predicted))
            lbs = np.hstack((lbs, labels))
            val_loss = criterion(outputs.float(), labels)
    
    cm = confusion_matrix(lbs, preds, labels=[0,1])
    if(disp):
        print(cm)
    tn, fp, fn, tp = cm.ravel()
    acc = (tn + tp) / (tn + fp + fn + tp)
    prec = tp / (tp + fp)
    rec = tp / (tp + fn)
    f1 = 2 * (prec * rec) / (prec + rec)
    
    rslts['test'].append([acc, prec, rec, f1])

    return rslts


# In[11]:


class MergeNet(nn.Module):
    def __init__(self):
        super(MergeNet, self).__init__()
        
        self.fc1 = nn.Linear(6, 100)
        self.fc2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.fc1(F.leaky_relu(x)))
        x = F.softmax(self.fc2(x), -1)
        return x


# In[ ]:


def train_merge(model_arch, tr_data, tr_lb, te_data, te_lb, disp = False):

    rslts = {}
    rslts['train'] = []
    rslts['test'] = []
    rslts['valid'] = []
    rslts['test_cm'] = []
    rslts['train_loss'] = []
    rslts['valid_loss'] = []
    rslts['model'] = None

    training_reach = False

    epoch = 1
    harm_f1 = np.nan
    t_f1 = np.nan
    f1 = np.nan
    patience=5
    min_delta=0
    counter = 0
    best_loss = 100000.0
    batch_size = 16
    while True:

        nb_batch = len(tr_data)//batch_size
        
        clf = model_arch()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(clf.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)
        if(disp):
            if epoch % 10 == 0:
                print('current epoch', epoch)
        t_preds = np.empty(0)
        t_lbs = np.empty(0)
        epoch_loss = 0

        # training set shuffling 

        shf_id = list(range(len(tr_data)))
        random.shuffle(shf_id)
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


# In[1]:


def get_merge_model(train_csv, test_csv, save_name, var_save_name, disp=False):
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
            print('nb_feat wrong -- get_naive_model: ', fn)


    # In[10]:


    txt_feats = {}
    with open('aida-17k-ms-textfeat.csv', 'r') as f:
        for l in f:
            fts = l.strip().split(',')
            fn = fts[0][fts[0].rfind('/')+1:]
            txt_feats[fn] = [float(it) for it in fts[1:]]

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
        
    exp_names = ['TextBase', 'VisBase', 'TextVisBase']
    exp_cates = ['txt', 'vis', 'txtvis']
    model_archs = [TxtNet, VisNet, TxtVisNet]
    tr_data_sets = [train_txt_ft, train_vis_ft, train_txtvis_ft]
    te_data_sets = [test_txt_ft, test_vis_ft, test_txtvis_ft]

    models = []
    for m, exp_name in zip(model_archs, exp_names):
        pth = exp_name + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        model = m()
        model.load_state_dict(torch.load(pth))
        models.append(model)

    # get train set logits
    train_logits = []
    for i in range(len(tr_data_sets)):
        clf = models[i]
        with torch.no_grad():
            data = torch.from_numpy(np.array(tr_data_sets[i])).type(torch.float)
            outputs = clf(data)
            train_logits.append(outputs.detach().numpy())
    train_logits = np.hstack(train_logits)
    
    # get test set logits
    test_logits = []
    for i in range(len(te_data_sets)):
        clf = models[i]
        with torch.no_grad():
            data = torch.from_numpy(np.array(te_data_sets[i])).type(torch.float)
            outputs = clf(data)
            test_logits.append(outputs.detach().numpy())
    test_logits = np.hstack(test_logits)
    
    # train merge model
    rslt = train_merge(MergeNet, train_logits, tr_lbs, test_logits, te_lbs, disp = False)
    
    # save merge model:
    save_path = 'pseudoGT_merge_model_'+save_name+'_'+var_save_name+'.pt'
    
    return rslt['model']


# In[ ]:


def gen_pgt(train_csv, test_csv, save_name, var_save_name, disp=False):
    # tr_fnames
    # te_fnames
    # tr_lbs
    # te_lbs
    # new_vis_feats
    # txt_feats
    
    # train merge model
    merge_model = get_merge_model(train_csv, test_csv, save_name, var_save_name, disp)

    unknow_fnames = []
    te_fnames = []
    unknow_lbs = []
    te_lbs = []
    with open('pseudoGT_unknow_14000.csv', 'r') as f:
        for l in f:
            info = l.strip().split(',')
            fp = info[0]
            fp = fp.replace('/','\\')
            fname = fp[fp.rfind('\\')+1:]
            unknow_fnames.append(fname)
            unknow_lbs.append(int(info[1]))

    with open('pseudoGT_test_1700.csv', 'r') as f:
        for l in f:
            info = l.strip().split(',')
            fp = info[0]
            fp = fp.replace('/','\\')
            fname = fp[fp.rfind('\\')+1:]
            te_fnames.append(fname)
            te_lbs.append(int(info[1]))
            
    if(disp):
        print(len(unknow_fnames))
        print(te_fnames[0])
        print(te_lbs[0])

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
    # print(len(new_vis_feats[fts[0]]))
    for fn in new_vis_feats:
        if len(new_vis_feats[fn]) != 20:
            print('nb_feat wrong -- gen_pgt', fn)


    # In[9]:


    txt_feats = {}
    with open('aida-17k-ms-textfeat.csv', 'r') as f:
        for l in f:
            fts = l.strip().split(',')
            fn = fts[0][fts[0].rfind('/')+1:]
            txt_feats[fn] = [float(it) for it in fts[1:]]

    if(disp):
        print(len(txt_feats))


    test_txt_ft = []
    test_vis_ft = []
    test_txtvis_ft = []
    test_lbs = te_lbs
    test_fns = []
    for fn in te_fnames:
        test_txt_ft.append(txt_feats[fn])
        test_vis_ft.append(new_vis_feats[fn])
        test_txtvis_ft.append(txt_feats[fn]+new_vis_feats[fn])
        test_fns.append(fn)
    unknow_txt_ft = []
    unknow_vis_ft = []
    unknow_txtvis_ft = []
    unknow_fns = []
    for fn in unknow_fnames:
        unknow_txt_ft.append(txt_feats[fn])
        unknow_vis_ft.append(new_vis_feats[fn])
        unknow_txtvis_ft.append(txt_feats[fn]+new_vis_feats[fn])
        unknow_fns.append(fn)


    # In[20]:


    if(disp):
        print(len(test_fns), len(unknow_fns))


    # In[21]:


    if(disp):
        print(
              len(test_txt_ft),
              len(test_vis_ft),
              len(test_txtvis_ft),
              len(test_lbs),
              len(unknow_txt_ft),
              len(unknow_vis_ft),
              len(unknow_txtvis_ft),
              len(unknow_lbs)
             )     


    # In[32]:
    
    exp_names = ['TextBase', 'VisBase', 'TextVisBase']
    model_archs = [TxtNet, VisNet, TxtVisNet]

    models = []
    for m, exp_name in zip(model_archs, exp_names):
        pth = exp_name + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        model = m()
        model.load_state_dict(torch.load(pth))
        models.append(model)

    # get unknow set logits for pgt
    unknow_data_sets = [unknow_txt_ft, unknow_vis_ft, unknow_txtvis_ft]
    unknow_logits = []
    for i in range(len(unknow_data_sets)):
        clf = models[i]
        with torch.no_grad():
            data = torch.from_numpy(np.array(unknow_data_sets[i])).type(torch.float)
            outputs = clf(data)
            unknow_logits.append(outputs.detach().numpy())
    unknow_logits = np.hstack(unknow_logits)

    # get pgt using merge model 
    with torch.no_grad():
        data = torch.from_numpy(np.array(unknow_logits)).type(torch.float)
        outputs = merge_model(data)
        outputs = outputs.detach().numpy()
        
    with open('pseudoGT_unknow_14000_pgt_no160train_'+save_name+'_'+var_save_name+'.csv', 'w+') as f:
        for idx, fn in enumerate(unknow_fns):
            if unknow_lbs[idx] == 1:
                fnp = '8464_true_ori/'+fn
            elif unknow_lbs[idx] == 0:
                fnp = '8464_false_ori/'+fn
            f.write(fnp+',('+'|'.join([str(it) for it in outputs[idx]])+')\n')
            

