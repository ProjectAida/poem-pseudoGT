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

def gen_pgt(save_name, var_save_name, disp=False):
    # tr_fnames
    # te_fnames
    # tr_lbs
    # te_lbs
    # new_vis_feats
    # txt_feats


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


    # In[26]:


    outcomes = {}
    exp_names = ['TextBase', 'VisBase', 'TextVisBase']
    exp_cates = ['txt', 'vis', 'txtvis']
    te_data_sets = [test_txt_ft, test_vis_ft, test_txtvis_ft]
    model_archs = [TxtNet, VisNet, TxtVisNet]
    models = []
    for m, exp_name in zip(model_archs, exp_names):
        pth = exp_name + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        model = m()
        model.load_state_dict(torch.load(pth))
        models.append(model)

    # In[28]:

    for exp_name, exp_cate, model, te_data in zip(exp_names, exp_cates, models, te_data_sets):
        outcomes[exp_name] = runVerify(exp_cate, model, te_data, test_lbs)

    # In[29]:

    if(disp):
        for exp_name in exp_names:
            print(exp_name)
            print('test')
            print('acc,prec,rec,f1')
            for rslt in outcomes[exp_name]['test']:
                print(','.join([str(it) for it in rslt]))


    # <h3>Pseudo-GT

    # In[31]:


    clf_vis = models[1]
    clf_txt = models[0]
    clf_vis_txt = models[2]

    # get test set proba & pred
    with torch.no_grad():
        data = torch.from_numpy(np.array(test_vis_ft)).type(torch.float)
        outputs = clf_vis(data)
        prob_y_vis = outputs.detach().numpy()
        pred_y_vis = torch.argmax(outputs, axis=-1).detach().numpy().squeeze()
    with torch.no_grad():
        data = torch.from_numpy(np.array(test_txt_ft)).type(torch.float)
        outputs = clf_txt(data)
        prob_y_txt = outputs.detach().numpy()
        pred_y_txt = torch.argmax(outputs, axis=-1).detach().numpy().squeeze()
    with torch.no_grad():
        data = torch.from_numpy(np.array(test_txtvis_ft)).type(torch.float)
        outputs = clf_vis_txt(data)
        prob_y_vis_txt = outputs.detach().numpy()
        pred_y_vis_txt = torch.argmax(outputs, axis=-1).detach().numpy().squeeze()
        


    # In[32]:


    # get unknow set proba & pred
    with torch.no_grad():
        data = torch.from_numpy(np.array(unknow_vis_ft)).type(torch.float)
        outputs = clf_vis(data)
        prob_un_vis = outputs.detach().numpy()
        pred_un_vis = prob_un_vis.squeeze()
    with torch.no_grad():
        data = torch.from_numpy(np.array(unknow_txt_ft)).type(torch.float)
        outputs = clf_txt(data)
        prob_un_txt = outputs.detach().numpy()
        pred_un_txt = prob_un_txt.squeeze()
    with torch.no_grad():
        data = torch.from_numpy(np.array(unknow_txtvis_ft)).type(torch.float)
        outputs = clf_vis_txt(data)
        prob_un_vis_txt = outputs.detach().numpy()
        pred_un_vis_txt = prob_un_vis_txt.squeeze()


    # In[33]:


    # 3 model combine layer
    proba_txt_vis_sum_list = [((a) + (b) + (c))/3 
                              for a, b, c 
                              in zip(pred_un_txt, pred_un_vis, pred_un_vis_txt)]


    # In[34]:


    # 3 model combine
    lb = preprocessing.LabelBinarizer()
    lb.fit([0,1])
    proba_txt_vis_sum_pred = lb.inverse_transform(np.array(proba_txt_vis_sum_list))
    cm = confusion_matrix(unknow_lbs, proba_txt_vis_sum_pred)
    if(disp):
        print(cm)
        print('tn','fp','fn','tp')
        print(cm.ravel())
        print('tn','fp','fn','tp')
        print(cm.ravel())
        print('acc', (cm.ravel()[0]+cm.ravel()[3])/(cm.ravel().sum()))
        print('precision')
        print(cm.ravel()[3]/(cm.ravel()[3]+cm.ravel()[1]))
        print('recall')
        print(cm.ravel()[3]/(cm.ravel()[3]+cm.ravel()[2]))


    # In[36]:


    with open('pseudoGT_unknow_14000_pgt_no160train_'+save_name+'_'+var_save_name+'.csv', 'w+') as f:
        for idx, fn in enumerate(unknow_fns):
            if unknow_lbs[idx] == 1:
                fnp = '8464_true_ori/'+fn
            elif unknow_lbs[idx] == 0:
                fnp = '8464_false_ori/'+fn
            f.write(fnp+',('+'|'.join([str(it) for it in proba_txt_vis_sum_list[idx]])+')\n')
            

