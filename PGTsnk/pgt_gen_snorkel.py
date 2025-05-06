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

import snorkel
from snorkel.labeling import labeling_function
from snorkel.labeling import PandasLFApplier
from snorkel.labeling import LFAnalysis
from snorkel.labeling.model import MajorityLabelVoter
from snorkel.labeling.model import LabelModel

from pgt_naive_txtvis import TxtVisNet, TxtNet, VisNet

# In[4]:


class Snorkel_PGT:
    def __init__(self, train_csv, test_csv, save_name, var_save_name, disp=False):
        self.disp = disp
        
        self.train_csv = train_csv
        self.test_csv = test_csv
        self.save_name = save_name
        self.var_save_name = var_save_name

        pth = "TextBase" + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        self.textnet = TxtNet()
        self.textnet.load_state_dict(torch.load(pth))
        
        pth = "VisBase" + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        self.visnet = VisNet()
        self.visnet.load_state_dict(torch.load(pth))
        
        pth = "TextVisBase" + "_pgt_160sample_"+save_name+"_"+var_save_name+".pt"
        self.txtvisnet = TxtVisNet()
        self.txtvisnet.load_state_dict(torch.load(pth))
        
        self._prep_fts(train_csv, test_csv)
        
        self.df_train = pd.DataFrame({'vis' : self.train_vis_ft,
                                      'txt' : self.train_txt_ft,
                                      'vis_txt' : self.train_txtvis_ft,
                                      'fname' : self.tr_fnames,
                                      'lb' : self.tr_lbs,
                                      'txtbase' : self.textnet,
                                      'visbase' : self.visnet,
                                      'txtvisbase' : self.txtvisnet
                                      })

        self.df_test = pd.DataFrame({'vis' : self.test_vis_ft,
                                     'txt' : self.test_txt_ft,
                                     'vis_txt' : self.test_txtvis_ft,
                                     'fname' : self.te_lbs,
                                     'lb' : self.te_lbs,
                                     'txtbase' : self.textnet,
                                     'visbase' : self.visnet,
                                     'txtvisbase' : self.txtvisnet
                                    })
        
        self.ABSTAIN = -1
        self.NONPOEM = 0
        self.POEM = 1
        
    def _prep_fts(self, train_csv, test_csv):
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
        
        for fn in new_vis_feats:
            if len(new_vis_feats[fn]) != 20:
                print('nb_feat wrong -- get_naive_model: ', fn)

        self.new_vis_feats = new_vis_feats
        # In[10]:

        txt_feats = {}
        with open('aida-17k-ms-textfeat.csv', 'r') as f:
            for l in f:
                fts = l.strip().split(',')
                fn = fts[0][fts[0].rfind('/')+1:]
                txt_feats[fn] = [float(it) for it in fts[1:]]

        self.txt_feats = txt_feats
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
            
        self.test_txt_ft = test_txt_ft
        self.test_vis_ft = test_vis_ft
        self.test_txtvis_ft = test_txtvis_ft
        
        self.train_txt_ft = train_txt_ft
        self.train_vis_ft = train_vis_ft
        self.train_txtvis_ft = train_txtvis_ft
        
        self.tr_fnames = tr_fnames
        self.te_fnames = te_fnames
        self.tr_lbs = tr_lbs
        self.te_lbs = te_lbs
        
        lfs = [self.lf_txt_ft, 
               self.lf_vis_ft,
               self.lf_txtvis_ft]
        
        self.applier = PandasLFApplier(lfs=lfs)
        
        return
        

    @labeling_function()
    def lf_txt_ft(x):
        # Return opinion of TextNet, otherwise ABSTAIN
        with torch.no_grad():
            data = torch.from_numpy(np.array([x.txt])).type(torch.float)
            pred = torch.argmax(x.txtbase(data), dim=-1).detach().cpu().numpy().squeeze()
        
        if pred == 0 or pred == 1:
            return pred
        else:
            return self.ABSTAIN
        
    @labeling_function()
    def lf_vis_ft(x):
        # Return opinion of TextNet, otherwise ABSTAIN
        with torch.no_grad():
            data = torch.from_numpy(np.array([x.vis])).type(torch.float)
            pred = torch.argmax(x.visbase(data), dim=-1).detach().cpu().numpy().squeeze()
        
        if pred == 0 or pred == 1:
            return pred
        else:
            return self.ABSTAIN
        
    @labeling_function()
    def lf_txtvis_ft(x):
        # Return opinion of TextNet, otherwise ABSTAIN
        with torch.no_grad():
            data = torch.from_numpy(np.array([x.vis_txt])).type(torch.float)
            pred = torch.argmax(x.txtvisbase(data), dim=-1).detach().cpu().numpy().squeeze()
        
        if pred == 0 or pred == 1:
            return pred
        else:
            return self.ABSTAIN
        
    def get_snorkel_model(self):
        
        L_train = self.applier.apply(df=self.df_train)
        label_model = LabelModel(cardinality=2, verbose=self.disp)
        label_model.fit(L_train=L_train, n_epochs=5000, seed=1)
        self.label_model = label_model
        save_path = 'pseudoGT_snk_model_'+self.save_name+'_'+self.var_save_name+'.pt'
        self.label_model.save(save_path)
        
        return label_model
        
    def test_snorkel_model(self):
        print('test: <br/>')

        L_test = self.applier.apply(df=self.df_test)
        Y_test = self.df_test.lb.values

        label_model_acc = self.label_model.score(L=L_test, Y=Y_test, tie_break_policy="random")["accuracy"]
        print(f"{'Label Model Accuracy:':<25} {label_model_acc * 100:.2f}%" + ' <br/>')
        print(str(self.label_model.score(L=L_test, Y=Y_test, tie_break_policy="random", metrics=['precision'])) + ' <br/>')
        print(str(self.label_model.score(L=L_test, Y=Y_test, tie_break_policy="random", metrics=['recall'])) + ' <br/>')
        print(str(self.label_model.score(L=L_test, Y=Y_test, tie_break_policy="random", metrics=['f1'])) + ' <br/>')
              
        
    
    def prep_unknow_df(self, unknow_csv = 'pseudoGT_unknow_14000.csv'):
        unknow_fnames = []
        unknow_lbs = []
        with open(unknow_csv, 'r') as f:
            for l in f:
                info = l.strip().split(',')
                fp = info[0]
                fp = fp.replace('/','\\')
                fname = fp[fp.rfind('\\')+1:]
                unknow_fnames.append(fname)
                unknow_lbs.append(int(info[1]))
        unknow_txt_ft = []
        unknow_vis_ft = []
        unknow_txtvis_ft = []
        for fn in unknow_fnames:
            unknow_txt_ft.append(self.txt_feats[fn])
            unknow_vis_ft.append(self.new_vis_feats[fn])
            unknow_txtvis_ft.append(self.txt_feats[fn]+self.new_vis_feats[fn])
              
        self.df_unknow = pd.DataFrame({'vis' : unknow_vis_ft,
                                      'txt' : unknow_txt_ft,
                                      'vis_txt' : unknow_txtvis_ft,
                                      'fname' : unknow_fnames,
                                      'lb' : unknow_lbs,
                                      'txtbase' : self.textnet,
                                      'visbase' : self.visnet,
                                      'txtvisbase' : self.txtvisnet
                                      })
        return self.df_unknow
    
    def get_pgt(self):

        L_unknow = self.applier.apply(df=self.df_unknow)
              
        probs_unknow = self.label_model.predict_proba(L_unknow)
        return probs_unknow


# In[5]:


def gen_pgt(train_csv, test_csv, save_name, var_save_name, disp=False):
    
    # train snorkel model
    pgtgen = Snorkel_PGT(train_csv, test_csv, save_name, var_save_name)
    snorkel_model = pgtgen.get_snorkel_model()
    
    # get pgt
    unknow_df = pgtgen.prep_unknow_df()
    prob_unknow = pgtgen.get_pgt()
    
    unknow_fns = unknow_df.fname.values
    unknow_lbs = unknow_df.lb.values
    
    if(disp):
        for idx, fn in enumerate(unknow_fns):
            if unknow_lbs[idx] == 1:
                fnp = '8464_true_ori/'+fn
            elif unknow_lbs[idx] == 0:
                fnp = '8464_false_ori/'+fn
            print(fnp+',('+'|'.join([str(it) for it in prob_unknow[idx]])+')\n')

    with open('pseudoGT_unknow_14000_pgt_no160train_'+save_name+'_'+var_save_name+'.csv', 'w+') as f:
        for idx, fn in enumerate(unknow_fns):
            if unknow_lbs[idx] == 1:
                fnp = '8464_true_ori/'+fn
            elif unknow_lbs[idx] == 0:
                fnp = '8464_false_ori/'+fn
            f.write(fnp+',('+'|'.join([str(it) for it in prob_unknow[idx]])+')\n')
            


# In[ ]:




