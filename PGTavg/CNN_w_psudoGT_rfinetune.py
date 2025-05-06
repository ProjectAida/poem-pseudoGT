# ---------------------------------------------
# data loader
# ---------------------------------------------

# In[1]:


# data loader
from __future__ import print_function, division
import os
import sys
import torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.distributions as distributions
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import pandas as pd
from skimage import io, transform
from skimage.color import rgb2gray
from sklearn.metrics import confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import pickle
import time
# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

from ..combine_pgt_gt import combine_pgt_gt
from pgt_naive_txtvis import get_naive_model
from pgt_gen import gen_pgt

torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# In[2]:

class Aida17kDataset(Dataset):
    """Aida-17k dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with labels.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.csv = pd.read_csv(csv_file, header=None, dtype=str)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # ori
        _ipath = self.csv.iloc[idx, 0]
        img_name = os.path.join(self.root_dir, _ipath)
        image = io.imread(img_name)
        if image.shape[-1] == 3:
            image = rgb2gray(image)-0.5
        else:
            image = (image / 255) - 0.5
        
        _lb = self.csv.iloc[idx, 1]
#         print(_lb)
        if '(' in _lb:
            _lb = _lb.replace('(', '').replace(')', '')
            _lb = _lb.split('|')
            _lb = [float(it) for it in _lb]
            label = np.array(_lb)
        else:
            label = np.zeros(2)
            label[int(_lb)] = 1.
    
        cls_label = np.array(label)
        sample = image, cls_label
        if self.transform:
            sample = self.transform(sample)

        return sample
    
class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample
        h, w = image.shape
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size
        new_h, new_w = int(new_h), int(new_w)
        image = transform.resize(image, (new_h, new_w))
        return image, label


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, label = sample

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = np.expand_dims(image, axis=2)
        image = image.transpose((2, 0, 1))
        return torch.from_numpy(image).type(torch.FloatTensor), torch.from_numpy(label).type(torch.FloatTensor)

# ---------------------------------------------
# model
# ---------------------------------------------

# In[6]:


class Classifier(nn.Module):
    def __init__(self, device = None):
        super(Classifier, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.drop = nn.Dropout()
        self.conv1 = nn.Conv2d(1, 64, 5, padding=(2,2))
        self.conv2 = nn.Conv2d(64, 128, 5, padding=(2,2))
        self.fc1 = nn.Linear(int((192/4) * (128/4) * 128), 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = self.drop(x)
        x = self.fc2(x)
        return x



# In[7]:


def train(clf, optimizer, trainloader, criterion, disp):
    count = 0
    policy_losses = []
    value_losses = []
    episode_reward = []
    if(disp):
        print(device)
    for i, data in enumerate(trainloader, 0):
        count += 1
        if device is None:
            inputs = data[0].type(torch.FloatTensor)
            labels = data[1].type(torch.FloatTensor)
        else:
            inputs = data[0].type(torch.FloatTensor).to(device)
            labels = data[1].type(torch.FloatTensor).to(device)
        
        value_pred = clf(inputs)
        value_loss = criterion(value_pred.float(), labels).sum()
        
        optimizer.zero_grad()
        value_loss.backward()
        optimizer.step()
        
        value_losses.append(float(value_loss.item()))

    return sum(value_losses)/len(value_losses)

# In[8]:

# evaluation
def comp_test(clf, testloader, criterion, disp):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    preds = np.empty(0)
    lbs = np.empty(0)
    loss = []
    if(disp):
        print(device)
    with torch.no_grad():
        for data in testloader:
            if device is None:
                inputs = data[0]
                labels = data[1]
            else:
                inputs = data[0].to(device)
                labels = data[1].to(device)

            outputs = clf(inputs)
            val_loss = criterion(outputs.float(), labels).sum()
            loss.append(val_loss)
            predicted = torch.argmax(torch.softmax(outputs, dim=-1), dim=-1)
            pred_npy = predicted.detach().cpu().numpy()
            total += labels.size(0)
            labels = torch.argmax(torch.softmax(labels, dim=-1), dim=-1)
            lb_npy = labels.detach().cpu().numpy()
            correct += (pred_npy == lb_npy).sum().item()
            preds = np.hstack((preds, pred_npy.squeeze()))
            lbs = np.hstack((lbs, lb_npy.squeeze()))

    conmx = confusion_matrix(lbs, preds)
    if(disp):
        print('Accuracy of the network on the test images: %.6f %%' % (100 * correct / total))
    tn, fp, fn, tp = conmx.ravel()
    if (tp + fp) == 0:
        prec = 0
    else:
        prec = tp / (tp + fp)
    if (tp + fn) == 0:
        recl = 0
    else:
        recl = tp / (tp + fn)
    if (prec+recl) == 0:
        f1 = 0
    else:
        f1 = (2*prec*recl) / (prec+recl)
    if(disp):
        print('Precision:', prec)
        print('Recall:', recl)
        print('F1:', f1)
    return prec, recl, f1, (correct / total), conmx, sum(loss)/len(loss)


# In[9]:


def run_train_pgt(train_csv, val_csv, root_folder, save_path, disp):
    start_time = time.time()

    test_csv = val_csv

    aida17k_train_dataset = Aida17kDataset(csv_file=train_csv,
                                   root_dir=root_folder,
                                   transform=transforms.Compose([
                                       Rescale((192,128)),
                                       ToTensor()
                                   ]))
    aida17k_test_dataset = Aida17kDataset(csv_file=test_csv,
                                          root_dir=root_folder,
                                          transform=transforms.Compose([
                                              Rescale((192,128)),
                                              ToTensor()
                                          ]))

    trainloader = torch.utils.data.DataLoader(aida17k_train_dataset, batch_size=10, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(aida17k_test_dataset, batch_size=10, num_workers=0)
    classes = ('non-poem', 'poem')

    clf = Classifier()
    clf.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_clf = optim.Adam(clf.parameters(), lr=0.0001)

    max_test_perf = 10000
    min_delta = 0
    patience = 5
    counter = 0

    MAX_EPISODES = 60
    PRINT_EVERY = 1

    for episode in range(1, MAX_EPISODES+1):  # loop over the dataset multiple times
        if(disp):
            print('episode:', episode)
        critic_loss = train(clf, optimizer_clf, trainloader, criterion, disp)
        if(disp):
            print('Train')
        tr_prec, tr_recl, tr_f1, tr_cur_acc, tr_conmx, tr_loss = comp_test(clf, trainloader, criterion, disp)
        if(disp):
            print('train loss: ', critic_loss)
            print('train loss: ', tr_loss)
            print('Validation')
        prec, recl, f1, cur_acc, conmx, val_loss = comp_test(clf, testloader, criterion, disp)
        if(disp):
            print('validation loss: ', val_loss)
        if max_test_perf - val_loss > min_delta:
            if(disp):
                print(1)
            max_test_perf = val_loss
            counter = 0
            # save model
            cur_high = [prec, recl, f1, cur_acc, conmx]
            torch.save(clf.state_dict(), save_path)
        elif max_test_perf - val_loss < min_delta:
            if(disp):
                print(2)
            counter += 1
            if counter >= patience:
                break

    if(disp):
        print('Finished Training')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)


# In[10]:


def run_test(test_csv, root_folder, model_path, disp):
    # run test
    start_time = time.time()

    pth = model_path
    
    aida17k_test_dataset = Aida17kDataset(csv_file=test_csv,
                                          root_dir=root_folder,
                                          transform=transforms.Compose([
                                              Rescale((192,128)),
                                              ToTensor()
                                          ]))
    testloader = torch.utils.data.DataLoader(aida17k_test_dataset, batch_size=10, num_workers=0)
    clf = Classifier()
    clf.load_state_dict(torch.load(pth))
    clf.to(device)
    criterion = nn.BCEWithLogitsLoss()
    prec, recl, f1, cur_acc, conmx, val_loss = comp_test(clf, testloader, criterion, disp)

    if(disp):
        print('Finished Testing')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)
    
    return prec, recl, f1, cur_acc



# In[11]:


def run_fine_tune(model_path, train_csv, val_csv, root_folder, save_path, disp):
    # fine tune

    start_time = time.time()

    pth = model_path

    test_csv = val_csv

    aida17k_train_dataset = Aida17kDataset(csv_file=train_csv,
                                   root_dir=root_folder,
                                   transform=transforms.Compose([
                                       Rescale((192,128)),
                                       ToTensor()
                                   ]))
    aida17k_test_dataset = Aida17kDataset(csv_file=test_csv,
                                          root_dir=root_folder,
                                          transform=transforms.Compose([
                                              Rescale((192,128)),
                                              ToTensor()
                                          ]))

    trainloader = torch.utils.data.DataLoader(aida17k_train_dataset, batch_size=10, shuffle=True, num_workers=0)
    testloader = torch.utils.data.DataLoader(aida17k_test_dataset, batch_size=10, num_workers=0)
    classes = ('non-poem', 'poem')

    clf = Classifier()
    clf.load_state_dict(torch.load(pth))
    clf.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer_clf = optim.Adam(clf.parameters(), lr=0.0001)

    max_test_perf = 10000
    min_delta = 0
    patience = 5
    counter = 0

    MAX_EPISODES = 60
    DISCOUNT_FACTOR = 0.99
    PRINT_EVERY = 1

    for episode in range(1, MAX_EPISODES+1):  # loop over the dataset multiple times
        if(disp):
            print('episode:', episode)
        critic_loss = train(clf, optimizer_clf, trainloader, criterion, disp)
        if(disp):
            print('Train')
        tr_prec, tr_recl, tr_f1, tr_cur_acc, tr_conmx, tr_loss = comp_test(clf, trainloader, criterion, disp)
        if(disp):
            print('train loss: ', critic_loss)
            print('train loss: ', tr_loss)
            print('Validation')
        prec, recl, f1, cur_acc, conmx, val_loss = comp_test(clf, testloader, criterion, disp)
        if(disp):
            print('validation loss: ', val_loss)
        if max_test_perf - val_loss > min_delta:
            if(disp):
                print(1)
            max_test_perf = val_loss
            counter = 0
            # save model
            cur_high = [prec, recl, f1, cur_acc, conmx]
            torch.save(clf.state_dict(), save_path)
        elif max_test_perf - val_loss < min_delta:
            if(disp):
                print(2)
            counter += 1
            if counter >= patience:
                break

    if(disp):
        print('Finished Training')
    end_time = time.time()
    time_elapsed = (end_time - start_time)
    if(disp):
        print(time_elapsed)


# In[12]:
def main():
    args = sys.argv[1:]
    print('args: ', args)
    var_save_name = args[0]
    run_count = int(args[1])
    disp = False
    exp_name = ['original',
                'reduct_f2', 'reduct_f4', 'reduct_f8',
                'enlarge_f2', 'enlarge_f4', 'enlarge_f8']
                
    exps_rslts = {}
    for exp in exp_name:
        exps_rslts[exp] = {
                           'revfinetune':[] # gt->pgt strategy
                          }
    for iter_count in range(run_count):
        # manual ground truth
        gt_sets = ['GT_comb_fold_0_train.csv',
                   'GT_comb_fold_0_train_reduct_f2.csv', 
                   'GT_comb_fold_0_train_reduct_f4.csv',
                   'GT_comb_fold_0_train_reduct_f8.csv',
                   'GT_comb_fold_0_train_enlarge_f2.csv', 
                   'GT_comb_fold_0_train_enlarge_f4.csv', 
                   'GT_comb_fold_0_train_enlarge_f8.csv']
        # baseline model save path
        gt_model_paths = ['cnn_160gt_baseline_forGTGen_'+var_save_name+'.pt',
                          'cnn_160gt_baseline_forGTGen_reduct_f2_'+var_save_name+'.pt',
                          'cnn_160gt_baseline_forGTGen_reduct_f4_'+var_save_name+'.pt',
                          'cnn_160gt_baseline_forGTGen_reduct_f8_'+var_save_name+'.pt',
                          'cnn_160gt_baseline_forGTGen_enlarge_f2_'+var_save_name+'.pt',
                          'cnn_160gt_baseline_forGTGen_enlarge_f4_'+var_save_name+'.pt',
                          'cnn_160gt_baseline_forGTGen_enlarge_f8_'+var_save_name+'.pt']

        for gt_set, exp in zip(gt_sets, exp_name):
            print(gt_set, exp)
            get_naive_model(gt_set, 'GT_comb_fold_0_val.csv', exp, var_save_name)
            gen_pgt(exp, var_save_name)

        for gt_set, gt_model, exp in zip(gt_sets,
                                         gt_model_paths,
                                         exp_name):
            print(gt_set, gt_model, exp)
            print(exp, 'train')
            run_train_pgt(gt_set, 
                          'GT_comb_fold_0_val.csv',
                          'images',
                          gt_model,
                          disp
                         )
            print(exp, 'baseline', 'train')
            prec, recl, f1, cur_acc = run_test('pseudoGT_test_1700.csv',
                                               'images',
                                               gt_model,
                                               disp
                                              )
            print(exp, 'baseline', 'test')
            exps_rslts[exp]['baseline'].append([prec, recl, f1, cur_acc])
                    
        # generated pseudo-groundtruth 
        pgt_sets = ['pseudoGT_unknow_14000_pgt_no160train_original_'+var_save_name+'.csv',
                    'pseudoGT_unknow_14000_pgt_no160train_reduct_f2_'+var_save_name+'.csv',
                    'pseudoGT_unknow_14000_pgt_no160train_reduct_f4_'+var_save_name+'.csv', 
                    'pseudoGT_unknow_14000_pgt_no160train_reduct_f8_'+var_save_name+'.csv',
                    'pseudoGT_unknow_14000_pgt_no160train_enlarge_f2_'+var_save_name+'.csv',
                    'pseudoGT_unknow_14000_pgt_no160train_enlarge_f4_'+var_save_name+'.csv', 
                    'pseudoGT_unknow_14000_pgt_no160train_enlarge_f8_'+var_save_name+'.csv']
                    
        # gt -> pgt strategy model save path
        fine_tune_model_paths = ['cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_original_'+var_save_name+'.pt',
                                 'cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_reduct_f2_'+var_save_name+'.pt',
                                 'cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_reduct_f4_'+var_save_name+'.pt',
                                 'cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_reduct_f8_'+var_save_name+'.pt',
                                 'cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_enlarge_f2_'+var_save_name+'.pt',
                                 'cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_enlarge_f4_'+var_save_name+'.pt',
                                 'cnn_160gt_forGTGen_w_160train_tune_pgt_160trainless_enlarge_f8_'+var_save_name+'.pt']

        for pgt_set, gt_model, gt_set, tune_model, exp in zip(pgt_sets, 
                                                              gt_model_paths, 
                                                              gt_sets, 
                                                              fine_tune_model_paths, 
                                                              exp_name):
            print(pgt_set, gt_model, gt_set, tune_model, exp)

            run_fine_tune(gt_model,
                          pgt_set,
                          'GT_comb_fold_0_val.csv',
                          'images',
                          tune_model,
                          disp
                         )
            print(exp, 'revfinetune', 'train')
            prec, recl, f1, cur_acc = run_test('pseudoGT_test_1700.csv',
                                               'images',
                                               tune_model,
                                               disp
                                              )
            print(exp, 'revfinetune', 'test')
            exps_rslts[exp]['revfinetune'].append([prec, recl, f1, cur_acc])

        with open('result_data_'+var_save_name+'_'+str(iter_count)+'.pkl', 'wb') as fp:
            pickle.dump(exps_rslts, fp)
            print('exps rslts saved successfully to file: ', iter_count)


if __name__ == "__main__":
    main()


