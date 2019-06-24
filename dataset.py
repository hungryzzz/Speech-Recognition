import os
os.sys.path.append('..')
import torch
from torch.utils.data import Dataset
from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import glob
import scipy.io.wavfile as wav

from preprocess import Signal

class MyDataset(Dataset):

  def __init__(self, opt, is_dev=False):
    super(MyDataset, self).__init__()
    if is_dev:
      path = opt.dev_dataset
    else:
      path = opt.train_dataset
    self.files = glob.glob(path + '/*.npy')

  def __len__(self):
    return len(self.files)

  def __getitem__(self, index):
    filename = self.files[index]

    mfcc_feat = np.load(filename)
    target = int(filename.split('-')[1])
    
    return torch.FloatTensor(mfcc_feat).unsqueeze(0), torch.LongTensor([target])



