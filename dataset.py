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

  def __init__(self, path, is_dev=False):
    super(MyDataset, self).__init__()
    
    folders = glob.glob(path + '/*')
    all_files = sum([glob.glob(folder + '/*') for folder in folders], [])
    if is_dev:
      self.files = all_files[-400:]
    else:
      self.files = all_files[:-400]

  def __len__(self):
    return len(self.files)

  def normalize(self, sig, amp=1.0):
    high, low = abs(max(sig)), abs((min(sig)))
    return amp * sig / max(high, low)

  def delete_noisy(self, sig, num1=3000, num2=5000):
    a = np.append(np.zeros((num1)), sig[num1:len(sig)-num2])
    return np.append(a, np.zeros((num2)))

  def __getitem__(self, index):
    filename = self.files[index]

    sampling_freq, file = wav.read(filename)
    file = self.delete_noisy(file)
    signal = Signal(self.normalize(file), sampling_freq=sampling_freq)

    mfcc_feat, _ = signal.get_mfcc()
    target = int(filename.split('\\')[2].split('-')[1])
    
    return Variable(torch.FloatTensor(mfcc_feat).unsqueeze(0), requires_grad=True), torch.LongTensor([target])



