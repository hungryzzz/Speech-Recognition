import os
os.sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import argparse

from model import vgg11_bn
from dataset import MyDataset


class Trainer(object):

  def __init__(self, opt):
    self.data_path = opt.dataset
    self.batch_size = opt.batch_size
    self.iter_num = opt.iter_num
    self.opt = opt
    self.start = 0

    self.model = vgg11_bn()
    self.loss_func = nn.CrossEntropyLoss()
    self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
    self.load_model()
    print("# epoches : {}".format(opt.iter_num))

    train_dataset = MyDataset(self.data_path)
    dev_dataset = MyDataset(self.data_path, is_dev=True)
    self.train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True)
    self.dev_loader = DataLoader(dev_dataset, batch_size=opt.batch_size)
    print("# train dataset : {}".format(len(train_dataset)))
    print("# development dataset : {}\n\n".format(len(dev_dataset)))


  def load_model(self):
    if self.opt.load == 0:
      return
    try:
      checkpoint = torch.load(self.opt.save)
    except FileNotFoundError:
      return
    else:
      self.start = checkpoint["epoch"]
      self.model.load_state_dict(checkpoint["model"])
      self.optimizer.load_state_dict(checkpoint["optimizer"])
      print("Model Loaded! From the {} epoch".format(self.start))

  def train(self):
    for epoch in range(self.start, self.iter_num):
      # train
      self.model.train()
      self.model.zero_grad()
      ave_loss = 0
      for i, data in enumerate(self.train_loader):
        output = self.model(data[0])
        loss = self.loss_func(output, data[1].view(-1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ave_loss += loss

        if (i+1) % 40 == 0:
          print("epoch : {}\titerator : {}".format(epoch+1, i+1))
          print("# loss : {}\n\n".format(loss))

      print('epoch : {}\ttrain dataset average loss : {}'.format(epoch+1, ave_loss/len(self.train_loader)))
      
      # evaluate
      self.model.eval()
      ave_loss = 0
      for i, data in enumerate(self.dev_loader):
        output = self.model(data[0])
        loss = self.loss_func(output, data[1].view(-1))
        ave_loss += loss
      
      print('epoch : {}\tdevelopment dataset average loss : {}'.format(epoch+1, ave_loss/len(self.dev_loader)))
    
    # save model
    if self.start < self.iter_num:
      state_dict = {"model": self.model.state_dict(), 
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.iter_num}
      torch.save(state_dict, self.opt.save)
      print("model saved to {}\n".format(self.opt.save))



class Options():

  def __init__(self):
    self.parser = argparse.ArgumentParser()
  
  def initialize(self):
    self.parser.add_argument('--dataset', type=str, default='../speech_data/', help='path to dataset')
    self.parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    self.parser.add_argument('--iter_num', type=int, default=50, help='number of iteration')
    self.parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    self.parser.add_argument('--save', type=str, default='dataset.pkl', help='path to save model')
    self.parser.add_argument('--load', type=int, default=1, help='weather to load last model')

  def parse(self):
    self.initialize()
    return self.parser.parse_args()


