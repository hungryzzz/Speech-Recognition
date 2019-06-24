import os
os.sys.path.append('..')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader

import argparse

from model import vgg11_bn, Mymodel
from dataset import MyDataset


class Trainer(object):

  def __init__(self, opt):
    self.batch_size = opt.batch_size
    self.iter_num = opt.iter_num
    self.opt = opt
    self.start = 0

    train_dataset = MyDataset(opt)
    dev_dataset = MyDataset(opt, is_dev=True)
    self.train_len = len(train_dataset)
    self.dev_len = len(dev_dataset)
    self.train_loader = DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=True, num_workers=8)
    self.dev_loader = DataLoader(dev_dataset, batch_size=opt.batch_size, num_workers=8)
    print("# train dataset : {}".format(self.train_len))
    print("# development dataset : {}".format(self.dev_len))

    self.model = vgg11_bn().cuda()
    # class config(object):
    #   input_s = train_dataset[0][0].shape[-1]
    #   hidden_s = opt.hidden_s
    #   max_len = train_dataset[0][0].shape[0]
    #   dropout_rate = opt.dr
    #   num_class = 20
    # self.model = Mymodel(config).cuda()
    self.loss_func = nn.CrossEntropyLoss().cuda()
    self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr)
    self.softmax = nn.Softmax(dim=1)
    self.load_model()
    print("# epoches : {}\n\n".format(opt.iter_num))


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
        output = self.model(data[0].cuda())
        loss = self.loss_func(output, data[1].view(-1).cuda())

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        ave_loss += loss.detach()

        if (i+1) % 100 == 0:
          print("epoch : {}\titerator : {}".format(epoch+1, i+1))
          print("# loss : {}\n\n".format(loss))

      print('epoch : {}\ttrain dataset average loss : {}'.format(epoch+1, ave_loss/len(self.train_loader)))
      
      # evaluate
      self.model.eval()
      ave_loss = 0
      for i, data in enumerate(self.dev_loader):
        output = self.model(data[0].cuda())
        loss = self.loss_func(output, data[1].view(-1).cuda())
        ave_loss += loss.detach()
      
      print('epoch : {}\tdevelopment dataset average loss : {}\n\n'.format(epoch+1, ave_loss/len(self.dev_loader)))
    
    # save model
    if self.start < self.iter_num:
      state_dict = {"model": self.model.state_dict(), 
                    "optimizer": self.optimizer.state_dict(),
                    "epoch": self.iter_num}
      torch.save(state_dict, self.opt.save)
      print("model saved to {}\n".format(self.opt.save))

  def cal_accuracy(self, output, target):
    pre_target = torch.max(self.softmax(output), dim=1)[1]
    diff = torch.abs(pre_target-target)
    diff = torch.where(diff == 0, torch.full_like(diff, 1), torch.full_like(diff, 0))
    return torch.sum(diff).item()

  def get_accuracy(self):
    self.model.eval()
    accuracy = 0
    for i, data in enumerate(self.train_loader):
      output = self.model(data[0].cuda())
      accuracy += self.cal_accuracy(output, data[1].view(-1).cuda())
      
    print('accuracy of train dataset : {}'.format(accuracy/self.train_len))
      
    accuracy = 0 
    for i, data in enumerate(self.dev_loader):
      output = self.model(data[0].cuda())
      accuracy += self.cal_accuracy(output, data[1].view(-1).cuda())
    
    print('accuracy of development dataset : {}\n\n'.format(accuracy/self.dev_len))



class Options():

  def __init__(self):
    self.parser = argparse.ArgumentParser()
  
  def initialize(self):
    self.parser.add_argument('--train_dataset', type=str, default='../dataset/train', help='path to training dataset')
    self.parser.add_argument('--dev_dataset', type=str, default='../dataset/dev', help='path to development dataset')
    self.parser.add_argument('--batch_size', type=int, default=32, help='batch size')
    self.parser.add_argument('--iter_num', type=int, default=50, help='number of iteration')
    self.parser.add_argument('--lr', type=float, default=4e-5, help='learning rate')
    self.parser.add_argument('--save', type=str, default='model.pkl', help='path to save model')
    self.parser.add_argument('--load', type=int, default=1, help='weather to load last model')
    self.parser.add_argument('--hidden_s', type=int, default=256, help='hidden size of lstm')
    self.parser.add_argument('--dr', type=float, default=0.5, help='dropout rate')

  def parse(self):
    self.initialize()
    return self.parser.parse_args()


