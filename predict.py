import torch
import torch.nn as nn
import numpy as np

from preprocess import preprocess

import glob


words = ['数字', '语音', '语言', '识别', '中国',
        '总工', '北京', '背景', '上海', '商行',
        '复旦', '饭店', 'Speech', 'Speaker', 'Signal',
        'Process', 'Print', 'Open', 'Close', 'Project']

softmax = nn.Softmax(dim=1)

def get_ans(output):
  pre_target = torch.max(softmax(output), dim=1)[1].item()
  return words[pre_target]

def predict(filename, myModel):
  mfcc_feat = preprocess(filename)
  if mfcc_feat is False:
    return False
  else:
    input_data = torch.FloatTensor(mfcc_feat).unsqueeze(0).unsqueeze(0)
    output = myModel(input_data)
    ans = get_ans(output)
    return ans

# path = "./data/16307130345/"
# files = glob.glob(path + '*')

# for file in files:
#   print(words[int(file.split('-')[1])], predict(file, model))

  