import os
os.sys.path.append('..')
import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile as wav
import math
from scipy.fftpack import dct

from python_speech_features import mfcc
from python_speech_features import logfbank
import librosa
import librosa.display

# from fastNLP.core import DataSet, Instance
import torch
import glob

path = '../speech_data/'
save_train_path = '../dataset/train/'
save_dev_path = '../dataset/dev/'

folders = glob.glob(path + '/*')
files = sum([glob.glob(folder + '/*') for folder in folders], [])


class Signal(object):
  def __init__(self, y, sampling_freq, t=None):
    self.y = y
    self.sampling_freq = sampling_freq
    
    if t is None:
      self.t = np.arange(len(y)) / sampling_freq
    else:
      self.t = t
  
  '''
    utils function
  '''
  def plot_sig(self):
    plt.plot(self.t, self.y)
    plt.show()

  '''
    endpoint_detection
  '''
  def get_short_term_energy(self, window_size):
    window = np.hamming(window_size)
    i, j = 0, window_size
    step = window_size // 2
    energe = []
    t = []
    while j < len(self.y):
      frame = self.y[i : j]
      frame_add_win = frame * window
      energe.append(np.sum(frame_add_win*frame_add_win))
      t.append((self.t[i]+self.t[j]) / 2)
      i += step
      j += step
    return Signal(energe, sampling_freq=1/(t[1]-t[0]) ,t=t)

  def get_short_term_ZCR(self, window_size):
    window = np.hamming(window_size)
    # plt.plot(window)
    # plt.show()
    i, j = 0, window_size
    step = window_size // 2
    sign = lambda num: 1 if num > 0 else -1 if num < 0 else 0
    zcr = []
    t = []
    while j < len(self.y):
      frame = self.y[i : j]
      f_w = frame * window
      s_f_w = np.array([sign(num) for num in f_w])
      zcr.append(np.sum(np.abs(s_f_w[1:]-s_f_w[:window_size-1])) / 2)
      t.append((self.t[i]+self.t[j]) / 2)
      i += step
      j += step
    return Signal(zcr, sampling_freq=1/(t[1]-t[0]), t=t)

  def plot_short_term_feature(self, window_size):
    energe = self.get_short_term_energy(window_size)
    zcr = self.get_short_term_ZCR(window_size)
    # plot
    plt.figure(num = 1)
    plt.subplot(311)
    plt.plot(self.t, self.y)
    plt.ylabel('Signal')
    plt.subplot(312)
    plt.plot(energe.t, energe.y)
    plt.ylabel('Energe')
    plt.subplot(313)
    plt.plot(zcr.t, zcr.y)
    plt.ylabel('ZCR')
    plt.show()

  def endpoint_detection(self, draw=False):
    frame_len = 0.01
    window_size = math.ceil(frame_len * self.sampling_freq)
    energe = self.get_short_term_energy(window_size)
    zcr = self.get_short_term_ZCR(window_size)
    EMax, EMin, C = 2, 1, 40

    def get_endpoint_index(energe, zcr):
      zcr_len = len(zcr)
      point_h = zcr_len-1
      for i in range(zcr_len):
        if energe[i] > EMax:
          point_h = i
          break
      point = point_h
      for i in range(point_h, -1, -1):
        if energe[i] < EMin:
          point = i
          break
      for i in range(point, -1, -1):
        if zcr[i] < C:
          return i
      return point

    begin = energe.t[get_endpoint_index(energe.y, zcr.y)]
    end = energe.t[::-1][get_endpoint_index(energe.y[::-1], zcr.y[::-1])]

    def get_endpoint_time(time):
      for i in range(len(self.t)):
        if self.t[i] > time:
          return i
      return 0

    begin = get_endpoint_time(begin)
    end = get_endpoint_time(end)

    # plot
    if draw:
      plt.figure(num = 1)
      plt.subplot(311)
      plt.plot(self.t, self.y)
      plt.plot(self.t[begin:end], self.y[begin:end], 'r')
      plt.ylabel('Signal')
      plt.subplot(312)
      plt.plot(energe.t, energe.y)
      plt.ylabel('Energe')
      plt.subplot(313)
      plt.plot(zcr.t, zcr.y)
      plt.ylabel('ZCR')
      plt.show()
      
    return begin, end

  '''
    extract mfcc
  '''
  def pre_emphasis(self, alpha=0.97):
    y = self.y
    self.y = np.append(y[0], y[1:] - alpha*y[:-1])

  def get_mfcc_via_PHF(self, draw=False, num_mel=128, init_sed=2):
    sampling_freq = self.sampling_freq
    # 端点检测
    begin, end = self.endpoint_detection(draw=draw)
    if (end - begin) < (0.025*sampling_freq):
      return False, False
    self.y = np.append(self.y[begin:end+1], np.zeros(sampling_freq*init_sed - (end - begin)))
    y = self.y

    mfcc_feat = mfcc(y, sampling_freq, nfilt=num_mel)
    fbank_feat = logfbank(y, sampling_freq, nfilt=num_mel)
    return fbank_feat, mfcc_feat

  
  def get_mfcc(self, draw=False, num_mel=128, num_cep=12, init_sed=2):
    sampling_freq = self.sampling_freq
    # 端点检测
    begin, end = self.endpoint_detection(draw=draw)

    # 预加重
    self.pre_emphasis()
    self.y = np.append(self.y[begin:end+1], np.zeros(sampling_freq*init_sed - (end - begin)))
    y = self.y

    # 分帧
    sig_len = len(y)
    frame_size, frame_stride = 0.03, 0.01
    frame_len, frame_step = round(frame_size*sampling_freq), round(frame_stride*sampling_freq)
    num_frames = math.ceil((sig_len-frame_len) / frame_step)
    if math.ceil((end-begin-frame_len)/frame_step) <= 0:
      return False, False

    pad_sig_len = num_frames*frame_step + frame_len
    pad_sig = np.append(y, np.zeros((pad_sig_len-sig_len)))
    num_frames += 1
    frames = np.hstack([pad_sig[i*frame_step : i*frame_step+frame_len].reshape(-1, 1) for i in range(0, num_frames)])

    # 加窗
    window = np.hamming(frame_len)
    frames = frames * window.reshape(-1, 1)

    # fft + 能量谱
    # 区别？？
    frames = np.abs(np.fft.rfft(frames)) ** 2
    # frames = np.abs(np.fft.fft(frames)) ** 2

    # mel滤波
    mel_freq = 2595 * np.log10(1 + sampling_freq / 2 / 700)
    mel_points = np.linspace(0, mel_freq, num_mel+2)
    hz_points = 700 * (10**(mel_points/2595) - 1)
    freq_index = np.floor((frame_len+1) / sampling_freq * hz_points).astype(int)
    mel_filters = np.zeros((num_mel, frame_len))
    for i in range(1, num_mel+1):
      l = freq_index[i-1]
      m = freq_index[i]
      r = freq_index[i+1]
      for k in range(l, m):
        mel_filters[i-1][k] = (k-l) / (m-l)
      for k in range(m, r):
        mel_filters[i-1][k] = (r-k) / (r-m)
    
    mel_feat = mel_filters.dot(frames)
    # mel_feat = 20 * np.log10(mel_feat)
    mfcc = dct(mel_feat, type=2, axis=1, norm='ortho')[1: num_cep+1, :]
    mel_filters -= np.mean(mel_filters, 1, keepdims=True)
    mfcc -= np.mean(mfcc, 1, keepdims=True)
    return mel_feat, mfcc

  def extract_mfcc(self, draw=False):
    # 端点检测
    # begin, end = self.endpoint_detection(draw=draw)

    # 预加重
    self.pre_emphasis()
    # y = self.y[begin:end]
    y = self.y

    # 分帧
    sig_len = len(y)
    frame_size, frame_stride = 0.025, 0.01
    frame_len, frame_step = round(frame_size*self.sampling_freq), round(frame_stride*self.sampling_freq)
    num_frames = math.ceil((sig_len-frame_len) / frame_step)
    if num_frames is 0:
      return False, False

    pad_sig_len = num_frames*frame_step + frame_len
    pad_sig = np.append(y, np.zeros((pad_sig_len-sig_len)))

    indices = np.tile(np.arange(0, frame_len), (num_frames, 1)) + np.tile(np.arange(0, num_frames*frame_step, frame_step), (frame_len, 1)).T
    frames = pad_sig[np.mat(indices).astype(np.int32, copy=False)]
    
    # 加窗
    frames *= np.hamming(frame_len)

    # 傅里叶变换和功率谱
    NFFT = 512
    mag_frames = np.absolute(np.fft.rfft(frames, NFFT))
    pow_frames = (1.0/NFFT) * (mag_frames**2)

    # 转为MEL频率
    low_freq_mel = 0
    nfilt = 40
    high_freq_mel = 2595 * np.log10(1 + (self.sampling_freq/2)/700)
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt+2)
    hz_points = 700 * (10**(mel_points/2595) - 1)

    bin = np.floor((NFFT+1) * hz_points / self.sampling_freq)

    fbank = np.zeros((nfilt, int(np.floor(NFFT/2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * np.log10(filter_banks)  # dB

    num_ceps = 98
    cep_lifter = 22

    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape
    n = np.arange(ncoeff)

    lift = 1 + (cep_lifter / 2) * np.sin(np.pi * n / cep_lifter)
    mfcc *= lift  
    mfcc -= (np.mean(mfcc, axis=0) + 1e-8)

    return mfcc, filter_banks

    
def normalize(sig, amp=1.0):
  high, low = abs(max(sig)), abs((min(sig)))
  return amp * sig / max(high, low)

def delete_noisy(sig, num1=3000, num2=5000):
  a = np.append(np.zeros((num1)), sig[num1:len(sig)-num2])
  return np.append(a, np.zeros((num2)))

def read_sig(filename):
  sampling_freq, sig = wav.read(filename)
  sig = delete_noisy(sig)
  return Signal(normalize(sig), sampling_freq=sampling_freq)

def preprocess(filename):
  signal = read_sig(filename)
  mfcc_feat, _ = signal.get_mfcc_via_PHF()
  if mfcc_feat is False:
    return False
  else:
    return mfcc_feat
    

def get_preprocess_dataset():
  for i in range(0, len(files)):
    filename = files[i]
    signal = read_sig(filename)
    # mfcc_feat, _ = signal.get_mfcc()
    mfcc_feat, _ = signal.get_mfcc_via_PHF()
    if mfcc_feat is False:
      continue
    if (i+1) % 20 == 0:
      np.save(save_dev_path+filename.split('\\')[-1].split('.')[0], mfcc_feat)
    else:
      np.save(save_train_path+filename.split('\\')[-1].split('.')[0], mfcc_feat)
    if (i+1) % 200 == 0:
      print('{} saved!'.format(i+1))


    

if __name__ == "__main__":
  # get_preprocess_dataset()

  filelist = [files[1664], files[1678], files[1275], files[868]]
  signal = read_sig(files[0])
  mfcc_feat, _ = signal.get_mfcc()
  print(mfcc_feat.shape)
  # plt.figure(num = 1)
  # a = [1, 1, 2, 2]
  # b = [1, 2, 1, 2]
  # for i in range(len(filelist)):
  #   plt.figure()
  #   y, sr = librosa.load(filelist[i], sr=None)
  #   plt.subplot(2, 1, 1)
  #   librosa.display.waveplot(y, sr)
  #   melspec = librosa.feature.melspectrogram(y, sr, n_fft=1024, hop_length=512, n_mels=128)
  #   logmelspec = librosa.power_to_db(melspec)
  #   plt.subplot(2, 1, 2)
  #   librosa.display.specshow(logmelspec, sr=sr, x_axis='time', y_axis='mel')
  #   plt.tight_layout()
  #   plt.show()


  
