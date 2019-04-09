import os
import matplotlib as mpl
if os.environ.get('DISPLAY','') == '':
    print('no display found. Using non-interactive Agg backend')
    mpl.use('Agg')
import matplotlib.pyplot as plt
import scipy.io as sio
import torch
import torch.nn as nn
import numpy as np
from src.torchviz import  make_dot, make_dot_from_trace


class stats:
    def __init__(self, path, start_epoch):
        if start_epoch is not 0:
           stats_ = sio.loadmat(os.path.join(path,'stats.mat'))
           data = stats_['data']
           content = data[0,0]
           self.trainObj1 = content['trainObj1'][:,:start_epoch].squeeze().tolist()
           self.trainObj2 = content['trainObj2'][:,:start_epoch].squeeze().tolist()
           self.trainTop1_1 = content['trainTop1_1'][:,:start_epoch].squeeze().tolist()
           self.trainTop1_2 = content['trainTop1_2'][:,:start_epoch].squeeze().tolist()
           self.valObj1 = content['valObj1'][:,:start_epoch].squeeze().tolist()
           self.valObj2 = content['valObj2'][:,:start_epoch].squeeze().tolist()
           self.valTop1_1 = content['valTop1_1'][:,:start_epoch].squeeze().tolist()
           self.valTop1_2 = content['valTop1_2'][:,:start_epoch].squeeze().tolist()
           if start_epoch is 1:
               self.trainObj1 = [self.trainObj1]
               self.trainObj2 = [self.trainObj2]
               self.trainTop1_1 = [self.trainTop1_1]
               self.trainTop1_2 = [self.trainTop1_2]
               self.valObj1 = [self.valObj1]
               self.valObj2 = [self.valObj2]
               self.valTop1_1 = [self.valTop1_1]
               self.valTop1_2 = [self.valTop1_2]
        else:
           self.trainObj1 = []
           self.trainObj2 = []
           self.trainTop1_1 = []
           self.trainTop1_2 = []
           self.valObj1 = []
           self.valObj2 = []
           self.valTop1_1 = []
           self.valTop1_2 = []
    def _update(self, trainObj1, trainObj2, Top1_1, Top1_2, valObj1, valObj2, prec1, prec5):
        self.trainObj1.append(trainObj1)
        self.trainObj2.append(trainObj2)
        self.trainTop1_1.append(Top1_1.cpu().numpy())
        self.trainTop1_2.append(Top1_2.cpu().numpy())
        self.valObj1.append(valObj1)
        self.valObj2.append(valObj2)
        self.valTop1_1.append(prec1.cpu().numpy())
        self.valTop1_2.append(prec5.cpu().numpy())


def vizNet(model, path):
    x = torch.randn(10,3,224,224)
    y, y2 = model(x)
    g = make_dot(y)
    g.render(os.path.join(path,'graph'), view=False)

def plot_curve(stats, path, iserr):
    trainObj1 = np.array(stats.trainObj1)
    trainObj2 = np.array(stats.trainObj2)
    valObj1 = np.array(stats.valObj1)
    valObj2 = np.array(stats.valObj2)
    if iserr:
        trainTop1_1 = 100 - np.array(stats.trainTop1_1)
        trainTop1_2 = 100 - np.array(stats.trainTop1_2)
        valTop1_1 = 100 - np.array(stats.valTop1_1)
        valTop1_2 = 100 - np.array(stats.valTop1_2)
        titleName = 'error'
    else:
        trainTop1_1 = np.array(stats.trainTop1_1)
        trainTop1_2 = np.array(stats.trainTop1_2)
        valTop1_1 = np.array(stats.valTop1_1)
        valTop1_2 = np.array(stats.valTop1_2)
        titleName = 'accuracy'
    epoch = len(trainObj1)
    figure = plt.figure()
    obj = plt.subplot(1,3,1)
    obj.plot(range(1,epoch+1),trainObj1,'o-',label = 'train1')
    obj.plot(range(1,epoch+1),trainObj2,'o-',label = 'train2')
    obj.plot(range(1,epoch+1),valObj1,'o-',label = 'val1')
    obj.plot(range(1,epoch+1),valObj2,'o-',label = 'val2')
    plt.xlabel('epoch')
    plt.title('objective')
    handles, labels = obj.get_legend_handles_labels()
    obj.legend(handles[::-1], labels[::-1])
    Top1_1 = plt.subplot(1,3,2)
    Top1_1.plot(range(1,epoch+1),trainTop1_1,'o-',label = 'train')
    Top1_1.plot(range(1,epoch+1),valTop1_1,'o-',label = 'val')
    plt.title('Top1_1'+titleName)
    plt.xlabel('epoch')
    handles, labels = Top1_1.get_legend_handles_labels()
    Top1_1.legend(handles[::-1], labels[::-1])
    Top1_2 = plt.subplot(1,3,3)
    Top1_2.plot(range(1,epoch+1),trainTop1_2,'o-',label = 'train')
    Top1_2.plot(range(1,epoch+1),valTop1_2,'o-',label = 'val')
    plt.title('Top1_2'+titleName)
    plt.xlabel('epoch')
    handles, labels = Top1_2.get_legend_handles_labels()
    Top1_2.legend(handles[::-1], labels[::-1])
    filename = os.path.join(path, 'net-train.pdf')
    figure.savefig(filename, bbox_inches='tight')
    plt.close()

def decode_params(input_params):
    params = input_params[0]
    out_params = []
    _start=0
    _end=0
    for i in range(len(params)):
        if params[i] == ',':
            out_params.append(float(params[_start:_end]))
            _start=_end+1
        _end+=1
    out_params.append(float(params[_start:_end]))
    return out_params
