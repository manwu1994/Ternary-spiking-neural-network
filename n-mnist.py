
import sys, os
CURRENT_TEST_DIR = os.getcwd()
sys.path.append("/home/wuman/SNN/DiaNet+spiking/N-MNIST/")
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import pandas as pd
import torch.optim as optim
from torchvision import datasets,transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
import torch.utils.data as utils
from datetime import datetime
#from learningStats import learningStats
import zipfile
import spikingjelly
import nmist
from torchsummary import summary
from torchstat import stat
from ptflops import get_model_complexity_info
from spikingjelly.datasets.n_mnist import NMNIST
#import slayerSNN as snn
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size=128
class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        #out = torch.zeros_like(input).cuda()
        #out[input > 0] = 1.0
        return input.ge(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        #grad = torch.exp( -(grad_input - 0.3) **2/(2 * 0.3 ** 2) ) / ((2 * 0.3 * 3.141592653589793) ** 0.5)
        #grad = grad_input * grad
        #grad = abs(grad_input - threshold) < lens
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)*2
        #grad=F.hardtanh(grad_input) 
        #grad =grad_input * 0.3*torch.exp(-0.01*torch.abs(input))
        #print(grad)
        #grad =grad_input * 0.3 * torch.exp(F.threshold(1.0 - torch.abs(input), 0, 0))
        return grad#grad

class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=1.0, img_size=34,default_threshold = 1.0,num_cls=10):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        #self.threshold=nn.Parameter(torch.tensor(1.0, dtype=torch.float), requires_grad=True)
        #self.register_buffer('threshold', torch.tensor([1.]))
        #self.threshold = nn.ParameterDict()
        print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        affine_flag = True
        bias_flag = True

        self.conv1 = nn.Conv2d(2, 16, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.BatchNorm2d(16, eps=1e-4, momentum=0.8, affine=affine_flag) 
        
        self.conv2 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.BatchNorm2d(16, eps=1e-4, momentum=0.8, affine=affine_flag) 
        
        self.conv3 = nn.Conv2d(16, 16, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.BatchNorm2d(16, eps=1e-4, momentum=0.8, affine=affine_flag) 
        
        self.pool3 = nn.MaxPool2d(2,stride=2)


        self.fc1 = nn.Linear((self.img_size//2)*(self.img_size//2)*16, 128, bias=bias_flag)
        self.bntt_fc = nn.BatchNorm1d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.fc2 = nn.Linear(128, self.num_cls, bias=bias_flag)

       
        self.fc_list=[self.fc1,self.fc2]
        self.threshold1=nn.Parameter(torch.tensor(default_threshold))
        self.threshold2=nn.Parameter(torch.tensor(default_threshold))
        self.threshold3=nn.Parameter(torch.tensor(default_threshold))
        #self.threshold4=nn.Parameter(torch.tensor(default_threshold))
        #self.threshold5=nn.Parameter(torch.tensor(default_threshold))
        #self.threshold6=nn.Parameter(torch.tensor(default_threshold))
        #self.threshold7=nn.Parameter(torch.tensor(default_threshold))
        self.threshold10=nn.Parameter(torch.tensor(default_threshold))
        #self.dropout = nn.Dropout(0.20)
        # Turn off bias of BNTT
        #for bn_list in self.bntt_list:
            #for bn_temp in bn_list:
                #bn_temp.bias = None

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
        # Initialize the firing thresholds of all the layers
        #for m in self.modules():
            #if (isinstance(m, nn.Conv2d)):
                #m.threshold = 1.0
                #torch.nn.init.xavier_uniform_(m.weight, gain=2)
            #elif (isinstance(m, nn.Linear)):
                #m.threshold = 1.0
                #torch.nn.init.xavier_uniform_(m.weight, gain=2)


    def forward(self, inp):
        
        #inp = inp.permute(1, 0, 2, 3, 4)
        inp_2 = inp.permute(1, 0, 2, 3, 4)
        #print("inp_2",inp_2.size())
        batch_size = inp_2.size(1)
        mem_conv1 = torch.zeros(batch_size, 16, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 16, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 16, self.img_size, self.img_size).cuda()

        mem_fc1 = torch.zeros(batch_size, 128).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()


        for t in range(self.num_steps):
            #xa = inp > torch.rand(inp.size(), device=device)
            x = inp.permute(1, 0, 2, 3, 4)
            #rand_inp = torch.rand_like(inp).cuda()
            #spike_inp = torch.mul(torch.le(rand_inp , torch.abs(inp)).float(), torch.sign(inp)) ##le means smaller and equal
            #spike_inp = PoissonGen(inp)
            #out_prev = spike_inp
            #print(mem_conv1.size())
            mem_conv1 = self.leak_mem* mem_conv1 + self.bntt1(self.conv1(x[0]))
            mem_thr = (mem_conv1 / self.threshold1 - 1.0)
            out = self.spike_fn(mem_thr)
            #rst = torch.zeros_like(mem_conv1).cuda()
            rst = self.threshold1* (mem_thr>0).float() # (mem_thr>0) return 1
            mem_conv=mem_conv1.clone()
            mem_conv1 = mem_conv - rst
            #print("mem_conv1 - rst",mem_conv1.size())
            out_prev1 = out.clone()

            mem_conv2 = self.leak_mem * mem_conv2 + self.bntt2(self.conv2(out_prev1))
            #print("mem_conv2",mem_conv2.size())
            mem_thr = (mem_conv2 / self.threshold2) - 1.0  ###mem_conv2 compare with threshold
            out = self.spike_fn(mem_thr)
            rst = self.threshold2* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_conv2).cuda()
            #rst[mem_thr > 0] = self.threshold2  ###soft reset:
            mem_conv2 = mem_conv2 - rst ###rst=0 means mem_thr<0,means mem_conv2<threshold,thus:mem_conv2 not change
            out_prev2 = out.clone()  ###rst=threshold means mem_thr>0,means mem_conv2>threshold,thus:mem_conv2 =0 or mem_conv2 - threshold 
            
            mem_conv3 = self.leak_mem * mem_conv3 + self.bntt3(self.conv3(out_prev2))
            #print("mem_conv2",mem_conv2.size())
            mem_thr = (mem_conv3 / self.threshold3) - 1.0  ###mem_conv2 compare with threshold
            out = self.spike_fn(mem_thr)
            rst = self.threshold3* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_conv2).cuda()
            #rst[mem_thr > 0] = self.threshold2  ###soft reset:
            mem_conv3 = mem_conv3 - rst ###rst=0 means mem_thr<0,means mem_conv2<threshold,thus:mem_conv2 not change
            out_prev3 = out.clone()
            
            out_pool3=self.pool3(out_prev3)

            out_pool3 = out_pool3.reshape(batch_size, -1)
            #self.fc1.weight.data =torch.tanh(self.fc1.weight.data)
            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc(self.fc1(out_pool3))  ### the last layer input
            mem_thr = (mem_fc1 / self.threshold10) - 1.0  ###
            out = self.spike_fn(mem_thr)
            rst = self.threshold10* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_fc1).cuda()
            #rst[mem_thr > 0] = self.threshold8
            mem_fc1 = mem_fc1 - rst
            out_prev9 = out.clone()
            #self.fc2.weight.data =torch.tanh(self.fc2.weight.data)
            # accumulate voltage in the last layer
            mem_fc=self.fc2(out_prev9)
            mem_fc2 = mem_fc2 + mem_fc
            #print("mem_fc2",mem_fc2.size())
        out_voltage = mem_fc2 / self.num_steps
        return out_voltage

model = SNN_VGG9_BNTT(num_steps = 2, leak_mem=1.0, img_size=34,  num_cls=10)
model = model.cuda()
LR = 0.05
Momentum = 0.9
epochs = 70
class TernarizeOp:
    def __init__(self,model):
        count_targets = 0
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                count_targets += 1
        self.ternarize_range = np.linspace(0,count_targets-1,count_targets).astype('int').tolist()
        self.num_of_params = len(self.ternarize_range)
        self.saved_params = []
        self.target_modules = []
        self.alpha=[]
        self.delta=[]
        self.saved_alpha=[]
        self.out=[]
        for m in model.modules():
            if isinstance(m,nn.Conv2d) or isinstance(m,nn.Linear):
                tmp = m.weight.data.clone()
                self.saved_params.append(tmp) #tensor
                self.target_modules.append(m.weight) #Parameter
    
    def SaveWeights(self):
        for index in range(self.num_of_params):
            self.saved_params[index].copy_(self.target_modules[index].data)
            

    def TernarizeWeights(self):
        for index in range(self.num_of_params):
            self.delta,self.alpha,self.out,self.target_modules[index].data = self.Ternarize(self.target_modules[index].data)
    
    def Ternarize(self,tensor):
        tensor = tensor.cpu()
        output = torch.zeros(tensor.size())
        delta = self.Delta(tensor)
        alpha = self.Alpha(tensor,delta)
        #tensor.size()[0] input_channel and input neuron
        for i in range(tensor.size()[0]):
            for w in tensor[i].view(1,-1):
                pos_one = (w > delta[i]).type(torch.FloatTensor)
                neg_one = torch.mul((w < -delta[i]).type(torch.FloatTensor),-1)
            out = torch.add(pos_one,neg_one).view(tensor.size()[1:])
            output[i] = torch.add(output[i],torch.mul(out,alpha[i]))
            #output[i] = torch.add(output[i],torch.mul(out,alpha[i]/alpha[i]))
        return delta,alpha,out,output.cuda()
            

    def Alpha(self,tensor,delta):
        Alpha = []
        for i in range(tensor.size()[0]):
            count = 0
            abssum = 0
            absvalue = tensor[i].view(1,-1).abs()
            for w in absvalue:
                truth_value = w > delta[i] #print to see
            count = truth_value.sum()
            abssum = torch.matmul(absvalue,truth_value.type(torch.FloatTensor).view(-1,1))
            Alpha.append(abssum/count)
        alpha = Alpha[0]
        for i in range(len(Alpha) - 1):
            alpha = torch.cat((alpha,Alpha[i+1]))
        return alpha

    def Delta(self,tensor):
        n = tensor[0].nelement()
        if(len(tensor.size()) == 4):     #convolution layer
            delta = 0.7 * tensor.norm(1,3).sum(2).sum(1).div(n)
        elif(len(tensor.size()) == 2):   #fc layer
            delta = 0.7 * tensor.norm(1,1).div(n)
        return delta
            
    
    def Ternarization(self):
        self.SaveWeights()
        self.TernarizeWeights()
    
    def Restore(self):
        for index in range(self.num_of_params):
            self.target_modules[index].data.copy_(self.saved_params[index])
ternarize_op = TernarizeOp(model)
##################################################################
import os
import argparse
import time
import sys
parser = argparse.ArgumentParser()
parser.add_argument('-dataset_name', type=str,default='NMNIST') #
parser.add_argument('-T', type=int,default=30)
parser.add_argument('-channels', type=int,default=2)
parser.add_argument('-split_by', type=str,default='number')
parser.add_argument('-normalization', type=str,default=None)
args = parser.parse_args()
argv = ' '.join(sys.argv)
#print(args)
dataset_name = args.dataset_name
dataset_dir = "/home/wuman/SNN/DiaNet+spiking/N-MNIST/dataset_dir"
T = args.T
train_loader = torch.utils.data.DataLoader(
    dataset=NMNIST(dataset_dir, train=True, use_frame=True, frames_num=10, split_by='number'), #
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
test_data_loader = torch.utils.data.DataLoader(
    dataset=NMNIST(root=dataset_dir, train=False, use_frame=True, frames_num=10, split_by='number'), #, normalization='max'
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=False,
        pin_memory=True)
##########################################################################
#net = Net().to(device)
base_lr=0.05
criteon = nn.CrossEntropyLoss()#定义损失函数
optimizer = optim.SGD(model.parameters(),lr=base_lr,momentum=Momentum,weight_decay=0.000001) 
#optimizer = optim.Adam(model.parameters(), lr=base_lr,betas=(0.9, 0.99),weight_decay=0.00001)
def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = base_lr * (0.5 ** (epoch // 10))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr  

for epoch in range(epochs):
    adjust_learning_rate(epoch)
    for batch_idx, (data, label) in enumerate (train_loader):
        data = data.cuda()
        label = label.cuda()
        logits = model(data)
        
        ternarize_op.Ternarization()
        optimizer.zero_grad()
        loss = criteon(logits, label)
        loss.backward()
        ternarize_op.Restore()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


    test_loss = 0
    correct = 0
    model.eval()
    ternarize_op.Ternarization()
    for batch_idx, (data, label) in enumerate (test_data_loader):
        data = data.cuda()
        logits = model(data)
        label = label.cuda()
        test_loss += criteon(logits, label).item()
        pred = logits.cpu().max(1)[1].to(device)
        correct += pred.eq(label.data).sum()

    #test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_data_loader.dataset),
        100. * correct / len(test_data_loader.dataset)))

#feature_1 = net.featuremap1.transpose(1,0).cuda()
#np.savetxt('feature_1.txt',feature_1.cuda().detach().cpu().clone().numpy(),fmt='%.04f')
model.eval()
test_loss = 0
correct = 0
#ternarize_op.Ternarization()
for batch_idx, (data, label) in enumerate (test_data_loader):
    data = data.cuda()
    logits = model(data).cuda()
    label = label.to(device)
    test_loss += criteon(logits, label).item()
   
    pred = logits.cpu().max(1)[1].cuda()
    correct += pred.eq(label.data).sum()
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_data_loader.dataset),
    100. * correct / len(test_data_loader.dataset)))
bias=open("/home/wuman/SNN/DiaNet+spiking/N-MNIST/bias.txt",'w+')
weight=open("/home/wuman/SNN/DiaNet+spiking/N-MNIST/weight.txt",'w+')
parameter=open("/home/wuman/SNN/DiaNet+spiking/N-MNIST/parameters.txt",'w+')
for name, parameters in model.state_dict().items():
    if "bias" in name:
        np.set_printoptions(suppress=True)
        parameters = parameters.cpu()
        parameters = parameters.numpy()
        print(name,':',parameters,file=bias)
for name, parameters in model.state_dict().items():
    if "weight" in name:
        np.set_printoptions(suppress=True)
        parameters = parameters.cpu()
        parameters = parameters.numpy()
        np.set_printoptions(threshold=70000)
        print(name,':',parameters,file=weight)
for name, parameters in model.state_dict().items():
    np.set_printoptions(suppress=True)
    parameters = parameters.cpu()
    parameters = parameters.numpy()
    print(name,':',"\n",parameters,file=parameter)
torch.save({
            'epoch': epoch,
            'net_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            }, "/home/wuman/SNN/DiaNet+spiking/N-MNIST/n_mnist.pkl" )




