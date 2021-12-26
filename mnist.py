import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import math
from torchvision import datasets,transforms
import torchvision
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Surrogate_BP_Function(torch.autograd.Function):


    @staticmethod
    def forward(self, input):
        self.save_for_backward(input)
        return input.ge(0).type(torch.cuda.FloatTensor)

    @staticmethod
    def backward(self, grad_output):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad = grad_input * 0.3 * F.threshold(1.0 - torch.abs(input), 0, 0)*2
        return grad

num_steps=6
leak_mem=1.0
batch_size=128
lr=0.01
num_epochs=100
class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=1.0, img_size=28,default_threshold = 1.0,num_cls=10):
        super(SNN_VGG9_BNTT, self).__init__()

        self.img_size = img_size
        self.num_cls = num_cls
        self.num_steps = num_steps
        self.spike_fn = Surrogate_BP_Function.apply
        self.leak_mem = leak_mem
        self.batch_num = self.num_steps
        print (">>>>>>>>>>>>>>>>>>> VGG 9 >>>>>>>>>>>>>>>>>>>>>>")
        print ("***** time step per batchnorm".format(self.batch_num))
        print (">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        affine_flag = True
        bias_flag = True

        self.conv1 = nn.Conv2d(1, 6, 3, 1, 2, bias=bias_flag)
        self.bntt1 = nn.BatchNorm2d(6, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(6, 16, 5, bias=bias_flag)
        self.bntt2 = nn.BatchNorm2d(16, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.pool2 = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(400,120, bias=bias_flag)
        self.bntt_fc = nn.BatchNorm1d(120, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.fc2 = nn.Linear(120,10, bias=bias_flag)

        self.threshold1=nn.Parameter(torch.tensor(default_threshold))
        self.threshold2=nn.Parameter(torch.tensor(default_threshold))
        self.threshold3=nn.Parameter(torch.tensor(default_threshold))

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


    def forward(self, inp):

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 6, 30, 30).cuda()
        mem_conv2 = torch.zeros(batch_size, 16, 11, 11).cuda()

        mem_fc1 = torch.zeros(batch_size, 120).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()


        for t in range(self.num_steps):
            
            
            rand_inp = torch.rand_like(inp).cuda()
            spike_inp = torch.mul(torch.le(rand_inp , torch.abs(inp)).float(), torch.sign(inp)) ##le means smaller and equal
            #spike_inp = PoissonGen(inp)
            #out_prev = spike_inp
            #print(spike_inp.size())
            mem_conv1 = self.leak_mem* mem_conv1 + self.bntt1(self.conv1(spike_inp))
            mem_thr = (mem_conv1 / self.threshold1 - 1.0)
            out = self.spike_fn(mem_thr)
            rst = self.threshold1* (mem_thr>0).float() # (mem_thr>0) return 1
            mem_conv=mem_conv1.clone()
            mem_conv1 = mem_conv - rst
            out_prev1 = out.clone()
            #print(out_prev1.size())
            out_pool1=self.pool1(out_prev1)
            #print("out_pool1",out_pool1.size())
            mem_conv2 = self.leak_mem * mem_conv2 + self.bntt2(self.conv2(out_pool1))
            mem_thr = (mem_conv2 / self.threshold2) - 1.0  
            out = self.spike_fn(mem_thr)
            rst = self.threshold2* (mem_thr>0).float()
            mem_conv2 = mem_conv2 - rst ###rst=0 means mem_thr<0,means mem_conv2<threshold,thus:mem_conv2 not change
            out_prev2 = out.clone()  ###rst=threshold means mem_thr>0,means mem_conv2>threshold,thus:mem_conv2 =0 or mem_conv2 - threshold 
            #print("out_prev2",out_prev2.size())
            out_pool2=self.pool2(out_prev2)
            out_pool3 = out_pool2.reshape(batch_size, -1)
            #print("out_pool3",out_pool3.size())
            mem_fc1 = self.leak_mem * mem_fc1 + self.bntt_fc(self.fc1(out_pool3))  ### the last layer input
            mem_thr = (mem_fc1 / self.threshold3) - 1.0  ###
            out = self.spike_fn(mem_thr)
            rst = self.threshold3* (mem_thr>0).float()
            mem_fc1 = mem_fc1 - rst
            out_prev9 = out.clone()
            mem_fc=self.fc2(out_prev9)
            mem_fc2 = mem_fc2 + mem_fc

        out_voltage = mem_fc2 / self.num_steps


        return out_voltage

model = SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=28,  num_cls=10)
model.to(device)

train_db = datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,)),
                   ]))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)

test_db = datasets.MNIST('../data', train=False, transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))]))
test_loader = torch.utils.data.DataLoader(test_db,
    batch_size=10000, shuffle=True)
train_db, val_db = torch.utils.data.random_split(train_db, [50000, 10000])
#print('db1:', len(train_db), 'db2:', len(val_db))
train_loader = torch.utils.data.DataLoader(
    train_db,
    batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(
    val_db,
    batch_size=batch_size, shuffle=True)

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
            #self.saved_alpha=self.alpha[:]
            

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

# Configure the loss function and optimizer
criteon = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr,momentum=0.9,weight_decay=1e-4)
#optimizer = optim.Adam(model.parameters(), lr=args.lr,betas=(0.9, 0.99),weight_decay=1e-4)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)
best_acc = 0

# Print the SNN model, optimizer, and simulation parameters
print('********** SNN simulation parameters **********')
print('Simulation # time-step : {}'.format(num_steps))
print('Membrane decay rate : {0:.2f}\n'.format(leak_mem))

print('********** SNN learning parameters **********')
print('Backprop optimizer     : SGD')
print('Batch size (training)  : {}'.format(batch_size))
print('Number of epochs       : {}'.format(num_epochs))
print('Learning rate          : {}'.format(lr))

#--------------------------------------------------
# Train the SNN using surrogate gradients
#--------------------------------------------------
print('********** SNN training and evaluation **********')
for epoch in range(num_epochs):
    #adjust_learning_rate(epoch)
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.cuda()

        logits = model(data)
        loss = criteon(logits, target)

        optimizer.zero_grad()
        ternarize_op.Ternarization()
        loss.backward()
        ternarize_op.Restore()
        optimizer.step()
        if batch_idx % 100 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))

    #model.eval()
    test_loss = 0
    correct = 0
    model.eval()
    ternarize_op.Ternarization()
    for data, target in val_loader:
        data, target = data.to(device), target.cuda()
        logits = model(data)
        test_loss += criteon(logits, target).item()

        pred = logits.data.max(1)[1]
        correct += pred.eq(target.data).sum()

    test_loss /= len(val_loader.dataset)
    print('\nVAL set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset),
        100. * correct / len(val_loader.dataset)))
    
model.eval()
test_loss = 0
correct = 0
ternarize_op.Ternarization()
for data, target in test_loader:
    data, target = data.to(device), target.cuda()
    logits = model(data)
    test_loss += criteon(logits, target).item()

    pred = logits.data.max(1)[1]
    correct += pred.eq(target.data).sum()
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
print("--------------------------------")   