import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import copy
import math
batch_size=64
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
def PoissonGen(inp, rescale_fac=2.0):
    rand_inp = torch.rand_like(inp).cuda()
    return torch.mul(torch.le(rand_inp , torch.abs(inp)).float(), torch.sign(inp))
    
class SNN_VGG9_BNTT(nn.Module):
    def __init__(self, num_steps, leak_mem=1.0, img_size=32,default_threshold = 1.0,num_cls=10):
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

        self.conv1 = nn.Conv2d(3, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt1 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt2 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        #self.pool1 = nn.AvgPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt3 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt4 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        #self.pool2 = nn.AvgPool2d(kernel_size=2)

        self.conv5 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt5 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.conv6 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt6 = nn.BatchNorm2d(128, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.conv7 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=bias_flag)
        self.bntt7 = nn.BatchNorm2d(256, eps=1e-4, momentum=0.8, affine=affine_flag) 
        
        #self.pool3 =  nn.AvgPool2d(kernel_size=2)
        self.pool3 = nn.MaxPool2d(2,stride=2)


        self.fc1 = nn.Linear((self.img_size//2)*(self.img_size//2)*256, 1024, bias=bias_flag)
        self.bntt_fc = nn.BatchNorm1d(1024, eps=1e-4, momentum=0.8, affine=affine_flag) 
        self.fc2 = nn.Linear(1024, self.num_cls, bias=bias_flag)

       
        self.fc_list=[self.fc1,self.fc2]
        self.threshold1=nn.Parameter(torch.tensor(default_threshold))
        self.threshold2=nn.Parameter(torch.tensor(default_threshold))
        self.threshold3=nn.Parameter(torch.tensor(default_threshold))
        self.threshold4=nn.Parameter(torch.tensor(default_threshold))
        self.threshold5=nn.Parameter(torch.tensor(default_threshold))
        self.threshold6=nn.Parameter(torch.tensor(default_threshold))
        self.threshold7=nn.Parameter(torch.tensor(default_threshold))
        self.threshold10=nn.Parameter(torch.tensor(default_threshold))
        self.dropout = nn.Dropout(0.20)
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

        batch_size = inp.size(0)
        mem_conv1 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv2 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv3 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv4 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv5 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv6 = torch.zeros(batch_size, 128, self.img_size, self.img_size).cuda()
        mem_conv7 = torch.zeros(batch_size, 256, self.img_size, self.img_size).cuda()
        #mem_conv_list = [mem_conv1, mem_conv2, mem_conv3, mem_conv4, mem_conv5, mem_conv6, mem_conv7]

        mem_fc1 = torch.zeros(batch_size, 1024).cuda()
        mem_fc2 = torch.zeros(batch_size, self.num_cls).cuda()


        for t in range(self.num_steps):
            
            
            #rand_inp = torch.rand_like(inp).cuda()
            #spike_inp = torch.mul(torch.le(rand_inp , torch.abs(inp)).float(), torch.sign(inp)) ##le means smaller and equal
            #spike_inp = PoissonGen(inp)
            #out_prev = spike_inp
            mem_conv1 = self.leak_mem* mem_conv1 + self.bntt1(self.conv1(inp))
            mem_thr = (mem_conv1 / self.threshold1 - 1.0)
            out = self.spike_fn(mem_thr)
            #rst = torch.zeros_like(mem_conv1).cuda()
            rst = self.threshold1* (mem_thr>0).float() # (mem_thr>0) return 1
            mem_conv=mem_conv1.clone()
            mem_conv1 = mem_conv - rst
            #print("mem_conv1 - rst",mem_conv1.size())
            out_prev1 = out.clone()
            #print("out_prev1",out_prev1.size())
            #self.conv2.weight.data =torch.tanh(self.conv2.weight.data)
            mem_conv2 = self.leak_mem * mem_conv2 + self.bntt2(self.conv2(out_prev1))
            #print("mem_conv2",mem_conv2.size())
            mem_thr = (mem_conv2 / self.threshold2) - 1.0  ###mem_conv2 compare with threshold
            out = self.spike_fn(mem_thr)
            rst = self.threshold2* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_conv2).cuda()
            #rst[mem_thr > 0] = self.threshold2  ###soft reset:
            mem_conv2 = mem_conv2 - rst ###rst=0 means mem_thr<0,means mem_conv2<threshold,thus:mem_conv2 not change
            out_prev2 = out.clone()  ###rst=threshold means mem_thr>0,means mem_conv2>threshold,thus:mem_conv2 =0 or mem_conv2 - threshold 
            #self.conv3.weight.data =torch.tanh(self.conv3.weight.data)
            mem_conv3 = self.leak_mem * mem_conv3 + self.bntt3(self.conv3(out_prev2+out_prev1))
            mem_thr = (mem_conv3 / self.threshold3) - 1.0
            out = self.spike_fn(mem_thr)
            rst = self.threshold3* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_conv3).cuda()
            #rst[mem_thr > 0] = self.threshold3
            mem_conv3 = mem_conv3 - rst
            out_prev3 = out.clone()
            out_prev33=out_prev3+out_prev1
            #self.conv4.weight.data =torch.tanh(self.conv4.weight.data)
            mem_conv4 = self.leak_mem * mem_conv4 + self.bntt4(self.conv4(out_prev33))
            mem_thr = (mem_conv4 / self.threshold4) - 1.0
            out = self.spike_fn(mem_thr)
            rst = self.threshold4* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_conv4).cuda()
            #rst[mem_thr > 0] = self.threshold4
            mem_conv4 = mem_conv4 - rst
            out_prev4 = out.clone()
            out_prev44=out_prev4+out_prev3
            #self.conv5.weight.data =torch.tanh(self.conv5.weight.data)
            mem_conv5 = self.leak_mem * mem_conv5 + self.bntt5(self.conv5(out_prev44))
            mem_thr = (mem_conv5 / self.threshold5) - 1.0
            out = self.spike_fn(mem_thr)
            rst = self.threshold5* (mem_thr>0).float()
            #rst = torch.zeros_like(mem_conv5).cuda()
            #rst[mem_thr > 0] = self.conv5.threshold5
            mem_conv5 = mem_conv5 - rst
            out_prev5 = out.clone()
            out_prev55=out_prev5+out_prev4
            mem_conv6 = self.leak_mem * mem_conv6 + self.bntt6(self.conv6(out_prev55))
            mem_thr = (mem_conv6 / self.threshold6) - 1.0
            out = self.spike_fn(mem_thr)
            rst = self.threshold6* (mem_thr>0).float()
            rst = torch.zeros_like(mem_conv6).cuda()
            rst[mem_thr > 0] = self.threshold6
            mem_conv6 = mem_conv6 - rst
            out_prev6 = out.clone()
            out_prev66=out_prev6+out_prev5
            mem_conv7 = self.leak_mem * mem_conv7 + self.bntt7(self.conv7(out_prev66))
            mem_thr = (mem_conv7 / self.threshold7) - 1.0
            out = self.spike_fn(mem_thr)
            rst = self.threshold7* (mem_thr>0).float()
            rst = torch.zeros_like(mem_conv7).cuda()
            rst[mem_thr > 0] = self.threshold7
            mem_conv7 = mem_conv7 - rst
            out_prev7 = out.clone()
            
            
            out_pool3=self.pool3(out_prev7)
            #print("out_pool3",out_pool3.size())
            out_pool3 = self.dropout(out_pool3)
            #out_pool3=Pooling_sNeuron(out_pool3,0.2)
            #print(out_prev8.size())
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