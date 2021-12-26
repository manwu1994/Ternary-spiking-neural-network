#############################################
#   @author: Youngeun Kim and Priya Panda   #
#############################################
#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch.optim as optim
import torchvision
from   torch.utils.data.dataloader import DataLoader
from   torchvision import transforms
from  vgg11  import * # vgg11 model_v2 vgg11
print("****************vgg11.py CIFAR100 with 8 timew_step*********************")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import argparse
import os.path
import numpy as np
import torch.backends.cudnn as cudnn
from utills import *

cudnn.benchmark = True
cudnn.deterministic = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
parser = argparse.ArgumentParser(description='SNN trained with BNTT', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--seed',                  default=25,        type=int,   help='Random seed')
parser.add_argument('--num_steps',             default=8,    type=int, help='Number of time-step')
parser.add_argument('--batch_size',            default=64,       type=int,   help='Batch size')
parser.add_argument('--lr',                    default=0.1,   type=float, help='Learning rate')
parser.add_argument('--leak_mem',              default=1.0,   type=float, help='Leak_mem')
parser.add_argument('--arch',              default='vgg11',   type=str, help='Dataset [vgg9, vgg11]')
parser.add_argument('--dataset',              default='cifar100',   type=str, help='Dataset [cifar10, cifar100]')
parser.add_argument('--num_epochs',            default=202,       type=int,   help='Number of epochs')
parser.add_argument('--num_workers',           default=4, type=int, help='number of workers')
parser.add_argument('--train_display_freq',    default=2, type=int, help='display_freq for train')
parser.add_argument('--test_display_freq',     default=2, type=int, help='display_freq for test')


global args
args = parser.parse_args()



#--------------------------------------------------
# Initialize tensorboard setting
#--------------------------------------------------
log_dir = 'modelsave'
if os.path.isdir(log_dir) is not True:
    os.mkdir(log_dir)


user_foldername = (args.dataset)+(args.arch)+'_timestep'+str(args.num_steps) +'_lr'+str(args.lr) + '_epoch' + str(args.num_epochs) + '_leak' + str(args.leak_mem)



#--------------------------------------------------
# Initialize seed
#--------------------------------------------------
seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

#--------------------------------------------------
# SNN configuration parameters
#--------------------------------------------------
# Leaky-Integrate-and-Fire (LIF) neuron parameters
leak_mem = args.leak_mem

# SNN learning and evaluation parameters
batch_size      = args.batch_size
batch_size_test = args.batch_size
num_epochs      = args.num_epochs
num_steps       = args.num_steps
lr   = args.lr


#--------------------------------------------------
# Load  dataset
#--------------------------------------------------
if args.dataset == 'cifar10':
        normalize   = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
elif args.dataset == 'cifar100':
        normalize   = transforms.Normalize((0.5071,0.4867,0.4408), (0.2675,0.2565,0.2761))


transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),normalize])

transform_test = transforms.Compose([
    transforms.ToTensor(),normalize])



if args.dataset == 'cifar10':
    num_cls = 10
    img_size = 32

    train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transform_test)
elif args.dataset == 'cifar100': #(0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)
    num_cls = 100
    img_size = 32

    train_set = torchvision.datasets.CIFAR100(root='./data', train=True,
                                            download=True, transform=transform_train)
    test_set = torchvision.datasets.CIFAR100(root='./data', train=False,
                                            download=True, transform=transform_test)
else:
    print("not implemented yet..")
    exit()



trainloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
testloader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True, drop_last=True)
#x, y = next(iter(trainloader))
#print(x.shape, y.shape,  x.min(), x.max())


#--------------------------------------------------
# Instantiate the SNN model and optimizer
#--------------------------------------------------
if args.arch == 'vgg9':
    model = SNN_VGG9_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
elif args.arch == 'vgg7':
    model = SNN_VGG7_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
elif args.arch == 'vgg11':
    model = SNN_VGG11_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
elif args.arch == 'vgg13':
    model = SNN_VGG13_BNTT(num_steps = num_steps, leak_mem=leak_mem, img_size=img_size,  num_cls=num_cls)
else:
    print("not implemented yet..")
    exit()

model = model.cuda()
#model = torch.nn.DataParallel(model)
#clipper = WeightClipper()
#model.apply(clipper)
#--------------------------------------------------
# TernarizeOp
#--------------------------------------------------

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

# Configure the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=args.lr,momentum=0.9,weight_decay=1e-4)
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
print('Batch size (testing)   : {}'.format(batch_size_test))
print('Number of epochs       : {}'.format(num_epochs))
print('Learning rate          : {}'.format(lr))

#--------------------------------------------------
# Train the SNN using surrogate gradients
#--------------------------------------------------
print('********** SNN training and evaluation **********')
train_loss_list = []
test_acc_list = []
for epoch in range(num_epochs):
    train_loss = AverageMeter()
    for i, data in enumerate(trainloader):
        inputs, labels = data
        inputs = inputs.cuda()
        labels = labels.cuda()
        #print(labels.size())
        optimizer.zero_grad()
        ternarize_op.Ternarization()
        output = model(inputs)
        
        loss   = criterion(output, labels)

        prec1, prec5 = accuracy(output, labels, topk=(1, 5))
        train_loss.update(loss.item(), labels.size(0))

        loss.backward()
        ternarize_op.Restore()
        optimizer.step()

    if (epoch+1) % args.train_display_freq ==0:
        print("Epoch: {}/{};".format(epoch+1, num_epochs), "########## Training loss: {}".format(train_loss.avg))

    adjust_learning_rate(optimizer, epoch, num_epochs)



    if (epoch+1) %  args.test_display_freq ==0:
        acc_top1, acc_top5 = [], []
        model.eval()
        ternarize_op.Ternarization()
        with torch.no_grad():
             for j, data in enumerate(testloader, 0):

                 images, labels = data
                 images = images.cuda()
                 labels = labels.cuda()
                 #print(labels)

                 out = model(images)
                 prec1, prec5 = accuracy(out, labels, topk=(1, 5))
                 acc_top1.append(float(prec1))
                 acc_top5.append(float(prec5))


        test_accuracy = np.mean(acc_top1)
        #print("threshold after step:{}".format(model.threshold1.data))
        print ("test_accuracy : {}". format(test_accuracy))

        # Model save
        if best_acc < test_accuracy:
            best_acc = test_accuracy

            model_dict = {
            
                    'global_step': epoch + 1,
                    'state_dict': model.state_dict(),
                    'accuracy': test_accuracy}

            torch.save(model_dict, log_dir+'/'+user_foldername+'_bestmodel_cifar100.pth.tar')
    #print(list(model.named_parameters()))
    #net_parameters = filter(lambda p: p.requires_grad, model.parameters())
    #params = sum([np.prod(p.size()) for p in net_parameters])
    #print("params",params)
    parameter=open(log_dir+'/'+user_foldername+"spiking_parameters.txt",'w+')
    for name, parameters in model.state_dict().items():
        np.set_printoptions(suppress=True)
        parameters = parameters.cpu()
        parameters = parameters.detach().numpy() 
        print(name,':',"\n",parameters,file=parameter)

#print("##############################################################")
sys.exit(0)


