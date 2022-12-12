# import time #clock of your computer
import torch #operated by facebook with a lot of library
import torch.nn as nn
# import torchvision.datasets as datasets
# import torch.optim as optim
# import torchvision.transforms as transforms
# from torch.utils.data import DataLoader
# from torch.autograd import Variable
# !pip install torchsummaryX --quiet
# from torchsummaryX import summary as summaryX
# from torchsummary import summary
#from torch.utils.tensorboard import SummaryWriter
#writer = SummaryWriter()

# class LeNet(nn.Module):
#   def __init__(self):
#     super(LeNet, self).__init__()

#     self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 6, 
#                            kernel_size = 5, stride = 1, padding = 0)
#     self.conv2 = nn.Conv2d(in_channels = 6, out_channels = 16, 
#                            kernel_size = 5, stride = 1, padding = 0)
#     self.conv3 = nn.Conv2d(in_channels = 16, out_channels = 120, 
#                            kernel_size = 5, stride = 1, padding = 0)
#     self.linear1 = nn.Linear(120, 84)
#     self.linear2 = nn.Linear(84, 10)
#     self.tanh = nn.Tanh()
#     self.avgpool = nn.AvgPool2d(kernel_size = 2, stride = 2)

#   def forward(self, x):
#     x = self.conv1(x)
#     x = self.tanh(x)
#     x = self.avgpool(x)
#     x = self.conv2(x)
#     x = self.tanh(x)
#     x = self.avgpool(x)
#     x = self.conv3(x)
#     x = self.tanh(x)
    
#     x = x.reshape(x.shape[0], -1)
#     x = self.linear1(x)
#     x = self.tanh(x)
#     x = self.linear2(x)
#     return x
class AlexNet(nn.Module):
    def __init__(self, num_classes = 10 ) :
        super(AlexNet, self). __init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3,96,kernel_size = 11 , stride = 4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(96,256,kernel_size=5, stride =1 , padding = 2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(256,384,kernel_size = 3, stride =1 , padding = 1),
            nn.BatchNorm2d(384), 
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(384,384,kernel_size = 3 , stride = 1, padding =1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            nn.Conv2d(384,256,kernel_size= 3, stride = 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride =2 ))
        self.fc ==nn. Sequential(
            nn.Dropout(0.5), 
            nn.Linear(9216,4096),
            nn.ReLU())
        self.fc1 == nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, num_classes))
    def forward(self,x):
        out = self.layer1(x)
        out= self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1 )
        out = self.fc(out)
        out = self.fc1 (out)
        out = self.fc2(out)
        return out
class VGG16(nn.Module):
    def __init__(self,num_classes=10):
        super(VGG16,self).__init__()
        self.layer1= nn.Sequential(
            nn.Conv2d(3,64, kernel_size = 3, stride =1 , padding = 1),
            nn.BatchNormd2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64,64,kernel_size=3, stride =1, padding = 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2 ))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64,128, kernel_size = 3, stride= 1 ,padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128,128,kernel_size = 3, stride =1 , padding =1 ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128,256,kernel_size = 3, stride =1 ,padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256,256,kernel_size = 3,stride =1 , padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 =nn.Sequential(
            nn.Conv2d(256,256,kernel_size= 3 ,stride = 1,padding =1 ),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size =2, stride= 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256,512,kernel_size =3,stride =1, padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3, stride =1 ,padding =1 ),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size = 3,stride =1, padding =1 ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride =2 ))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size= 3, stride =1, padding =1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size=3, stride =1, padding =1 ),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512,512,kernel_size = 3, stride =1 , padding =1 ),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096) ,
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096,4096),
            nn.ReLU())
        self.fc2 = nn.Sequential(
            nn.Linear(4096,num_classes))
    def forward(self,x):
        out= self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out= self.layer4(out)
        out= self.layer5(out)
        out = self.layer6(out)
        out= self.layer7(out)
        out=self.layer8(out)
        out=self.layer9(out)
        out=self.layer10(out)
        out=self.layer11(out)
        out=self.layer12(out)
        out=self.layer13(out)
        out= out.reshape(out.size(0),-1)
        out=self.fc(out)
        out= self.fc1(out)
        out= self.fc2(out)
        return out
            
