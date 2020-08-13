import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
# from myNet.CRNN import DataConsistencyInKspace
# from myNet.CRNN_At import RG_Block2
# from CRNN import RCN,Recursive_Block,RCN2,DataConsistencyInKspace
# from CRNN import DataConsistencyInKspace
# from CRNN_At import RG_Block2
import torch.nn.functional as F


def gendc_data(data,masklist,dn=8):
    dc_data = []
    x = data.clone()
    dn_len = masklist[-1].shape[-3]//dn
    x = x.view(x.shape[0],x.shape[1],x.shape[2],x.shape[3],1)   #(n, 3, nx, ny,1)
    add = torch.zeros((x.shape)).float().cuda()
    x = torch.cat((x,add),-1)
        
    x = torch.fft(x,2,normalized=True)  
    for i in range(len(masklist)):
        dc = torch.zeros(masklist[i].shape).float().cuda()
            #从data与mask的shape中生成应该生成的大小
        dc[:,:,0:dn_len//2,:,:] = x[:,:,0:dn_len//2,:,:]
        dc[:,:,-dn_len//2:,:,:] = x[:,:,-dn_len//2:,:,:]
        dc_data.append(dc)
    return dc_data


def lrelu():
    return nn.LeakyReLU(0.01, inplace=True)


def relu():
    return nn.ReLU(inplace=True)



#CBAM网络结构中的通道注意力，与SE的区别：增加了MaxPool分支
class CAM(nn.Module):
    def __init__(self,in_c,reduction):
        super(CAM,self).__init__()
        self.ap = nn.AdaptiveAvgPool2d((1, 1))
        self.mp = nn.AdaptiveMaxPool2d((1, 1))
        self.fc = nn.Sequential(
            nn.Conv2d(in_c,in_c//reduction,kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_c//reduction,in_c,kernel_size=1),
            )
        self.sg = nn.Sigmoid()
    def forward(self,x):
        y1 = self.ap(x)
        y1 = self.fc(y1)
        
        y2 = self.mp(x)
        y2 = self.fc(y2)
        y = self.sg(y1+y2)
        return x*y
#CBAM网络结构中的空间注意力    
class SAM(nn.Module):
    def __init__(self,ks=7):
        super(SAM,self).__init__()
        assert ks%2==1,"kernel_size = {}".format(ks)
        pad = (ks-1)//2
        
        self.__layer = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=ks, padding=pad),
            nn.Sigmoid(),
        )
    def forward(self,x):
        y1 = torch.mean(x,dim=1,keepdim=True)
        y2, _  = torch.max(x,dim=1,keepdim=True)
        
        mask = torch.cat((y1,y2),dim=1)
        mask = self.__layer(mask)
        return x*mask
#CBAM网络结构中注意机制与RCAN中的短跳跃连接模块结合起来
class CBAM_Block(nn.Module):
    def __init__(self,in_channels,reduction):
        super(CBAM_Block,self).__init__()
        self.rcab=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            CAM(in_channels, reduction),
            SAM()
        )
    def forward(self,x):
        return x+self.rcab(x) 


class RG_Block2(nn.Module):
    def __init__(self,in_channels,num_crab,reduction):
        super(RG_Block2,self).__init__()
#         print("RG_Block2")
        self.rg_block=[CBAM_Block(in_channels,reduction) for _ in range(num_crab)]
        self.rg_block.append(nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1))
        self.rg_block=nn.Sequential(*self.rg_block)
    def forward(self,x):
        return x+self.rg_block(x)
    
#CBAM网络结构中注意机制与RCAN中的短跳跃连接模块结合起来
class CBAM_Block(nn.Module):
    def __init__(self,in_channels,reduction):
        super(CBAM_Block,self).__init__()
        self.rcab=nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            CAM(in_channels, reduction),
            SAM()
        )
    def forward(self,x):
        return x+self.rcab(x)   

                       
#采用CBAM注意力模块
class LDnCn4(nn.Module):
    def __init__(self,in_c=6,out_c=3,nd=5,cn=5,RGn=5,reduction=16,nf=32,dn=8):
        super(LDnCn4,self).__init__()
        print("init LDnCn BType=4")
        self.inputl = []
        self.outs = []
        self.RGS = []
        self.nd = nd
#         self.block1 = RG_Block if type==1 else RG_Block2
        self.block1 = RG_Block2
        for i in range(nd):
            self.inputl.append(nn.Conv2d(in_c,nf,kernel_size=3,stride=1,padding=1))
            self.outs.append(nn.Conv2d(nf,out_c,kernel_size=3,stride=1,padding=1))
            self.RGS.append(self.block1(nf,RGn,reduction))
        self.inputl = nn.ModuleList(self.inputl)
        
        self.outs = nn.ModuleList(self.outs)
        self.RGS = nn.ModuleList(self.RGS)
        self.dc = DataConsistencyInKspace(norm='ortho')
        
    def genSC(self,data,mask):
        new_data = torch.zeros(mask.shape).float().to(data.device)       
        idata = data.clone()
        idata = idata.view(idata.shape+(1,))
        add = torch.zeros(idata.shape).float().to(data.device)
        idata = torch.cat((idata,add),-1)
        fdata = torch.fft(idata,2,normalized=True)
        new_data[:,:,:data.shape[-2]//2,:,:] = fdata[:,:,:data.shape[-2]//2,:,:] 
        new_data[:,:,-data.shape[-2]//2:,:,:] = fdata[:,:,-data.shape[-2]//2:,:,:] 
        new_data = torch.ifft(new_data,2,normalized=True)
        new_data = new_data.narrow(-1,0,1)
        new_data = torch.squeeze(new_data,-1)
        new_data = new_data.contiguous()
        return new_data
    
    def forward(self,t2,t1,mask):
        dc_data = gendc_data(t2, [mask], 8)
        t2 = self.genSC(t2, mask)
        y = t2.clone()
                
        for i in range(self.nd):
            y = torch.cat((y,t1),1)
            y = self.inputl[i](y)
            y = self.RGS[i](y)
            y = self.outs[i](y)
            y = y+t2
            y = self.dc.perform(y, dc_data[0], mask)
        return y  
# net = LDnCn4().cuda()
# print("net:",net)
# t2 = torch.ones(1,3,42,261).cuda()
# t1 = torch.zeros(1,3,336,261).cuda()
# mask =  torch.zeros(1,3,336,261,2).cuda()    
# y = net(t2,t1,mask)
# print(y.shape)   


