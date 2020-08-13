#coding=utf-8
import argparse,os
import torch
from torch.autograd import Variable
import numpy as np
import time, math, glob
import scipy.io as sio
import matplotlib.pyplot as plt
import h5py
from skimage.measure import compare_ssim
from numpy.fft import fftn, ifftn, fftshift, ifftshift
from myNet.Model_Block.LDnCn import LDnCn4


def PSNR_3D(pred, gt):
    maxv = 1.0
    res = pred - gt
    # print(sum(res.flatten()))
    rmse = math.sqrt(np.mean(res ** 2))
    # print(f'rmse = {rmse}')
    if rmse == 0:
        return 100, 0
    psnr = 20 * math.log10(maxv / rmse)
    return psnr, rmse
#生成2通道的Mask
def genTenMask(T2Dn,mask_i):
    mask = np.zeros((T2Dn.shape[0],T2Dn.shape[1],mask_i.shape[-2],mask_i.shape[-1]))      
    for i in range(0,T2Dn.shape[0]):
        mask[i] = mask_i        
    mask_v = Variable(torch.from_numpy(mask).float())
    mask_v = mask_v.view(mask_v.shape[0],mask_v.shape[1],mask_v.shape[2],mask_v.shape[3],1)
    mask_v2 = torch.cat((mask_v,mask_v),-1)
    return mask_v2


def forward(t2Dn,t1_1,t1_2,t1_3,model,mask1,mask2,mask3):
    t2Dn = Variable(torch.from_numpy(t2Dn).contiguous().float())
    t2Dn= t2Dn.view(1, -1, t2Dn.shape[-2], t2Dn.shape[-1])      
    t2Dn =t2Dn.cuda()
    
    t1_3 = Variable(torch.from_numpy(t1_3).contiguous().float())
    t1_3 = t1_3.view(1, -1, t1_3.shape[-2], t1_3.shape[-1])      
    t1_3 = t1_3.cuda()
    
    tmask3 = genTenMask(t2Dn, mask3).cuda()
    
    y = model(t2Dn,t1_3,tmask3) 
    y = y.cpu().data[0].numpy()
#     y2 = y2.cpu().data[0].numpy()  # .astype(np.float64) # maybe 1 channel

    return y

def predict(model,imslices,label,t1_1s,t1_2s,t1_3s,slice_h=3):#slice_h=3
#     y_dim, z_dim = imslices.shape[2:]        
    mask1 = np.zeros((3,84,261))
    d_i = range(21,63)
    mask1[:,d_i] = 1
    mask1 = np.fft.ifftshift(mask1, axes=(2,1)) 
    
    mask2 = np.zeros((3,168,261))
    d_i = range(63,105)
    mask2[:,d_i] = 1
    mask2 = np.fft.ifftshift(mask2, axes=(2,1))
    
    mask3 = np.zeros((3,336,261))
    d_i = range(147,189)
    mask3[:,d_i] = 1
    mask3 = np.fft.ifftshift(mask3, axes=(2,1))
    
    imsize = (imslices.shape[0] + slice_h - 1, 336, 261)
    Y4,C,LY = np.zeros(imsize, np.single),np.zeros(imsize, np.single),np.zeros(imsize, np.single)
    for i in range(imslices.shape[0]):
        t2Dn = imslices[i, :]
        t1_1 = t1_1s[i, :]
        t1_2 = t1_2s[i, :]
        t1_3 = t1_3s[i, :]
        l_data = label[i,:]
        y4= forward(t2Dn,t1_1,t1_2,t1_3,model,mask1,mask2,mask3)
        C[i: (i + slice_h), :] += 1        
        Y4[i: (i + slice_h), :] += y4
        LY[i:(i + slice_h), :] += l_data
    
    Y4 = Y4 / C
    LY = LY / C
    np.clip(Y4, 0.0, 1.0, out=Y4)
    np.clip(LY, 0.0, 1.0, out=LY)
    
    return Y4,LY

def norm_img(img): 
    new_img = (img-np.min(img))/(np.max(img)-np.min(img)+0.000001)
    return new_img


def gen_fftData(data):
    data = data.view(data.shape+(1,)) 
    add = Variable(torch.zeros(data.shape).float()).to(data.device)
    data_k = torch.cat((data,add),-1)
    data_k = torch.fft(data_k,2,normalized=True)
    return data_k 



def test(opt1,modelPath,dataPath):
    global opt
    #opt = parser.parse_args()
    opt = opt1
    cuda = opt.cuda


    y_psnr1 = []
    y_psnr2 = []
    y_ssim1 = []
    y_ssim2 = []
    
    x = []
    best_psnr1 = 0
    best_psnr2 = 0
    best_ssim1 = 0
    best_ssim2 = 0
    
    psnr_i1=ssim_i1 = 0
    psnr_i2=ssim_i2 = 0

    h5file = h5py.File(dataPath,"r")
    dn_data = h5file["T2Dn"]
    t1_1 = h5file["T1_2"]
    t1_2 = h5file["T1_3"]
    t1_3 = h5file["T1_4"]

    label_3 = h5file["label_3"]

    model = LDnCn4()
 
    for i in range(opt.start_epoch,opt.max_epoch+1):          #进入模型循环
        model_path = modelPath + "model_epoch_{}.pth".format(i)
        print(model_path)  
#         model.load_state_dict(torch.load(model_path, map_location=lambda storage, loc: storage))
        model.load_state_dict(torch.load(model_path)) 
#     model = torch.load(model_path)["model"]
        if cuda:
#             model = model.cuda(int(opt.gpus))      
            model = model.cuda()  
        else:
            model = model.cpu()
     
        model.eval()
     
        HR1,T2 = predict(model,dn_data,label_3,t1_1,t1_2,t1_3)  
     
        HR1[T2 == 0] = 0
        psnr2, rmse2 = PSNR_3D(HR1,T2)
        ssim2 = compare_ssim(np.array(HR1), np.array(T2))
     
        HR1 = norm_img(HR1)
        T2 = norm_img(T2)
        psnr3, rmse3 = PSNR_3D(HR1, T2)
        ssim3 = compare_ssim(np.array(HR1), np.array(T2))
     
        print("epoch={},psnr2={:.5f},ssim2={:.5f},psnr3={:.5f},ssim3={:.5f}:".format(i,psnr2,ssim2,psnr3,ssim3))

        y_psnr1.append(psnr2)
        y_psnr2.append(psnr3)
        y_ssim1.append(ssim2)
        y_ssim2.append(ssim3)
 
        x.append(i)
        if psnr3 > best_psnr2:
            best_psnr2 = psnr3
            psnr_i2 = i
        if psnr2 > best_psnr1:
            best_psnr1 = psnr2
            psnr_i1 = i
        if ssim3 > best_ssim2:
            best_ssim2 = ssim3
            ssim_i2 = i
        if ssim2 > best_ssim1:
            best_ssim1 = ssim2
            ssim_i1 = i
    y_psnr1 = np.array(y_psnr1)
    data_file1 = opt.model + "PSNR1Test.npz"
    np.savez(data_file1, psnr=y_psnr1)
    y_psnr2 = np.array(y_psnr2)
    data_file2 = opt.model + "PSNR2Test.npz"
    np.savez(data_file2, psnr=y_psnr2)

    y_ssim1 = np.array(y_ssim1)
    data_file3 = opt.model + "SSIM1Test.npz"
    np.savez(data_file3, ssim=y_ssim1)
    y_ssim2 = np.array(y_ssim2)
    data_file4 = opt.model + "SSIM2Test.npz"
    np.savez(data_file4, ssim=y_ssim2)
   
    print("Save Done")
 
    print("max_epoch={},max_psnr2={:.5f},max_epoch={},max_ssim2={:.5f}".format(psnr_i1,best_psnr1,ssim_i1,best_ssim1))
    print("max_epoch={},max_psnr3={:.5f},max_epoch={},max_ssim3={:.5f}".format(psnr_i2,best_psnr2,ssim_i2,best_ssim2))