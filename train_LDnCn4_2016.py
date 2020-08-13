import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from DataLoader.dataset_LapSRNDCMRI_T1_T2Dn_T2 import DatasetFromHdf5
from torch.utils.data import DataLoader
from myNet.Model_Block.LDnCn import LDnCn4
import numpy as np
import logging



def startTrain(opt1,dataPath):

    global opt, model,logger
    opt = opt1
    print(opt)
    model_folder = "MSSEGModel/TestBlock/LDnCn{0}/LDnCn{0}_Dn{1}_lr{2}_step{3}_{4}_T1_T2Dn_MSSEG2016_Sub({5})/".format(opt.modelT,opt.Dn,opt.lr,opt.step,opt.optim,opt.Sub)
    
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)       
    logger = get_logger("MSSEGModel/TestBlock/LDnCn{0}/LDnCn{0}_Dn{1}_lr{2}_step{3}_{4}_T1_T2Dn_MSSEG2016_Sub({5})/train.log".format(opt.modelT,opt.Dn,opt.lr,opt.step,opt.optim,opt.Sub))

    cuda = opt.cuda                                            

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True

    print("===> Loading datasets")
    #train_set = DatasetFromHdf5("/home/wks/users/lxj/MRI/pytorch-LapSRN-master/data/S08.Sub(2-5).Slice3.T2-T1-T2Dn8.DnCnPyFft.h5")
#     train_set = DatasetFromHdf5("/home/wks/users/lyw/lxx/S08.Sub(2-5).Slice3.T2-T1-T2Dn8.lapSRN2.h5")
#     train_set = DatasetFromHdf5("./data/lap_pry_x4_small.h5")
    train_set = DatasetFromHdf5(dataPath)
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

    print("===> Building model")
    model = LDnCn4()

    criterion = nn.MSELoss()

    print("===> Setting GPU")
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()
    else:
        model = model.cpu() 

    print("===> Setting Optimizer")
#     optimizer = optim.Adam(model.parameters(), lr=opt.lr, betas=(0.5, 0.999))
    if opt.optim == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    elif opt.optim =='SGD':
        optimizer = optim.SGD(model.parameters(),lr=opt.lr,momentum=opt.momentum,weight_decay=opt.weight_decay)
    
    #betas (Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))

    print("===> Training")
    logger.info('start training!')
    for epoch in range(opt.start_epoch, opt.nEpochs + 1): 
        train(training_data_loader, optimizer, model, criterion, epoch)
        model_folder = save_checkpoint(model, epoch)
    logger.info("finish training!")
    return model_folder

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
#     lr = opt.lr * (0.1 ** (epoch // opt.step))
    lr = opt.lr * (0.1 ** (epoch // opt.step))
#     if epoch == 0 or epoch % 40:
#         return opt.lr
#     opt.lr = opt.lr/ 10.0
    return lr


def genTenMask(T2Dn,mask_i):
    mask = np.zeros((T2Dn.shape[0],T2Dn.shape[1],mask_i.shape[-2],mask_i.shape[-1]))      
    for i in range(0,T2Dn.shape[0]):
        mask[i] = mask_i        
    mask_v = Variable(torch.from_numpy(mask).float())
    mask_v = mask_v.view(mask_v.shape[0],mask_v.shape[1],mask_v.shape[2],mask_v.shape[3],1)
    mask_v2 = torch.cat((mask_v,mask_v),-1)
    return mask_v2

def train(training_data_loader, optimizer, model, criterion, epoch):

    lr = adjust_learning_rate(optimizer, epoch-1)

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

    
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))

    model.train()
    
    mask3 = np.zeros((3,336,261))
    d_i = range(147, 189)
    mask3[:,d_i] = 1
    mask3 = np.fft.ifftshift(mask3, axes=(2,1))
   

    for iteration, batch in enumerate(training_data_loader, 1):

        T2Dn,T1_1,T1_2,T1_3,label_1,label_2,label_3 = Variable(batch[0]),Variable(batch[1],requires_grad=False),Variable(batch[2],requires_grad=False),Variable(batch[3],requires_grad=False),Variable(batch[4],requires_grad=False),Variable(batch[5],requires_grad=False),Variable(batch[6],requires_grad=False)
        
        tmask3 = genTenMask(T2Dn, mask3)      

        
        if opt.cuda:
            tmask3 = tmask3.cuda()
            T1_3 = T1_3.cuda()
            T2Dn = T2Dn.cuda()
            label_3 = label_3.cuda()

        optimizer.zero_grad()
        
        HR = model(T2Dn,T1_3,tmask3)

        loss = criterion(HR,label_3)
    
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
        optimizer.step()

        if iteration%20 == 0:
            logger.info("===> Epoch[{}]({}/{}):\t Loss: {:.10f}\t".format(epoch, iteration, len(training_data_loader), loss.data))
#             print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data))
#             print("===> Epoch[{}]({}/{}): Loss: {:.10f}".format(epoch, iteration, len(training_data_loader), loss.data[0]))

def save_checkpoint(model, epoch):
#     model_folder = "StochasticD5C5_0.001_40/"
#     model_folder = "D5C5_pytorchFFt_0.001_40/"
    model_folder = "MSSEGModel/TestBlock/LDnCn{0}/LDnCn{0}_Dn{1}_lr{2}_step{3}_{4}_T1_T2Dn_MSSEG2016_Sub({5})/".format(opt.modelT,opt.Dn,opt.lr,opt.step,opt.optim,opt.Sub)
    model_out_path = model_folder + "model_epoch_{}.pth".format(epoch)
    if not os.path.exists(model_folder):
        os.makedirs(model_folder)
    torch.save(model.state_dict(),model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))
    return model_folder


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger

# if __name__ == "__main__":
#     main()
