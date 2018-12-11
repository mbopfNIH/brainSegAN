#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#CUDA_VISIBLE_DEVICES=X python train_basemodel.py --cuda --outpath ./outputs
#from __future__ import print_function #MWB
from __future__ import division, print_function, unicode_literals
import argparse
import os
#from glob import glob
#from tqdm import trange
#from itertools import chain
import numpy as np
#from PIL import Image
import torch
import torch.nn.parallel
#from torch.utils import data
import torch.backends.cudnn as cudnn
import torch.optim as optim
#from torch import nn
import torch.nn.parallel
#import torchvision.utils as vutils
from torch.autograd import Variable
from transform import ReLabel, ToLabel, Scale, Colorize, HorizontalFlip, VerticalFlip
#from torchvision.transforms import Compose, CenterCrop, Normalize, ToTensor
import torch.nn.functional as F
# from net import NetG, NetD
from net import NetD, NetG
from LoadData import Dataset, loader, Dataset_val
from logger import Logger
#from torch.optim.optimizer import Optimizer

# MWB - debug
import sys
print('__Python VERSION:', sys.version)
print('__pyTorch VERSION:', torch.__version__)
print('__CUDNN VERSION:', torch.backends.cudnn.version())
print('__Number CUDA Devices:', torch.cuda.device_count())
print('Active CUDA Device: GPU', torch.cuda.current_device())
print ('Available devices ', torch.cuda.device_count())
# MWB - end debug

# Training settings
parser = argparse.ArgumentParser(description='An Example')
parser.add_argument('--batchsize', type=int, default=15, help='training batch size')
#parser.add_argument('--testBatchSize', type=int, default=10, help='testing batch size')
parser.add_argument('--niter', type=int, default=500, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.01, help='Learning Rate. Default=0.02')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--decay', type=float, default=0.5, help='Learning rate decay. default=0.5')
parser.add_argument('--cuda', action='store_true', help='use cuda?')
parser.add_argument('--adversarial', action='store_true', help='adversarial training?')
#parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=666, help='random seed to use. Default=666')
parser.add_argument('--outpath', default='./SegAN', help='folder to output images and model checkpoints')
opt = parser.parse_args()

print(opt)

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('ConvTranspose2d') != -1:
#         m.weight.data.normal_(0.0, 0.02)
#     # elif classname.find('BatchNorm') != -1:
#     #     m.weight.data.normal_(1.0, 0.02)
#     #     m.bias.data.fill_(0)

# def dice_loss(input,target):
#     """
#     input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
#     target is a 1-hot representation of the groundtruth, shoud have same size as the input
#     """
#     assert input.size() == target.size(), "Input sizes must be equal."
#     assert input.dim() == 4, "Input must be a 4D Tensor."
#     # uniques=np.unique(target.data.numpy())
#     # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"
#     dice_temp = 0
#     for i in range(input.size()[0]):
#         probs = input[i]
#         num=probs*target[i]#b,c,h,w--p*g
#         num=torch.sum(num,dim=1)
#         num=torch.sum(num,dim=1)#b,c
#
#         den1=probs*probs#--p^2
#         den1=torch.sum(den1,dim=1)
#         den1=torch.sum(den1,dim=1)#b,c,1,1
#
#         den2=target[i]*target[i]#--g^2
#         den2=torch.sum(den2,dim=1)
#         den2=torch.sum(den2,dim=1)#b,c,1,1
#
#         dice=2*(num/(den1+den2))
#         # dice_eso=dice[:,1]#we ignore bg dice val, and take the fg
#
#         dice_total=1-1*torch.sum(dice)/dice.size(0)#divide by batch_sz
#         dice_temp += dice_total
#     dice_temp /= input.size()[0]
#
#     return dice_total
def dice_loss(input,target):
    """
    input is a torch variable of size BatchxnclassesxHxW representing log probabilities for each class
    target is a 1-hot representation of the groundtruth, shoud have same size as the input
    """
    assert input.size() == target.size(), "Input sizes must be equal."
    assert input.dim() == 4, "Input must be a 4D Tensor."
    # uniques=np.unique(target.data.numpy())
    # assert set(list(uniques))<=set([0,1]), "target must only contain zeros and ones"
    probs = input
    num=probs*target#b,c,h,w--p*g
    num=torch.sum(num,dim=2)
    num=torch.sum(num,dim=2)#b,c

    den1=probs*probs#--p^2
    den1=torch.sum(den1,dim=2)
    den1=torch.sum(den1,dim=2)#b,c,1,1

    den2=target*target#--g^2
    den2=torch.sum(den2,dim=2)
    den2=torch.sum(den2,dim=2)#b,c,1,1

    dice=2*(num/(den1+den2))
    # dice_eso=dice[:,1]#we ignore bg dice val, and take the fg

    dice_score=1-1*torch.sum(dice,dim=0)/dice.size(0)#divide by batch_sz
    dice_total = dice_score[0] + 3 * dice_score[1] + 1 * dice_score[2] +  2 * dice_score[3] + 1.5 * dice_score[4]
    # print(dice_score.type)

    return dice_total, dice_score

def sample_gumbel(input):
    noise = torch.rand(input.size())
    eps = 1e-20
    noise.add_(eps).log_().neg_()
    noise.add_(eps).log_().neg_()
    if cuda:
        return Variable(noise).cuda()
    else:
        return Variable(noise)

def gumbel_softmax_sample(input, temperature):
    temperature = temperature
    noise = sample_gumbel(input)
    x = (input + noise) / temperature
    assert x.dim() == 4, 'Softmax2d requires a 4D tensor as input'
    x = F.softmax(x)
    return x.view_as(input)

# def sigmoid(input, k):
#     size = input.size()
#
#     sigmoid = torch.ones(size)/torch.add(torch.exp(-input*k), 1)
#     return sigmoid

def to_np(x):
    return x.data.cpu().numpy()

try:
    os.makedirs(opt.outpath)
except OSError:
    pass


Adversarial = opt.adversarial
Adversarial = True

cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")

torch.manual_seed(opt.seed)
if cuda:
    torch.cuda.manual_seed(opt.seed)

cudnn.benchmark = True
#print(torch.__version__)
print('===> Building model')
# model = './outputs_base_newnet/netG_epoch_370.pth'
# model = './outputs_joint/netG_epoch_80.pth'
# model = './outputs_adversarial_new/netG_epoch_150.pth'
# weights = torch.load(model)
netG = NetG(ngpu = opt.ngpu)
# netG.load_state_dict(weights)
# netG.apply(weights_init)
print(netG)
if Adversarial:
    netD = NetD(ngpu = opt.ngpu)
    # netD.apply(weights_init)
    print(netD)
# L1 loss
# criterion = nn.L1Loss()

if cuda:
    netG = netG.cuda()
    if Adversarial:
        netD = netD.cuda()
    # criterion = criterion.cuda()

# setup optimizer
lr = opt.lr
lr = 0.00002
decay = opt.decay
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))
if Adversarial:
    optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))

dataloader = loader(Dataset('./'),opt.batchsize)

#dataloader_val = loader(Dataset_val('./'), 5)  # MWB
dataloader_val = loader(Dataset_val('./'), opt.batchsize)


max_iou = 0
k = 1

# Set the logger
logger = Logger(opt.outpath)

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 1):
        print("MWB:dataloader===> Epoch:{}, i: {}".format(epoch, i))
        #train D
        netD.zero_grad()
        # netG.zero_grad()
        #
        input= Variable(data)[:,0:4,:,:]
        images= Variable(data)[:,3,:,:]
        label= Variable(data)[:,4,:,:]
        # label[label>0] = 1
        if cuda:
            input = input.cuda()
            target = label.cuda()
        # target = target.unsqueeze(1)
        # print(target.type)
        target = label.type(torch.LongTensor)
        # # print(target_onehot.type)
        # # label = target.data
        # # print(label.type)
        index = target.clone().unsqueeze(1).data
        target_onehot = torch.FloatTensor(index.size()[0],5,index.size()[2],index.size()[3])
        target_onehot.zero_()
        target_onehot = target_onehot.scatter_(1,index,1)
        target_onehot = Variable(target_onehot)
        if cuda:
            target_onehot = target_onehot.cuda()
        output = netG(input)
        output = gumbel_softmax_sample(output,k)
        output = output.detach()

        output_masked = torch.FloatTensor(output.size()[0],4*5,output.size()[2],output.size()[3])
        output_masked.zero_()
        output_masked = Variable(output_masked)
        if cuda:
            output_masked = output_masked.cuda()
        # input_masked = input.clone()
        # for j in range(4):
        #     output_masked[:,5*j:5*j+5,:,:] = input[:,j,:,:].unsqueeze(1).expand(output.size()[0],5,output.size()[2],output.size()[3]) * output
        target_masked = torch.FloatTensor(target_onehot.size()[0],4*5,target_onehot.size()[2],target_onehot.size()[3])
        target_masked.zero_()
        target_masked = Variable(target_masked)
        if cuda:
            target_masked = target_masked.cuda()
        for j in range(4):
            target_masked[:,5*j:5*j+5,:,:] = input[:,j,:,:].unsqueeze(1).expand(target_onehot.size()[0],5,target_onehot.size()[2],target_onehot.size()[3]) * target_onehot

        # output_cat = torch.cat([output,input],1)
        # print(output_cat.type)
        # if epoch < 3:
        #     loss_dice = dice_loss(output,target)
        # output_masked = input.clone()
        # input_mask = input.clone()
        # #detach G from the network
        # for d in range(4):
        #     output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        # if cuda:
        #     output_masked = output_masked.cuda()
        # result = netD(output_masked)
        # target_masked = input.clone()
        # for d in range(4):
        #     target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        # if cuda:
        #     target_masked = target_masked.cuda()
        # result = netD(output_cat)

        result = netD(output_masked)
        # # target_D = netD(target_cat)
        target_D = netD(target_masked)
        loss_D = - torch.mean(torch.abs(result - target_D))
        loss_D.backward()
        optimizerD.step()
        #clip parameters in D
        for p in netD.parameters():
            p.data.clamp_(-0.05, 0.05)

        #train G
        netG.zero_grad()
        output = netG(input)
        output = gumbel_softmax_sample(output,k)
        # output = F.sigmoid(output)
        # print(output.type)
        for j in range(4):
            output_masked[:,5*j:5*j+5,:,:] = input[:,j,:,:].unsqueeze(1).expand(output.size()[0],5,output.size()[2],output.size()[3]) * output

        # output_cat = torch.cat([output,input],1)
        # for d in range(4):
        #     output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        if cuda:
            output_masked = output_masked.cuda()
        result = netD(output_masked)
        # for d in range(4):
        #     target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        # if cuda:
        #     target_masked = target_masked.cuda()

        # result = netD(output)
        target_G = netD(target_masked)
        loss_dice, dice_score = dice_loss(output,target_onehot)
        loss_G = torch.mean(torch.abs(result - target_G))
        loss_G_joint = loss_G + loss_dice
        loss_G_joint.backward()
        # loss_G.backward()
        optimizerG.step()
        # netG.zero_grad()
        #
        # # if epoch < 0:
        # #     for param in netG.convblock2.parameters():
        # #         param.requires_grad = False
        # #     for param in netG.convblock3.parameters():
        # #         param.requires_grad = False
        # #     for param in netG.convblock4.parameters():
        # #         param.requires_grad = False
        # #     for param in netG.convblock5.parameters():
        # #         param.requires_grad = False
        # # else:
        # #     for param in netG.convblock2.parameters():
        # #         param.requires_grad = True
        # #     for param in netG.convblock3.parameters():
        # #         param.requires_grad = True
        # #     for param in netG.convblock4.parameters():
        # #         param.requires_grad = True
        # #     for param in netG.convblock5.parameters():
        # #         param.requires_grad = True
        # input, label = Variable(data[0]), Variable(data[1])
        # if cuda:
        #     input = input.cuda()
        #     target = label.cuda()
        # target = target.type(torch.FloatTensor)
        # target = target.cuda()
        #
        # output = netG(input)
        # # print(output.type)
        # output = F.sigmoid(output*k)
        #
        # loss_dice = dice_loss(output,target)
        # # loss_G = torch.mean(torch.abs(result - target_G))
        # # loss_G_joint = torch.mean(torch.abs(result - target_G)) + loss_dice
        # # loss_G_joint.backward()
        # loss_dice.backward()
        # optimizerG.step()
        # if Adversarial:
        #     #train D
        #     netD.zero_grad()
        #     output = netG(input)
        #     # print(output.type)
        #     output = F.sigmoid(output*k)
        #     output = output.detach()
        #     output_masked = input.clone()
        #     input_mask = input.clone()
        #     for d in range(3):
        #         output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        #     if cuda:
        #         output_masked = output_masked.cuda()
        #     result = netD(output_masked)
        #     target_masked = input.clone()
        #     for d in range(3):
        #         target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        #     if cuda:
        #         target_masked = target_masked.cuda()
        #     target_D = netD(target_masked)
        #     loss_D = - torch.mean(torch.abs(result - target_D))
        #     loss_D.backward()
        #     optimizerD.step()
        #     #clip parameters in D
        #     for p in netD.parameters():
        #         p.data.clamp_(-0.02, 0.02)
        #
        #     #train G
        #     netG.zero_grad()
        #     output = netG(input)
        #     output = F.sigmoid(output*k)
        #     for d in range(3):
        #         output_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * output
        #     if cuda:
        #         output_masked = output_masked.cuda()
        #     result = netD(output_masked)
        #     for d in range(3):
        #         target_masked[:,d,:,:] = input_mask[:,d,:,:].unsqueeze(1) * target
        #     if cuda:
        #         target_masked = target_masked.cuda()
        #     target_G = netD(target_masked)
        #     loss_G = torch.mean(torch.abs(result - target_G))
        #     loss_G.backward()
        #     optimizerG.step()
        #     loss_dice = dice_loss(output,target)
        #train D
        # netD.zero_grad()
        #loss_D = criterion(result, target_D)
        #if i % 50 == 0:
        #if i % 50 == 1:
        if i % 10 == 1:  # MWB - more frequent output
            print("MWB:mod 10 == 1 ===> Epoch:{}, i: {}".format(epoch, i))
            dice_score = to_np(dice_score)
            print("===> Epoch[{}]({}/{}): Batch Dice 1: {:.4f}".format(epoch, i, len(dataloader), 1 - dice_score[1]))
            print("===> Epoch[{}]({}/{}): Batch Dice 2: {:.4f}".format(epoch, i, len(dataloader), 1 - dice_score[2]))
            print("===> Epoch[{}]({}/{}): Batch Dice 3: {:.4f}".format(epoch, i, len(dataloader), 1 - dice_score[3]))
            print("===> Epoch[{}]({}/{}): Batch Dice 4: {:.4f}".format(epoch, i, len(dataloader), 1 - dice_score[4]))
            # if Adversarial:
            #     if epoch >= 0:
            print("===> Epoch[{}]({}/{}): G_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_G.data[0]))
            print("===> Epoch[{}]({}/{}): D_Loss: {:.4f}".format(epoch, i, len(dataloader), loss_D.data[0]))
        #============ TensorBoard logging ============#
        # (1) Log the scalar values
            _, argmax = torch.max(output, 1)
            _, target_argmax = torch.max(target_onehot, 1)
            # print(output.type)
            # argmax = output.clone()
            # argmax[argmax>0.5] = 1
            # argmax[argmax<=0.5] = 0
            # argmax = argmax.squeeze(1)
            # print(argmax.type)
            target = target.squeeze(1)
            accuracy = (target_argmax == argmax).float().mean()
            info = {
                'G loss': loss_G.data[0],
                'D loss': loss_D.data[0],
                'Batch Accuracy': accuracy.data[0],
                'Batch Dice 1': 1 - dice_score[1],
                'Batch Dice 2': 1 - dice_score[2],
                'Batch Dice 3': 1 - dice_score[3],
                'Batch Dice 4': 1 - dice_score[4]
            }
            step = epoch * len(dataloader) + i
            for tag, value in info.items():
                logger.scalar_summary(tag, value, step)

            # # (2) Log values and gradients of the parameters (histogram)
            # for tag, value in netG.named_parameters():
            #     tag = tag.replace('.', '/')
            #     logger.histo_summary(tag, to_np(value), step)
            #     # logger.histo_summary(tag+'/grad', to_np(value.grad), step)
            # (3) Log the images
            info = {
                'images': to_np(images[:6]),
                'labels': to_np(target_argmax[:6]),
                # 'labels': to_np(target[:6]),
                'results':  to_np(argmax[:6])
            }

            for tag, images in info.items():
                logger.image_summary(tag, images, step)
	#============ TensorBoard logging ============#




    # vutils.save_image(data[0],
    #         '%s/input.png' % opt.outpath,
    #         normalize=True)
    # vutils.save_image(data[1],
    #         '%s/label.png' % opt.outpath,
    #         normalize=True)
    # # result = netG(input).cuda()
    # # result = result[0].data.max(0)[1]
    # # result = Colorize()(result)
    # vutils.save_image(output.data,
    #         #'%s/result_epoch_%03d.png' % (opt.outpath, epoch),
    #         '%s/result.png' % opt.outpath,
    #         normalize=True)
    # with open('%s/result_epoch_%03d.txt' % (opt.outpath, epoch), "w") as text_file:
    #     text_file.write("Batch Dice: {:.4f}".format(1 - loss_dice.data[0]))
    #     # text_file.write("G_Loss: {:.4f}".format(loss_G.data[0]))
    #     # text_file.write("D_Loss: {:.4f}".format(loss_D.data[0]))
    if epoch % 1 == 0:
        IoUs, dices, accs = [], [], []
        for i, data in enumerate(dataloader_val, 1):
            print("MWB:dataloader_val===> Epoch:{}, i: {}".format(epoch, i))
            input= Variable(data)[:,0:4,:,:]
            images= Variable(data)[:,3,:,:]
            target= Variable(data)[:,4,:,:]
            if cuda:
                input = input.cuda()
                target = target.cuda()
            target = target.unsqueeze(1)
            target = target.type(torch.LongTensor)
            target_onehot = torch.LongTensor(target.size()[0],5,target.size()[2],target.size()[3])
            target_onehot.zero_()
            index = target.clone().data
            gt = target_onehot.scatter_(1,index,1)
            # gt = Variable(gt)
            # if cuda:
            #     gt = gt.cuda()
            pred = netG(input)
            pred = gumbel_softmax_sample(pred,k)
            # pred = pred.detach()
            # pred[pred < 0.5] = 0
            # pred[pred >= 0.5] = 1
            # pred = pred.type(torch.LongTensor)
            gt = gt.type(torch.LongTensor)
            # print(pred.sum(2))
            pred_np = pred.data.cpu().numpy()
            gt_np = gt.cpu().numpy()
            _, pred_argmax = torch.max(pred, 1)
            _, target_argmax = torch.max(gt, 1)
            accuracy = (target_argmax.cpu() == pred_argmax.data.cpu()).float().mean()
            accs.append(accuracy)
            for x in range(input.size()[0]):
                IoU = np.sum(pred_np[x][gt_np[x]==1]) / float(np.sum(pred_np[x]) + np.sum(gt_np[x]) - np.sum(pred_np[x][gt_np[x]==1]))
                dice = np.sum(pred_np[x][gt_np[x]==1])*2 / float(np.sum(pred_np[x]) + np.sum(gt_np[x]))
                IoUs.append(IoU)
                dices.append(dice)
            # dices.append(dice)
            # for gt_, pred_ in zip(gt, pred_np):
            #     gts.append(gt_)
            #     preds.append(pred_)
        IoUs = np.array(IoUs, dtype=np.float32)
        dices = np.array(dices, dtype=np.float32)
        accs = np.array(accs, dtype=np.float32)
        # print(dices.shape)
        mIoU = np.mean(IoUs, axis=0)
        mdice = np.mean(dices, axis=0)
        macc = np.mean(accs, axis=0)
        print('mean accuracy: {:.4f}'.format(macc))
        print('mIoU: {:.4f}'.format(mIoU))
        print('Dice: {:.4f}'.format(mdice))
	#============ TensorBoard logging ============#
        info = {
            'Val Acc': macc,
            'Val mIoU': mIoU,
            'Val Dice': mdice
        }
        step = epoch * len(dataloader)
        for tag, value in info.items():
            logger.scalar_summary(tag, value, step)
        _, argmax = torch.max(pred, 1)
        target = target.squeeze(1)
        # (3) Log the images
        info = {
            'Val images': to_np(images),
            'Val labels': to_np(target),
            'Val results': to_np(argmax)
        }

        for tag, images in info.items():
            logger.image_summary(tag, images, step)
	#============ TensorBoard logging ============#


        # print('I: {:.4f}'.format(np.sum(preds[gts==1])))
        # print('U: {:.4f}'.format(np.sum(preds) + np.sum(gts)))
        if mIoU > max_iou:
            max_iou = mIoU
            torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outpath, epoch))
        # with open('%s/val_epoch_%03d.png' % (opt.outpath, epoch), "w") as text_file:
        #     text_file.write('mIoU: {:.4f}'.format(mIoU))
        #     text_file.write('Dice: {:.4f}'.format(mdice))
        # vutils.save_image(data[0],
        #         '%s/input_val.png' % opt.outpath,
        #         normalize=True)
        # vutils.save_image(data[1],
        #         '%s/label_val.png' % opt.outpath,
        #         normalize=True)
        # # print(pred.type)
        # vutils.save_image(pred.data,
        #         '%s/result_val.png' % opt.outpath,
        #         normalize=True)
        # lr_D = lr_D*decay
        # score, class_iou = scores(gts, preds, n_class=2)
        # for k, v in score.items():
        #     print (k, v)
    # k = 0.5
    if lr <= 0.000002:
        lr = 0.000002 - epoch * 0.0000001
        if lr < 0.0000001:
            lr = 0.0000001
    optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    if Adversarial:
        optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    # print('Learning Rate: {:.6f}'.format(lr))
    print('Learning Rate: {:.9f}'.format(lr))
    # print('K: {:.4f}'.format(k))
    print('Max mIoU: {:.4f}'.format(max_iou))
    if epoch % 10 == 0 and epoch > 0:
        lr = lr*decay
        k = k*0.9
        if k < 0.4:
            k = 0.4
        # if lr <= 0.0005:
        #     lr = 0.0005 - epoch * 0.000001
        # if lr < 0.00001:
        #     lr = 0.00001
        # print('Learning Rate: {:.6f}'.format(lr))
        print('K: {:.4f}'.format(k))
        if lr <= 0.000002:
            lr = 0.000002 - epoch * 0.0000001
            if lr < 0.0000001:
                lr = 0.0000001
    #     print('Max mIoU: {:.4f}'.format(max_iou))
        optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))
        if Adversarial:
            optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    # if lr <= 0.0005:
    #     lr = 0.0005 - epoch * 0.000001
    #     if lr < 0.00001:
    #         lr = 0.00001
    #     optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(opt.beta1, 0.999))
    #     if Adversarial:
    #         optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(opt.beta1, 0.999))
