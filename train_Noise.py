import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torchvision
import math
import sys
import os, argparse
from datetime import datetime
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from data_Noise import get_loader
from utils import clip_gradient, adjust_lr
from mve import MobileViT32FPN,MobileViTFPN,HOR, CriterionCWD
from mobilevit import MobileViT
# from trainmve import InpaintingForensics
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2,3'

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=200, help='epoch number')
parser.add_argument('--lr', type=float, default=1e-5, help='learning rate')
parser.add_argument('--batchsize', type=int, default=48, help='training batch size')
parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
parser.add_argument('--clip', type=float, default=0.5, help='gradient clipping margin')
parser.add_argument('--margin', type=float, default=0.2, help='triple loss margin')
parser.add_argument('--decay_rate', type=float, default=0.1, help='decay rate of learning rate')
parser.add_argument('--decay_epoch', type=int, default=50, help='every n epochs decay learning rate')
opt = parser.parse_args()

print('Learning Rate: {} batchsize: {}'.format(opt.lr, opt.batchsize))
# build models
# eval_model = InpaintingForensics()

torch.manual_seed(42)
model_Noise = MobileViT32FPN(
            image_size = (256, 256),
            dims = [64, 80, 96],
            channels = [16, 16, 24, 24, 48, 48, 64, 64, 80, 80, 320],
            num_classes = 1
        )
model_RGB = MobileViT(
            image_size = (256, 256),
            dims = [144, 192, 240],
            channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
            num_classes = 1
        )

model_Noise = nn.DataParallel(model_Noise).cuda()
model_RGB = nn.DataParallel(model_RGB).cuda()

state_dict=torch.load('./weights/model_Noise.pth')
model_Noise.load_state_dict(state_dict, strict=True)
model_RGB.load_state_dict(torch.load('./RGB_weights.pth'))


params = model_Noise.parameters()
optimizer = torch.optim.Adam(params, opt.lr)

image_root = './train.npy'

train_loader = get_loader(image_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
total_step = len(train_loader)
lamda = 0.5
CE = torch.nn.BCEWithLogitsLoss()

def cross_entropy2d(input, target, temperature=1, weight=None, size_average=True):
    target = target.long()
    n, c, h, w = input.size()
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)
    # target: (n*h*w,)
    mask = target >= 0
    target = target[mask]
    T = temperature
    loss = F.cross_entropy(input / T, target, weight=weight, size_average=size_average)
    # if size_average:
    #     loss /= mask.data.sum()
    return loss

def KD_KLDivLoss(Stu_output, Tea_output, temperature):
    T = temperature
    KD_loss = nn.BCELoss()(Stu_output, Tea_output)
    KD_loss = KD_loss
    return KD_loss

def Dilation(input):
    maxpool = nn.MaxPool2d(kernel_size=11, stride=1, padding=5)
    map_b = maxpool(input)
    return map_b
def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')

    pred = torch.sigmoid(pred)
    # print(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()

def cos_simi2(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=2)
    embedded_bg = F.normalize(embedded_bg, dim=2)
    sim = torch.matmul(embedded_fg, embedded_bg.permute(0,2,1))

    return torch.clamp(sim, min=0.0005, max=0.9995)

def cos_simi(embedded_fg, embedded_bg):
    embedded_fg = F.normalize(embedded_fg, dim=1)
    embedded_bg = F.normalize(embedded_bg, dim=1)
    sim = torch.matmul(embedded_fg, embedded_bg.T)

    return torch.clamp(sim, min=0.0005, max=0.9995)
def save_img(img,name):
    import cv2
    img = img[0].detach().cpu().numpy()
    print(name)
    cv2.imwrite(name,img[0]*255)
def feat_loss(RGB_features, x5, gts,lamda=0.5,reduction = 'mean'):
    weight = gts.view(gts.size(0), -1).mean(1)
    fg_weight = (1-weight).view(gts.size(0),1).float()
    bg_weight = weight.view(gts.size(0),1).float()
    value = torch.abs(RGB_features)
    N,C,H,W = RGB_features.shape

    fea_map = value.mean(axis=1, keepdim=True)

    T_attention = (H * W * F.softmax((fea_map/lamda).view(N,-1), dim=1)).view(N, H, W)

    Mask_fg = torch.zeros_like(T_attention).float()
    Mask_bg = torch.ones_like(T_attention).float()
    labels = F.interpolate(gts, (H, W)).view(N,H,W)
    for i in range(N):
        Mask_fg[i] = torch.where(labels[i]>0, fg_weight[i], torch.FloatTensor([0])[0].cuda())
        Mask_bg[i] = torch.where(Mask_fg[i]>0, torch.FloatTensor([0])[0].cuda(), bg_weight[i])

    fea_t= RGB_features*(torch.sqrt(T_attention).unsqueeze(1))
    Mask_fg = Mask_fg.unsqueeze(1)
    Mask_bg = Mask_bg.unsqueeze(1)
    fg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_fg))
    bg_fea_t = torch.mul(fea_t, torch.sqrt(Mask_bg))
    
    fg_fea_s = torch.mul(x5, torch.sqrt(Mask_fg))
    bg_fea_s = torch.mul(x5, torch.sqrt(Mask_bg))
    
    fg_feat = cos_simi2(fg_fea_t.view(N,-1,H*W), fg_fea_s.view(N,-1,H*W)).view(N,-1)
    bg_feat = cos_simi2(bg_fea_t.view(N,-1,H*W), bg_fea_s.view(N,-1,H*W)).view(N,-1)

    loss_unsim = cos_simi(fg_feat, bg_feat)
    loss_unsim = -torch.log(1 - loss_unsim)
    if reduction == 'mean':
        loss_unsim = torch.mean(loss_unsim)
    elif reduction == 'sum':
        loss_unsim = torch.sum(loss_unsim)
    return loss_unsim, T_attention

def ATL(predict, target, reduction = 'mean'):
    n,_,_,_ = target.shape
    loss_sim=cos_simi(predict.view(n,-1), target.view(n,-1))
    loss_sim=-torch.log(loss_sim)
    if reduction == 'mean':
        return torch.mean(loss_sim)
    elif reduction == 'sum':
        return torch.sum(loss_sim)

def train(train_loader, model_Noise, model_RGB, optimizer, epoch):
    model_Noise.train()

    for i, pack in enumerate(train_loader, start=1):
        optimizer.zero_grad()

        images, gts = pack

        images = Variable(images)
        gts = Variable(gts)
        images = images.cuda()
        gts = gts.cuda()

        n, c, h, w = images.size()

        dets_RGB, RGB_features = model_RGB(images)

        dets, att_3, att_4, att_5, x5 = model_Noise(images)
        shape = images.size()
        
        # dets = dets.mul(fg_weight)
        loss_gt = nn.BCELoss()(dets.view(dets.size(0), -1), gts.view(gts.size(0), -1))
        loss_RGB = nn.BCELoss()(dets_RGB.view(dets_RGB.size(0), -1).detach(), gts.view(gts.size(0), -1))
        
        # LIPU_loss = CriterionCWD(norm_type=None,divergence='mse', temperature=1)(dets, dets_RGB.detach())
        LIPU_loss = KD_KLDivLoss(dets.view(dets.size(0), -1), dets_RGB.view(dets_RGB.size(0), -1).detach(),20)
        alpha = math.exp(-3*loss_RGB)
        loss_adptative = (1-alpha)*loss_gt + alpha*LIPU_loss
        
        loss_unsim, T_attention = feat_loss(RGB_features, x5, gts)
        T_attention = torch.tanh(T_attention.view(n, 1, 32, 32))

        att_3=(torch.sigmoid(nn.InstanceNorm2d(1)(att_3))).view(n,1,32,32)
        att_4=(torch.sigmoid(nn.InstanceNorm2d(1)(att_4))).view(n,1,32,32)
        att_5=(torch.sigmoid(nn.InstanceNorm2d(1)(att_5))).view(n,1,32,32)
        loss_attention = ATL(att_3, T_attention.detach()) + \
                         ATL(att_4, T_attention.detach()) + \
                           ATL(att_5, T_attention.detach())
        

        loss = 4*loss_adptative +loss_unsim+ loss_attention
        loss.backward()
        optimizer.step()
        clip_gradient(optimizer, opt.clip, i)

        if i % 100 == 0 or i == total_step:

            print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], Loss: {:.4f}, loss_RGB: {:.4f}, Loss_gt: {:.4f}, Loss_att: {:.4f}, Loss_LIPU: {:.4f}, Loss_unsim: {:.4f}, alpha: {:.4f}'.
                  format(datetime.now(), epoch, opt.epoch, i, total_step, loss.data, loss_RGB.data, loss_gt.data, loss_attention.data, LIPU_loss.data, loss_unsim.data,alpha))


    save_path = './weights/'

    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if (epoch+1) % 1 == 0:
        torch.save(model_Noise.state_dict(), save_path+ 'model_Noise.pth')


print("Let's go!")

for epoch in range(1, opt.epoch):
    adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
    train(train_loader, model_Noise, model_RGB, optimizer, epoch)
