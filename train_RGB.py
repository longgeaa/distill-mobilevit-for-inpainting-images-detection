import os
import cv2
import random
import argparse
import numpy as np
import copy
# import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import transforms
from sklearn.metrics import accuracy_score, roc_auc_score

from mobilevit import MobileViT
os.environ['CUDA_VISIBLE_DEVICES'] = '1,2'


class RGB_Dataset(Dataset):
    def __init__(self, num, file, choice='train'):
        self.num = num
        self.choice = choice
        self.hight = 256
        if self.choice != 'test':
            try:
                self.filelist = np.load(file)
            except Exception:
                self.filelist = sorted(os.listdir('demo_input/'))
        else:
            try:
                self.filelist = np.load(file)
            except Exception:
                self.filelist = sorted(os.listdir('demo_input/'))

        self.transform = transforms.Compose([
            
            np.float32,
            
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    def __getitem__(self, idx):
        return self.load_item(idx)

    def __len__(self):
        return self.num

    def load_item(self, idx):
        if self.choice != 'test':
            fname1, fname2 = self.filelist[idx]
        else:
            try:
                fname1, fname2 = self.filelist[idx]
                
            except Exception:
                fname1, fname2 = 'demo_input/' + self.filelist[idx], ''

        img = cv2.imread(fname1)
        
        # img = cv2.resize(img,(self.hight ,self.hight ))
        H, W, _ = img.shape
        if fname2 == '':
            mask = np.zeros([H, W, 3])
            mask[np.random.randint(5, H-5), np.random.randint(5, W-5), 0] = 255
        else:
            mask = cv2.imread(fname2)
        # mask = cv2.resize(mask,(self.hight ,self.hight ))
        if self.choice == 'train':
            if random.random() < 0.5:
                img = cv2.flip(img, 0)
                mask = cv2.flip(mask, 0)
            if random.random() < 0.5:
                img = cv2.flip(img, 1)
                mask = cv2.flip(mask, 1)

        img = img.astype('float') / 255.
        mask = mask.astype('float') / 255.
        return self.transform(img), self.tensor(mask[:, :, :1]), fname1.split('/')[-1]

    def tensor(self, img):
        return torch.from_numpy(img).float().permute(2, 0, 1)


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean', ):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.crit = nn.BCELoss(reduction='none')

    def forward(self, logits, label):
        # compute loss
        logits = logits.float()  # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label)
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss


def dice_bce_loss(pred, mask):
    bce = F.binary_cross_entropy_with_logits(pred, mask, reduction='none')
    
    pred = torch.sigmoid(pred)
    # print(pred)
    inter = pred * mask
    union = pred + mask
    iou = 1 - (2. * inter + 1) / (union + 1)

    return (bce + iou).mean()
    
class RGB_Model(nn.Module):
    def __init__(self):
        super(RGB_Model, self).__init__()
        self.lr = 1e-4
        self.networks = MobileViT(
            image_size = (256, 256),
            dims = [144, 192, 240],
            channels = [16, 32, 64, 64, 96, 96, 128, 128, 160, 160, 640],
            num_classes = 1
        )

        pytorch_total_params = sum(p.numel() for p in self.networks.parameters() if p.requires_grad)
        print('Total Params: %d' % pytorch_total_params)
        with open('log.txt', 'a+') as f:
            f.write('\n\nRGB-Net, Total Params: %d' % pytorch_total_params)
        self.gen = nn.DataParallel(self.networks).cuda()
        # self.gen = self.networks.cuda()
        self.gen_optimizer = optim.Adam(self.gen.parameters(), lr=self.lr, betas=(0.9, 0.999))
        self.save_dir = 'weights/'
        alpha = 0.25

    def process(self, Ii, Mg):
        self.gen_optimizer.zero_grad()

        Mo ,_= self(Ii)
        
        gen_loss = dice_bce_loss(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1).float())
        gen_loss += nn.BCELoss()(Mo.view(Mo.size(0), -1), Mg.view(Mg.size(0), -1))

        return Mo, gen_loss

    def forward(self, Ii):
        return self.gen(Ii)

    def backward(self, gen_loss=None):
        if gen_loss:
            gen_loss.backward(retain_graph=False)
            self.gen_optimizer.step()

    def save(self, path=''):
        if not os.path.exists(self.save_dir + path):
            os.makedirs(self.save_dir + path)
        torch.save(self.gen.state_dict(), self.save_dir + path + 'RGB_weights.pth')

    def load(self, path=''):
        state_dict=torch.load(path)

        self.gen.load_state_dict(state_dict,strict=False)
        
    def load_from(self, path):
        pretrained_path = path
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model"  not in pretrained_dict:
                print("---start load pretrained modle by splitting---")
                pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.networks.load_state_dict(pretrained_dict,strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            model_dict = self.networks.state_dict()
            full_dict = copy.deepcopy(pretrained_dict)
            for k, v in pretrained_dict.items():
                if "layers." in k:
                    current_layer_num = 3-int(k[7:8])
                    current_k = "layers_up." + str(current_layer_num) + k[8:]
                    full_dict.update({current_k:v})
            for k in list(full_dict.keys()):
                if k in model_dict:
                    if full_dict[k].shape != model_dict[k].shape:
                        print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                        del full_dict[k]

            msg = self.networks.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")


class InpaintingForensics():
    def __init__(self):
        self.train_num = 48000
        self.val_num = 1000
        self.test_num = 6000
        self.batch_size = 96
        # For training, please provide the absolute path of training data that saved in numpy with following format
        # E.g., file = [['./training_input_1.png', './training_ground_truth_1.png'],
        #              ['./training_input_2.png', './training_ground_truth_2.png'],...]
        self.train_file = '/home/sun/inpainting/InpaintingForensics/train.npy'
        self.val_file = '/home/sun/inpainting/InpaintingForensics/val.npy'
        self.test_file = '/home/sun/inpainting/InpaintingForensics/test.npy'
        train_dataset = RGB_Dataset(self.train_num, self.train_file, choice='train')
        val_dataset = RGB_Dataset(self.val_num, self.val_file, choice='val')
        test_dataset = RGB_Dataset(self.test_num, self.test_file, choice='test')

        self.grgb_model = RGB_Model().cuda()
        self.n_epochs = 1000
        self.train_loader = DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4)
        self.val_loader = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)
        self.test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, num_workers=4)

    def train(self):
        with open('log.txt', 'a+') as f:
            f.write('\nTrain %s with %d' % (self.train_file, self.train_num))
            f.write('\nVal %s with %d' % (self.val_file, self.val_num))
            f.write('\nTest %s with %d' % (self.test_file, self.test_num))
        scheduler_gan = ReduceLROnPlateau(self.grgb_model.gen_optimizer, patience=10, factor=0.5)
        best_auc = 0
        for epoch in range(self.n_epochs):
            
            cnt, gen_losses, auc = 0, [], []
            for items in self.train_loader:
                cnt += self.batch_size
                self.grgb_model.train()
                Ii, Mg = (item.cuda() for item in items[:-1])
                Mo, gen_loss = self.grgb_model.process(Ii, Mg)
                self.grgb_model.backward(gen_loss)
                gen_losses.append(gen_loss.item())
                Mg, Mo = self.convert2(Mg), self.convert2(Mo)
                N, H, W, C = Mg.shape
                auc.append(roc_auc_score(Mg.reshape(N * H * W * C).astype('int'), Mo.reshape(N * H * W * C)) * 100.)
                print('Tra (%d/%d): G:%6.3f A:%3.2f' % (cnt, self.train_num, np.mean(gen_losses), np.mean(auc)), end='\r')
                if cnt % self.train_num == 0 or cnt >= self.train_num:
                    val_gen_loss, val_auc = self.val()
                    scheduler_gan.step(val_auc)
                    print('Val (%d/%d): G:%6.3f A:%3.2f' % (cnt, self.train_num, val_gen_loss, val_auc))
                    if val_auc > best_auc:
                        best_auc = val_auc
                        self.grgb_model.save('mvbest/')
                    self.grgb_model.save('mvlatest/')

                    with open('log.txt', 'a+') as f:
                        f.write('\n(%d/%d): Tra: A:%4.2f Val: A:%4.2f' % (cnt, self.train_num, np.mean(auc), val_auc))
                    auc, gen_losses = [], []

    def val(self):
        self.grgb_model.eval()
        auc, gen_losses = [], []
        for cnt, items in enumerate(self.val_loader):
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1][0]
            Mo, gen_loss = self.grgb_model.process(Ii, Mg)
            gen_losses.append(gen_loss.item())
            Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg)[0], self.convert2(Mo)[0]
            H, W, _ = Mg.shape
            auc.append(roc_auc_score(Mg.reshape(H * W).astype('int'), Mo.reshape(H * W)) * 100.)

            # Sample 100 validation images for visualization
            if len(auc) <= 100:
                Mg, Mo = Mg * 255, Mo * 255
                out = np.zeros([H, H * 3, 3])
                out[:, :H, :] = Ii
                out[:, H:H*2, :] = np.concatenate([Mo, Mo, Mo], axis=2)
                out[:, H*2:, :] = np.concatenate([Mg, Mg, Mg], axis=2)
                cv2.imwrite('demo_val/' + filename, out)
        return np.mean(gen_losses), np.mean(auc)

    def test(self):
        self.grgb_model.load('./weights/model_Noise.pth')
        self.grgb_model.eval()
        auc = []
        for cnt, items in enumerate(self.test_loader):
            
            print(cnt, end='\r')
            Ii, Mg = (item.cuda() for item in items[:-1])
            filename = items[-1][0]
            Mo, gen_loss= self.grgb_model.process(Ii, Mg)
            # print(gen_loss)

            Ii, Mg, Mo = self.convert1(Ii), self.convert2(Mg)[0], self.convert2(Mo)[0]
            H, W, _ = Mg.shape
            auc.append(roc_auc_score(Mg.reshape(H * W).astype('int'), Mo.reshape(H * W)) * 100.)
            cv2.imwrite('./demo_output/' + filename, Mo * 255)

        print(np.mean(auc))

    def convert1(self, img):
        img = img * 127.5 + 127.5
        img = img.permute(0, 2, 3, 1)[0].cpu().detach().numpy()
        return img

    def convert2(self, x):
        return x.permute(0, 2, 3, 1).cpu().detach().numpy()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('type', type=str, help='train or test the model', choices=['train', 'test'])
    args = parser.parse_args()

    model = InpaintingForensics()
    if args.type == 'train':
        model.train()
    elif args.type == 'test':
        model.test()
