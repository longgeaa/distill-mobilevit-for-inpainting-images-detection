import os
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms
import torch
import numpy as np
class SalObjDataset(data.Dataset):
    def __init__(self, image_root, trainsize):
        self.trainsize = trainsize
        self.filelist = np.load(image_root)
        self.image = []
        self.gts = []
        for i,j in self.filelist:
            self.image.append(i)
            self.gts.append(j)

        self.image = sorted(self.image)
        self.gts = sorted(self.gts)
        self.filter_files()
        self.size = len(self.image)
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        # self.img_transform = transforms.Compose([
        #         transforms.RandomRotation(90, resample=False, expand=False, center=None, fill=None),
        #         transforms.RandomVerticalFlip(p=0.5),
        #         transforms.RandomHorizontalFlip(p=0.5),
        #         transforms.Resize((self.trainsize, self.trainsize)),
        #         transforms.ToTensor(),
        #         transforms.Normalize([0.485, 0.456, 0.406],
        #                              [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

    def __getitem__(self, index):
        image = self.rgb_loader(self.image[index])
        gt = self.binary_loader(self.gts[index])
        image = self.img_transform(image)
        gt = self.gt_transform(gt)
        return image, gt

    def filter_files(self):
        assert len(self.image) == len(self.gts)

        image = []
        gts = []
        for image_path, gt_path in zip(self.image, self.gts):
            img = Image.open(image_path)
            gt = Image.open(gt_path)
            if img.size == gt.size:
                image.append(image_path)
                gts.append(gt_path)

        self.image = image
        self.gts = gts

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            # return img.convert('1')
            return img.convert('L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


def get_loader(image_root, batchsize, trainsize, shuffle=True, pin_memory=True):

    dataset = SalObjDataset(image_root, trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  pin_memory=pin_memory)
    return data_loader


class test_dataset:
    def __init__(self, image_root, gt_root, testsize):
        self.testsize = testsize
        self.images = [image_root + f for f in os.listdir(image_root) if f.endswith('.jpg') or f.endswith('.png')]
        self.gts = [gt_root + f for f in os.listdir(gt_root) if f.endswith('.jpg')
                       or f.endswith('.png')]
        self.images = sorted(self.images)
        #　self.depth = sorted(self.depth)
        self.gts = sorted(self.gts)
        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.ToTensor()
        self.size = len(self.images)
        self.index = 0

    def load_data(self):
        image = self.rgb_loader(self.images[self.index])
        image = self.transform(image).unsqueeze(0)
        gt = self.binary_loader(self.gts[self.index])
        name = self.images[self.index].split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        self.index += 1
        return image, gt, name

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')


