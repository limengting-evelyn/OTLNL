from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import torch
import os
import json

class ImageNetPolicy(object):
    """Randomly choose one of the best 24 Sub-policies on ImageNet.

    Example:
    >>> policy = ImageNetPolicy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     ImageNetPolicy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.4, "posterize", 8, 0.6, "rotate", 9, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.6, "posterize", 7, 0.6, "posterize", 6, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.4, "equalize", 4, 0.8, "rotate", 8, fillcolor),
            SubPolicy(0.6, "solarize", 3, 0.6, "equalize", 7, fillcolor),
            SubPolicy(0.8, "posterize", 5, 1.0, "equalize", 2, fillcolor),
            SubPolicy(0.2, "rotate", 3, 0.6, "solarize", 8, fillcolor),
            SubPolicy(0.6, "equalize", 8, 0.4, "posterize", 6, fillcolor),
            SubPolicy(0.8, "rotate", 8, 0.4, "color", 0, fillcolor),
            SubPolicy(0.4, "rotate", 9, 0.6, "equalize", 2, fillcolor),
            SubPolicy(0.0, "equalize", 7, 0.8, "equalize", 8, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "rotate", 8, 1.0, "color", 2, fillcolor),
            SubPolicy(0.8, "color", 8, 0.8, "solarize", 7, fillcolor),
            SubPolicy(0.4, "sharpness", 7, 0.6, "invert", 8, fillcolor),
            SubPolicy(0.6, "shearX", 5, 1.0, "equalize", 9, fillcolor),
            SubPolicy(0.4, "color", 0, 0.6, "equalize", 3, fillcolor),
            SubPolicy(0.4, "equalize", 7, 0.2, "solarize", 4, fillcolor),
            SubPolicy(0.6, "solarize", 5, 0.6, "autocontrast", 5, fillcolor),
            SubPolicy(0.6, "invert", 4, 1.0, "equalize", 8, fillcolor),
            SubPolicy(0.6, "color", 4, 1.0, "contrast", 8, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.6, "equalize", 3, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment ImageNet Policy"

class CIFAR10Policy(object):
    """Randomly choose one of the best 25 Sub-policies on CIFAR10.

    Example:
    >>> policy = CIFAR10Policy()
    >>> transformed = policy(image)

    Example as a PyTorch Transform:
    >>> transform=transforms.Compose([
    >>>     transforms.Resize(256),
    >>>     CIFAR10Policy(),
    >>>     transforms.ToTensor()])
    """

    def __init__(self, fillcolor=(128, 128, 128)):
        self.policies = [
            SubPolicy(0.1, "invert", 7, 0.2, "contrast", 6, fillcolor),
            SubPolicy(0.7, "rotate", 2, 0.3, "translateX", 9, fillcolor),
            SubPolicy(0.8, "sharpness", 1, 0.9, "sharpness", 3, fillcolor),
            SubPolicy(0.5, "shearY", 8, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.5, "autocontrast", 8, 0.9, "equalize", 2, fillcolor),
            SubPolicy(0.2, "shearY", 7, 0.3, "posterize", 7, fillcolor),
            SubPolicy(0.4, "color", 3, 0.6, "brightness", 7, fillcolor),
            SubPolicy(0.3, "sharpness", 9, 0.7, "brightness", 9, fillcolor),
            SubPolicy(0.6, "equalize", 5, 0.5, "equalize", 1, fillcolor),
            SubPolicy(0.6, "contrast", 7, 0.6, "sharpness", 5, fillcolor),
            SubPolicy(0.7, "color", 7, 0.5, "translateX", 8, fillcolor),
            SubPolicy(0.3, "equalize", 7, 0.4, "autocontrast", 8, fillcolor),
            SubPolicy(0.4, "translateY", 3, 0.2, "sharpness", 6, fillcolor),
            SubPolicy(0.9, "brightness", 6, 0.2, "color", 8, fillcolor),
            SubPolicy(0.5, "solarize", 2, 0.0, "invert", 3, fillcolor),
            SubPolicy(0.2, "equalize", 0, 0.6, "autocontrast", 0, fillcolor),
            SubPolicy(0.2, "equalize", 8, 0.6, "equalize", 4, fillcolor),
            SubPolicy(0.9, "color", 9, 0.6, "equalize", 6, fillcolor),
            SubPolicy(0.8, "autocontrast", 4, 0.2, "solarize", 8, fillcolor),
            SubPolicy(0.1, "brightness", 3, 0.7, "color", 0, fillcolor),
            SubPolicy(0.4, "solarize", 5, 0.9, "autocontrast", 3, fillcolor),
            SubPolicy(0.9, "translateY", 9, 0.7, "translateY", 9, fillcolor),
            SubPolicy(0.9, "autocontrast", 2, 0.8, "solarize", 3, fillcolor),
            SubPolicy(0.8, "equalize", 8, 0.1, "invert", 3, fillcolor),
            SubPolicy(0.7, "translateY", 9, 0.9, "autocontrast", 1, fillcolor),
        ]

    def __call__(self, img):
        policy_idx = random.randint(0, len(self.policies) - 1)
        return self.policies[policy_idx](img)

    def __repr__(self):
        return "AutoAugment CIFAR10 Policy"
class SubPolicy(object):
    def __init__(
        self,
        p1,
        operation1,
        magnitude_idx1,
        p2,
        operation2,
        magnitude_idx2,
        fillcolor=(128, 128, 128),
    ):
        ranges = {
            "shearX": np.linspace(0, 0.3, 10),
            "shearY": np.linspace(0, 0.3, 10),
            "translateX": np.linspace(0, 150 / 331, 10),
            "translateY": np.linspace(0, 150 / 331, 10),
            "rotate": np.linspace(0, 30, 10),
            "color": np.linspace(0.0, 0.9, 10),
            "posterize": np.round(np.linspace(8, 4, 10), 0).astype(int),
            "solarize": np.linspace(256, 0, 10),
            "contrast": np.linspace(0.0, 0.9, 10),
            "sharpness": np.linspace(0.0, 0.9, 10),
            "brightness": np.linspace(0.0, 0.9, 10),
            "autocontrast": [0] * 10,
            "equalize": [0] * 10,
            "invert": [0] * 10,
        }

        # from https://stackoverflow.com/questions/5252170/specify-image-filling-color-when-rotating-in-python-with-pil-and-setting-expand
        def rotate_with_fill(img, magnitude):
            rot = img.convert("RGBA").rotate(magnitude)
            return Image.composite(
                rot, Image.new("RGBA", rot.size, (128,) * 4), rot
            ).convert(img.mode)

        func = {
            "shearX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, magnitude * random.choice([-1, 1]), 0, 0, 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "shearY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, magnitude * random.choice([-1, 1]), 1, 0),
                Image.BICUBIC,
                fillcolor=fillcolor,
            ),
            "translateX": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, magnitude * img.size[0] * random.choice([-1, 1]), 0, 1, 0),
                fillcolor=fillcolor,
            ),
            "translateY": lambda img, magnitude: img.transform(
                img.size,
                Image.AFFINE,
                (1, 0, 0, 0, 1, magnitude * img.size[1] * random.choice([-1, 1])),
                fillcolor=fillcolor,
            ),
            "rotate": lambda img, magnitude: rotate_with_fill(img, magnitude),
            "color": lambda img, magnitude: ImageEnhance.Color(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "posterize": lambda img, magnitude: ImageOps.posterize(img, magnitude),
            "solarize": lambda img, magnitude: ImageOps.solarize(img, magnitude),
            "contrast": lambda img, magnitude: ImageEnhance.Contrast(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "sharpness": lambda img, magnitude: ImageEnhance.Sharpness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "brightness": lambda img, magnitude: ImageEnhance.Brightness(img).enhance(
                1 + magnitude * random.choice([-1, 1])
            ),
            "autocontrast": lambda img, magnitude: ImageOps.autocontrast(img),
            "equalize": lambda img, magnitude: ImageOps.equalize(img),
            "invert": lambda img, magnitude: ImageOps.invert(img),
        }

        self.p1 = p1
        self.operation1 = func[operation1]
        self.magnitude1 = ranges[operation1][magnitude_idx1]
        self.p2 = p2
        self.operation2 = func[operation2]
        self.magnitude2 = ranges[operation2][magnitude_idx2]

    def __call__(self, img):
        if random.random() < self.p1:
            img = self.operation1(img, self.magnitude1)
        if random.random() < self.p2:
            img = self.operation2(img, self.magnitude2)
        return img


transform_weak_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)


def transform_weak_c1m(x):
    return transform_weak_c1m_c10_compose(x)


transform_strong_c1m_c10_compose = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        CIFAR10Policy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)


def transform_strong_c1m_c10(x):
    return transform_strong_c1m_c10_compose(x)


transform_strong_c1m_in_compose = transforms.Compose(
    [
        transforms.Resize(320),
        transforms.RandomCrop(299),
        transforms.RandomHorizontalFlip(),
        ImageNetPolicy(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
    ]
)
def transform_strong_c1m_in(x):
    return transform_strong_c1m_in_compose(x)

class imagenet_dataset(Dataset):
    def __init__(self, root_dir, transform, num_class):
        self.root = root_dir+'imagenet/val/'
        self.transform = transform
        self.val_data = []
        with open(os.path.join(root_dir, 'info/synsets.txt')) as f:
            lines = f.readlines()
        synsets = [x.split()[0] for x in lines]
        for c in range(num_class):
            class_path = os.path.join(self.root, synsets[c])
            imgs = os.listdir(class_path)
            for img in imgs:
                self.val_data.append([c, os.path.join(class_path, img)]) 
        """
        sysnet = os.listdir(self.root)
        for c in range(num_class):
            imgs = os.listdir(self.root+sysnet[c])
            for img in imgs:
                self.val_data.append([c,os.path.join(self.root,sysnet[c],img)])
        """
    def __getitem__(self, index):
        data = self.val_data[index]
        target = data[0]
        image = Image.open(data[1]).convert('RGB')
        img = self.transform(image)
        return img, target

    def __len__(self):
        return len(self.val_data)

class webvision_dataset(Dataset):
    def __init__(self, root_dir, transform, mode, num_class, pred=[], probability=[], log='', refine_labels=None, imb_type='exp', imb_factor=1):
        self.root = root_dir+'Mini-WebVision-master/'
        self.transform = transform
        self.mode = mode
        self.real_img_num_list = [0] * num_class

        if self.mode=='test':
            with open(os.path.join(root_dir, 'info/synsets.txt')) as f:
                lines = f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            synsets = [x.split()[0] for x in lines]
            for c in range(num_class):
                class_path = os.path.join(self.root, 'val', synsets[c])
                imgs = os.listdir(class_path)
                for img in imgs:
                    img = os.path.join(class_path, img)
                    self.val_imgs.append(img)
                    self.val_labels[img]=c
                    #self.test_data.append([c, os.path.join(class_path, img)])
            """
            with open(self.root+'info/val_filelist.txt') as f:
                lines=f.readlines()
            self.val_imgs = []
            self.val_labels = {}
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    self.val_imgs.append(self.root+img)
                    self.val_labels[img]=target
            """
        else:
            with open(os.path.join(root_dir, 'info/synsets.txt')) as f:
                lines = f.readlines()
            train_imgs = []
            self.train_labels = {}
            synsets = [x.split()[0] for x in lines]
            i = 0
            for c in range(num_class):
                if refine_labels is not None:
                        target = refine_labels[i]
                        i += 1
                else:
                    target = c
                class_path = os.path.join(self.root, 'train', synsets[c])
                imgs = os.listdir(class_path)
                for img in imgs:
                    img = os.path.join(class_path, img)
                    train_imgs.append(img)
                    self.train_labels[img]=target
            """
            with open(self.root+'info/train_filelist_google.txt') as f:
                lines=f.readlines()
            train_imgs = []
            self.train_labels = {}
            i = 0
            for line in lines:
                img, target = line.split()
                target = int(target)
                if target<num_class:
                    if refine_labels is not None:
                        target = refine_labels[i]
                        i += 1
                    train_imgs.append(img)
                    self.train_labels[img]=target
            """

            self.cls_num = num_class

            self.train_data = np.array(train_imgs)
            img_num_list = self.get_img_num_per_cls(self.cls_num, imb_type, imb_factor)
            self.img_num_list = img_num_list
            #print (img_num_list)
            # print (max(img_num_list), min(img_num_list), max(img_num_list) / min(img_num_list))
            # print (sum(img_num_list))
            
            if imb_factor < 0.5:
                imb_file = os.path.join('.', 'webvision_' + str(imb_factor))
                self.gen_imbalanced_data(img_num_list, imb_file)
                train_imgs = self.new_train_data

            if self.mode == 'all':
                self.train_imgs = train_imgs
                self.tmp_labels = torch.zeros(len(train_imgs))
                for idx, i in enumerate(self.train_imgs):
                    self.tmp_labels[idx] = self.train_labels[i]
                    self.real_img_num_list[self.train_labels[i]] += 1

                self.idx_class = []
                for i in range(num_class):
                    self.idx_class.append((self.tmp_labels == i).nonzero(as_tuple=True)[0])
            else:
                if self.mode == "labeled":
                    #print (len(pred))
                    #print (len(train_imgs))
                    pred_idx = pred.nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    self.probability = [probability[i] for i in pred_idx]
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))
                    log.write('Numer of labeled samples:%d \n'%(pred.sum()))
                    log.flush()
                elif self.mode == "unlabeled":
                    pred_idx = (1-pred).nonzero()[0]
                    self.train_imgs = [train_imgs[i] for i in pred_idx]
                    print("%s data has a size of %d"%(self.mode,len(self.train_imgs)))

    def __getitem__(self, index):
        if self.mode=='labeled':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            prob = self.probability[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2, target, prob
        elif self.mode=='unlabeled':
            img_path = self.train_imgs[index]
            image = Image.open(img_path).convert('RGB')
            img1 = self.transform[0](image)
            img2 = self.transform[1](image)
            if self.transform[2] == None:
                img3 = img1
                img4 = img2
            else:
                img3 = self.transform[2](image)
                img4 = self.transform[3](image)
            return img1, img2
        elif self.mode=='all':
            img_path = self.train_imgs[index]
            target = self.train_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target, index
        elif self.mode=='test':
            img_path = self.val_imgs[index]
            target = self.val_labels[img_path]
            image = Image.open(img_path).convert('RGB')
            img = self.transform(image)
            return img, target

    def __len__(self):
        if self.mode!='test':
            return len(self.train_imgs)
        else:
            return len(self.val_imgs)


    def get_img_num_per_cls(self, cls_num, imb_type, imb_factor):

        train_labels = np.array(list(self.train_labels.values()))
        raw_cls_num_list = np.array([sum(train_labels == i) for i in range(cls_num)])
        raw_cls_num_sort = raw_cls_num_list.argsort()[::-1]

        img_max = max(raw_cls_num_list)
        img_num_per_cls = []
        if imb_type == 'exp':
            for cls_idx in range(cls_num):
                num = img_max * (imb_factor**(cls_idx / (cls_num - 1.0)))
                img_num_per_cls.append(int(num))
        elif imb_type == 'step':
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max))
            for cls_idx in range(cls_num // 2):
                img_num_per_cls.append(int(img_max * imb_factor))
        else:
            img_num_per_cls.extend([int(img_max)] * cls_num)

        new_img_num_per_cls = [0 for _ in range(cls_num)]
        for i in range(cls_num):
            j = raw_cls_num_sort[i]
            new_img_num_per_cls[j] = min(img_num_per_cls[i], raw_cls_num_list[j])

        return new_img_num_per_cls

    def gen_imbalanced_data(self, img_num_per_cls, imb_file=None):
        if os.path.exists(imb_file):
            new_data = json.load(open(imb_file,"r"))
        else:
            new_data = []

            cls_idx = [[] for _ in range(50)]
            for i, img in enumerate(self.train_data):
                target = self.train_labels[img]
                cls_idx[target].append(i)

            classes = np.array(range(50))
            self.num_per_cls_dict = dict()
            for the_class, the_img_num in zip(classes, img_num_per_cls):
                self.num_per_cls_dict[the_class] = the_img_num
                idx = cls_idx[the_class]
                np.random.shuffle(idx)
                selec_idx = idx[:the_img_num]
                new_data.extend(self.train_data[selec_idx, ...])
            print ('saving imbalance data to %s ...' % imb_file)
            json.dump(new_data, open(imb_file, 'w'))

        self.new_train_data = new_data


class webvision_dataloader():
    def __init__(self, batch_size, num_class, num_workers, root_dir, log, imb_ratio=1):

        self.batch_size = batch_size
        self.num_class = num_class
        self.num_workers = num_workers
        self.root_dir = root_dir
        self.log = log
        self.imb_factor = imb_ratio

        self.transform_train = transforms.Compose([
                transforms.Resize(320),
                transforms.RandomResizedCrop(299),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.transform_test = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.transform_imagenet = transforms.Compose([
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ])
        self.transforms = {
            "warmup": transform_weak_c1m,
            "unlabeled": [
                        transform_weak_c1m,
                        transform_weak_c1m,
                        transform_strong_c1m_in,
                        transform_strong_c1m_in
                        
                    ],
            "labeled": [
                        transform_weak_c1m,
                        transform_weak_c1m,
                        transform_strong_c1m_in,
                        transform_strong_c1m_in
                        
                    ],
            "test": None,
        }
        self.transforms["test"] = transforms.Compose(
            [
                transforms.Resize(320),
                transforms.CenterCrop(299),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225)),
            ]
        )

    def run(self,mode,pred=[],prob=[],refine_labels=None, imb_factor=1):
        if mode=='warmup':
            all_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transforms['warmup'], mode="all", num_class=self.num_class, imb_factor=self.imb_factor)
            trainloader = DataLoader(
                dataset=all_dataset,
                batch_size=self.batch_size*2,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return trainloader

        elif mode=='train':
            labeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transforms['labeled'], mode="labeled",num_class=self.num_class,pred=pred,probability=prob,log=self.log, refine_labels=refine_labels, imb_factor=self.imb_factor)
            labeled_trainloader = DataLoader(
                dataset=labeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)

            unlabeled_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transforms['unlabeled'], mode="unlabeled",num_class=self.num_class,pred=pred,log=self.log, imb_factor=self.imb_factor)
            unlabeled_trainloader = DataLoader(
                dataset=unlabeled_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=self.num_workers,
                pin_memory=True)
            return labeled_trainloader, unlabeled_trainloader

        elif mode=='test':
            test_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transforms['test'], mode='test', num_class=self.num_class)
            test_loader = DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return test_loader

        elif mode=='eval_train':
            eval_dataset = webvision_dataset(root_dir=self.root_dir, transform=self.transforms['test'], mode='all', num_class=self.num_class, imb_factor=self.imb_factor)
            eval_loader = DataLoader(
                dataset=eval_dataset,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return eval_loader

        elif mode=='imagenet':
            imagenet_val = imagenet_dataset(root_dir=self.root_dir, transform=self.transform_imagenet, num_class=self.num_class)
            imagenet_loader = DataLoader(
                dataset=imagenet_val,
                batch_size=self.batch_size*20,
                shuffle=False,
                num_workers=self.num_workers,
                pin_memory=True)
            return imagenet_loader
