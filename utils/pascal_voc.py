r""" PASCAL-5i few-shot semantic segmentation dataset """
import os

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from torchvision import transforms
from os.path import join


class DatasetPASCAL(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, use_original_imgsize):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 20
        self.benchmark = 'pascal'
        self.shot = shot
        self.base_path = join(datapath, 'VOC2012')

        self.img_path = join(self.base_path, 'JPEGImages')
        self.ann_path = join(self.base_path, 'SegmentationClassAug')
        self.transform = transform

        self.class_ids = self.build_class_ids()
        self.img_metadata = self.build_img_metadata()
        self.img_metadata_classwise = self.build_img_metadata_classwise()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 1000
        # return 6000

    def __getitem__(self, idx):
        idx %= len(self.img_metadata)  # for testing, as n_images < 1000
        query_name, support_names, class_sample = self.sample_episode(idx)
        query_img, query_cmask, support_imgs, support_cmasks, org_qry_imsize = self.load_frame(query_name, support_names)

        query_mask, query_ignore_idx = self.extract_ignore_idx(query_cmask.float(), class_sample)
        # support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        support_masks = []
        # support_ignore_idxs = []
        for i, scmask in enumerate(support_cmasks):
            #scmask = F.interpolate(scmask.unsqueeze(0).unsqueeze(0).float(), support_imgs[i].size[::-1], mode='nearest').squeeze()
            support_mask, support_ignore_idx = self.extract_ignore_idx(scmask, class_sample)
            support_masks.append(support_mask)
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        assert len(support_imgs) == 1 and len(support_cmasks) == 1
        support_imgs, support_masks = support_imgs[0], support_cmasks[0]
        query_img, support_imgs = [to_tensor(x) for x in [query_img, support_imgs]]
        query_mask, support_masks = query_cmask[None].float(), support_masks[None].float()
        batch = {'image': query_img,
                 'label': query_mask*255,
                 #'org_query_imsize': org_qry_imsize,
                 'image_dual': support_imgs,
                 'label_dual': support_masks*255,
                 'is_inst': False,
                 'class_name': '',
                "shape": torch.tensor(query_img.shape[-2:]),
                "imidx": torch.from_numpy(np.array(idx)),
                 'class_id': torch.tensor(class_sample)
                 }
        if self.transform:
            batch = self.transform(batch)
        if self.split in ['val', 'test']:
            batch.update({
                'ori_label':query_mask * 255,
            })

        return batch

    def extract_ignore_idx(self, mask, class_id):
        # print('----------------',class_id)
        boundary = (mask / 255).floor()
        mask[mask != class_id + 1] = 0
        mask[mask == class_id + 1] = 1

        return mask, boundary

    def load_frame(self, query_name, support_names):
        query_img = self.read_img(query_name)
        query_mask = self.read_mask(query_name)
        support_imgs = [self.read_img(name) for name in support_names]
        support_masks = [self.read_mask(name) for name in support_names]

        org_qry_imsize = query_img.size

        return query_img, query_mask, support_imgs, support_masks, org_qry_imsize

    def read_mask(self, img_name):
        r"""Return segmentation mask in PIL Image"""
        mask = torch.tensor(np.array(Image.open(join(self.ann_path, img_name) + '.png')))
        return mask

    def read_img(self, img_name):
        r"""Return RGB image in PIL Image"""
        return Image.open(join(self.img_path, img_name) + '.jpg')

    def sample_episode(self, idx):
        query_name, class_sample = self.img_metadata[idx]

        support_names = []
        while True:  # keep sampling support set if query == support
            support_name = np.random.choice(self.img_metadata_classwise[class_sample], 1, replace=False)[0]
            if query_name != support_name: support_names.append(support_name)
            if len(support_names) == self.shot: break

        return query_name, support_names, class_sample

    def build_class_ids(self):
        nclass_trn = self.nclass // self.nfolds
        if self.fold != 4:
            class_ids_val = [self.fold * nclass_trn + i for i in range(nclass_trn)]
            class_ids_trn = [x for x in range(self.nclass) if x not in class_ids_val]
        else:
            class_ids_val = [0 + self.nfolds * v for v in range(nclass_trn)]
            class_ids_trn = list(range(self.nclass))

        if self.split == 'trn':
            return class_ids_trn
        else:
            return class_ids_val

    def build_img_metadata(self):

        def read_metadata(split, fold_id):
            fold_n_metadata = join(self.base_path, f'splits/{split}/fold{fold_id}.txt')
            with open(fold_n_metadata, 'r') as f:
                fold_n_metadata = f.read().split('\n')[:-1]
            fold_n_metadata = [[data.split('__')[0], int(data.split('__')[1]) - 1] for data in fold_n_metadata]
            return fold_n_metadata

        img_metadata = []
        if self.split == 'trn':  # For training, read image-metadata of "the other" folds
            for fold_id in range(self.nfolds):
                if fold_id == self.fold:  # Skip validation fold
                    continue
                img_metadata += read_metadata(self.split, fold_id)

        elif self.split == 'val':  # For validation, read image-metadata of "current" fold
            if self.fold != 4:
                img_metadata = read_metadata(self.split, self.fold)
            else:
                img_metadata = read_metadata(self.split, 0)

        else:
            raise Exception('Undefined split %s: ' % self.split)

        print('Total (%s) images are : %d' % (self.split, len(img_metadata)))

        return img_metadata

    def build_img_metadata_classwise(self):
        img_metadata_classwise = {}
        for class_id in range(self.nclass):
            img_metadata_classwise[class_id] = []

        for img_name, img_class in self.img_metadata:
            img_metadata_classwise[img_class] += [img_name]
        return img_metadata_classwise
    

def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetPASCAL(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
