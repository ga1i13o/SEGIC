import os
from os.path import join
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
import json
import cv2
import pycocotools.mask as mask_util
from torchvision import transforms


class DatasetPASCALPart(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, box_crop=True, use_original_imgsize=False):
        self.split = 'val' if split in ['val', 'test'] else 'train'
        self.cat = ['animals', 'indoor', 'person', 'vehicles'][fold]
        print(f"{fold}: {self.cat}")
        # self.nfolds = 4
        self.benchmark = 'pascal_part'
        self.shot = shot
        self.transform = transform
        self.box_crop = box_crop

        self.json_file = os.path.join(datapath, 'Pascal-Part/VOCdevkit/VOC2010/all_obj_part_to_image.json')
        self.img_file = os.path.join(datapath, 'Pascal-Part/VOCdevkit/VOC2010/JPEGImages/{}.jpg')
        self.anno_file = os.path.join(datapath, 'Pascal-Part/VOCdevkit/VOC2010/Annotations_Part_json_merged_part_classes/{}.json')
        js = json.load(open(self.json_file, 'r'))

        self.cat_annos = js[self.cat]

        cat_part_name = []
        cat_part_id = []

        new_id = 0
        for i, obj in enumerate(self.cat_annos['object']):
            for part in self.cat_annos['object'][obj]['part']:
                if len(self.cat_annos['object'][obj]['part'][part]['train']) > 0 and \
                        len(self.cat_annos['object'][obj]['part'][part]['val']) > 0:
                    if obj + '+' + part == 'aeroplane+TAIL':
                        continue
                    cat_part_name.append(obj + '+' + part)
                    cat_part_id.append(new_id)
                    new_id += 1
        self.cat_part_name = cat_part_name
        self.class_ids = self.cat_part_id = cat_part_id
        self.nclass = len(cat_part_id)

        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        if self.split == 'trn':
            return len(self.img_metadata)
        else:
            return min(len(self.img_metadata), 2500)

    def build_img_metadata(self):

        img_metadata = []
        for obj in self.cat_annos['object']:
            for part in self.cat_annos['object'][obj]['part']:
                img_metadata.extend(self.cat_annos['object'][obj]['part'][part][self.split])

        return img_metadata

    def __getitem__(self, idx):
        idx %= len(self.class_ids)
        query_img, query_mask, support_imgs, support_masks, query_img_id, support_img_ids, \
        class_sample, org_qry_imsize  = self.sample_episode(idx)

        # query_img = self.transform(query_img)
        # query_mask = torch.from_numpy(query_mask).float()
        # query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()
        # support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        # for midx, smask in enumerate(support_masks):
        #     support_masks[midx] = F.interpolate(torch.from_numpy(smask).unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        # support_masks = torch.stack(support_masks)
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        assert len(support_imgs) == 1 and len(support_masks) == 1
        support_imgs, support_masks = support_imgs[0], support_masks[0]
        query_img, support_imgs = [to_tensor(x) for x in [query_img, support_imgs]]
        query_mask, support_masks = torch.from_numpy(query_mask)[None].float(), torch.from_numpy(support_masks)[None].float()

        batch = {'image': query_img,
                 'label': query_mask*255,
                 #'org_query_imsize': org_qry_imsize,
                 'image_dual': support_imgs,
                 'label_dual': support_masks*255,
                 'is_inst': False,
                 'class_name': '',
                "shape": torch.tensor(query_img.shape[-2:]),
                "imidx": torch.from_numpy(np.array(idx)),
                 'class_id': torch.tensor(self.class_ids[self.cat_part_name.index(class_sample)])
                 }
        if self.transform:
            batch = self.transform(batch)

        if self.split in ['val', 'test']:
            batch.update({
                'ori_label':query_mask * 255,
            })

        return batch

    def sample_episode(self, idx):

        class_sample, class_sample_id = self.cat_part_name[idx], self.class_ids[idx]
        obj_n, part_n = class_sample.split('+')

        # query
        while True:
            query_img_id = np.random.choice(self.cat_annos['object'][obj_n]['part'][part_n][self.split], 1, replace=False)[0]
            anno = json.load(open(self.anno_file.format(query_img_id), 'r'))

            sel_obj_in_img = []
            for o in anno['object']:
                if o['name'] == obj_n:
                    sel_obj_in_img.append(o)

            assert len(sel_obj_in_img) > 0

            sel_obj = np.random.choice(sel_obj_in_img, 1, replace=False)[0]

            sel_parts = []
            for p in sel_obj['parts']:
                if p['name'] == part_n:
                    sel_parts.append(p)

            if not sel_parts:
                continue

            part_masks = []
            for sel_part in sel_parts:
                part_masks.extend(sel_part['mask'])
            for mask in part_masks:
                mask['counts'] = mask['counts'].encode("ascii")
            part_mask = mask_util.decode(part_masks)
            part_mask = part_mask.sum(-1) > 0

            if part_mask.size > 0:
                break

        query_img = Image.open(self.img_file.format(query_img_id)).convert('RGB')
        org_qry_imsize = query_img.size
        query_mask = part_mask
        query_obj_box = [int(sel_obj['bndbox'][b]) for b in sel_obj['bndbox']]  # xyxy

        support_img_ids = []
        support_masks = []
        support_boxes = []

        while True:  # keep sampling support set if query == support

            while True:
                support_img_id = \
                np.random.choice(self.cat_annos['object'][obj_n]['part'][part_n][self.split], 1, replace=False)[0]
                if support_img_id == query_img_id or support_img_id in support_img_ids: continue

                anno = json.load(open(self.anno_file.format(support_img_id), 'r'))

                sel_obj_in_img = []
                for o in anno['object']:
                    if o['name'] == obj_n:
                        sel_obj_in_img.append(o)

                assert len(sel_obj_in_img) > 0

                sel_obj = np.random.choice(sel_obj_in_img, 1, replace=False)[0]

                sel_parts = []
                for p in sel_obj['parts']:
                    if p['name'] == part_n:
                        sel_parts.append(p)

                if not sel_parts:
                    continue

                part_masks = []
                for sel_part in sel_parts:
                    part_masks.extend(sel_part['mask'])
                for mask in part_masks:
                    mask['counts'] = mask['counts'].encode("ascii")
                part_mask = mask_util.decode(part_masks)
                part_mask = part_mask.sum(-1) > 0

                if part_mask.size > 0:
                    break

            support_img_ids.append(support_img_id)
            support_masks.append(part_mask)
            support_boxes.append([int(sel_obj['bndbox'][b]) for b in sel_obj['bndbox']])  # xyxy
            if len(support_img_ids) == self.shot: break

        support_imgs = [Image.open(self.img_file.format(sup_img_id)).convert('RGB')
                        for sup_img_id in support_img_ids]

        if self.box_crop:
            query_img = np.asarray(query_img)
            query_img = query_img[query_obj_box[1]:query_obj_box[3], query_obj_box[0]:query_obj_box[2]]
            query_img = Image.fromarray(np.uint8(query_img))
            org_qry_imsize = query_img.size
            query_mask = query_mask[query_obj_box[1]:query_obj_box[3], query_obj_box[0]:query_obj_box[2]]

            new_support_imgs = []
            new_support_masks = []

            for sup_img, sup_mask, sup_box in zip(support_imgs, support_masks, support_boxes):
                sup_img = np.asarray(sup_img)
                sup_img = sup_img[sup_box[1]:sup_box[3], sup_box[0]:sup_box[2]]
                sup_img = Image.fromarray(np.uint8(sup_img))

                new_support_imgs.append(sup_img)
                new_support_masks.append(sup_mask[sup_box[1]:sup_box[3], sup_box[0]:sup_box[2]])

            support_imgs = new_support_imgs
            support_masks = new_support_masks

        return query_img, query_mask, support_imgs, support_masks, query_img_id, support_img_ids, class_sample, org_qry_imsize
    

def build(image_set, args):
    img_size = 518
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetPASCALPart(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
