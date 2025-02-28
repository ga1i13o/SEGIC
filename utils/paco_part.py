r""" PACO-Part few-shot semantic segmentation dataset """
import os
import pickle

from torch.utils.data import Dataset
import torch.nn.functional as F
import torch
import PIL.Image as Image
import numpy as np
from .transform_utils import polygons_to_bitmask
import pycocotools.mask as mask_util
from torchvision import transforms
from os.path import join


class DatasetPACOPart(Dataset):
    def __init__(self, datapath, fold, transform, split, shot, box_crop=True, use_original_imgsize=False):
        self.split = 'val' if split in ['val', 'test'] else 'trn'
        self.fold = fold
        self.nfolds = 4
        self.nclass = 448
        self.benchmark = 'paco_part'
        self.shot = shot
        self.img_path =  join(datapath, 'PACO-Part', 'coco')
        self.anno_path = join(datapath, 'PACO-Part', 'paco')

        self.transform = transform
        self.box_crop = box_crop

        self.class_ids_ori, self.cid2img, self.img2anno = self.build_img_metadata_classwise()
        self.class_ids_c = {cid: i for i, cid in enumerate(self.class_ids_ori)}
        self.class_ids = sorted(list(self.class_ids_c.values()))
        self.img_metadata = self.build_img_metadata()

    def __len__(self):
        return len(self.img_metadata) if self.split == 'trn' else 2500

    def __getitem__(self, idx):
        # ignores idx during training & testing and perform uniform sampling over object classes to form an episode
        # (due to the large size of the COCO dataset)
        ok = False
        while not ok:
            ok = True
            query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize = self.load_frame()
            for smask in support_masks:
                if 0 in smask.shape:
                    ok = False
                    break
            if 0 in query_mask.shape:
                ok = False
            
        # query_img = self.transform(query_img)
        # query_mask = query_mask.float()
        # query_mask = F.interpolate(query_mask.unsqueeze(0).unsqueeze(0).float(), query_img.size()[-2:], mode='nearest').squeeze()

        # support_imgs = torch.stack([self.transform(support_img) for support_img in support_imgs])
        # for midx, smask in enumerate(support_masks):
        #     support_masks[midx] = F.interpolate(smask.unsqueeze(0).unsqueeze(0).float(), support_imgs.size()[-2:], mode='nearest').squeeze()
        # support_masks = torch.stack(support_masks)
        def to_tensor(x):
            return torch.tensor(np.array(x), dtype=torch.float32).permute(2,0,1)
        assert len(support_imgs) == 1 and len(support_masks) == 1
        support_imgs, support_masks = support_imgs[0], support_masks[0]
        query_img, support_imgs = [to_tensor(x) for x in [query_img, support_imgs]]
        query_mask, support_masks = query_mask[None].float(), support_masks[None].float()

        batch = {'image': query_img,
                 'label': query_mask*255,
                 'query_name': query_name,
                 #'org_query_imsize': org_qry_imsize,
                 'image_dual': support_imgs,
                 'label_dual': support_masks*255,
                 'support_names': support_names,
                 'class_id': torch.tensor(self.class_ids_c[class_sample]),
                 'is_inst': False,
                 'class_name': '',
                "shape": torch.tensor(query_img.shape[-2:]),
                "imidx": torch.from_numpy(np.array(idx)),
        }
        if self.transform:
            batch = self.transform(batch)
        if self.split in ['val', 'test']:
            batch.update({
                'ori_label':query_mask * 255,
            })

        return batch

    def build_img_metadata_classwise(self):

        with open(join(self.anno_path, 'paco_part_train.pkl'), 'rb') as f:
            train_anno = pickle.load(f)
        with open(join(self.anno_path, 'paco_part_val.pkl'), 'rb') as f:
            test_anno = pickle.load(f)

        # Remove Duplicates
        new_cid2img = {}

        for cid_id in test_anno['cid2img']:
            id_list = []
            if cid_id not in new_cid2img:
                new_cid2img[cid_id] = []
            for img in test_anno['cid2img'][cid_id]:
                img_id = list(img.keys())[0]
                if img_id not in id_list:
                    id_list.append(img_id)
                    new_cid2img[cid_id].append(img)
        test_anno['cid2img'] = new_cid2img

        train_cat_ids = list(train_anno['cid2img'].keys())
        test_cat_ids = [i for i in list(test_anno['cid2img'].keys()) if len(test_anno['cid2img'][i]) > self.shot]
        assert len(train_cat_ids) == self.nclass

        nclass_trn = self.nclass // self.nfolds

        if self.split != -1:
            class_ids_val = [train_cat_ids[self.fold + self.nfolds * v] for v in range(nclass_trn)]
            class_ids_val = [x for x in class_ids_val if x in test_cat_ids]
            class_ids_trn = [x for x in train_cat_ids if x not in class_ids_val]
        else:
            class_ids_val = [train_cat_ids[0 + self.nfolds * v] for v in range(nclass_trn)]
            class_ids_val = [x for x in class_ids_val if x in test_cat_ids]
            class_ids_trn = train_cat_ids

        class_ids = class_ids_trn if self.split == 'trn' else class_ids_val
        img_metadata_classwise = train_anno if self.split == 'trn' else test_anno
        cid2img = img_metadata_classwise['cid2img']
        img2anno = img_metadata_classwise['img2anno']

        return class_ids, cid2img, img2anno

    def build_img_metadata(self):
        img_metadata = []
        for k in self.cid2img.keys():
            img_metadata += self.cid2img[k]
        return img_metadata

    def get_mask(self, segm, image_size):

        if isinstance(segm, list):
            # polygon
            polygons = [np.asarray(p) for p in segm]
            mask = polygons_to_bitmask(polygons, *image_size[::-1])
        elif isinstance(segm, dict):
            # COCO RLE
            mask = mask_util.decode(segm)
        elif isinstance(segm, np.ndarray):
            assert segm.ndim == 2, "Expect segmentation of 2 dimensions, got {}.".format(
                segm.ndim
            )
            # mask array
            mask = segm
        else:
            raise NotImplementedError

        return torch.tensor(mask)

    def load_frame(self):
        class_sample = np.random.choice(self.class_ids_ori, 1, replace=False)[0]
        query = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
        query_id, query_name = list(query.keys())[0], list(query.values())[0]
        query_name = '/'.join( query_name.split('/')[-2:])
        query_img = Image.open(os.path.join(self.img_path, query_name)).convert('RGB')
        org_qry_imsize = query_img.size
        query_annos = self.img2anno[query_id]

        query_obj_dict = {}

        for anno in query_annos:
            if anno['category_id'] == class_sample:
                obj_id = anno['obj_ann_id']
                if obj_id not in query_obj_dict:
                    query_obj_dict[obj_id] = {
                        'obj_bbox': [],
                        'segms': []
                    }
                query_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                query_obj_dict[obj_id]['segms'].append(self.get_mask(anno['segmentation'], org_qry_imsize)[None, ...])

        sel_query_id = np.random.choice(list(query_obj_dict.keys()), 1, replace=False)[0]
        query_obj_bbox = query_obj_dict[sel_query_id]['obj_bbox'][0]
        query_part_masks = query_obj_dict[sel_query_id]['segms']
        query_mask = torch.cat(query_part_masks, dim=0)
        query_mask = query_mask.sum(0) > 0

        support_names = []
        support_pre_masks = []
        support_boxes = []
        while True:  # keep sampling support set if query == support
            support = np.random.choice(self.cid2img[class_sample], 1, replace=False)[0]
            support_id, support_name = list(support.keys())[0], list(support.values())[0]
            support_name = '/'.join(support_name.split('/')[-2:])
            if query_name != support_name:
                support_names.append(support_name)
                support_annos = self.img2anno[support_id]

                support_obj_dict = {}
                for anno in support_annos:
                    if anno['category_id'] == class_sample:
                        obj_id = anno['obj_ann_id']
                        if obj_id not in support_obj_dict:
                            support_obj_dict[obj_id] = {
                                'obj_bbox': [],
                                'segms': []
                            }
                        support_obj_dict[obj_id]['obj_bbox'].append(anno['obj_bbox'])
                        support_obj_dict[obj_id]['segms'].append(anno['segmentation'])

                sel_support_id = np.random.choice(list(support_obj_dict.keys()), 1, replace=False)[0]
                support_obj_bbox = support_obj_dict[sel_support_id]['obj_bbox'][0]
                support_part_masks = support_obj_dict[sel_support_id]['segms']

                support_boxes.append(support_obj_bbox)
                support_pre_masks.append(support_part_masks)

            if len(support_names) == self.shot:
                break

        support_imgs = []
        support_masks = []
        for support_name, support_pre_mask in zip(support_names, support_pre_masks):
            support_img = Image.open(os.path.join(self.img_path, support_name)).convert('RGB')
            support_imgs.append(support_img)
            org_sup_imsize = support_img.size
            sup_masks = []
            for pre_mask in support_pre_mask:
                sup_masks.append(self.get_mask(pre_mask, org_sup_imsize)[None, ...])
            support_mask = torch.cat(sup_masks, dim=0)
            support_mask = support_mask.sum(0) > 0

            support_masks.append(support_mask)

        if self.box_crop:
            query_img = np.asarray(query_img)
            query_img = query_img[int(query_obj_bbox[1]):int(query_obj_bbox[1]+query_obj_bbox[3]), int(query_obj_bbox[0]):int(query_obj_bbox[0]+query_obj_bbox[2])]
            query_img = Image.fromarray(np.uint8(query_img))
            org_qry_imsize = query_img.size
            query_mask = query_mask[int(query_obj_bbox[1]):int(query_obj_bbox[1]+query_obj_bbox[3]), int(query_obj_bbox[0]):int(query_obj_bbox[0]+query_obj_bbox[2])]

            new_support_imgs = []
            new_support_masks = []

            for sup_img, sup_mask, sup_box in zip(support_imgs, support_masks, support_boxes):
                sup_img = np.asarray(sup_img)
                sup_img = sup_img[int(sup_box[1]):int(sup_box[1]+sup_box[3]), int(sup_box[0]):int(sup_box[0]+sup_box[2])]
                sup_img = Image.fromarray(np.uint8(sup_img))

                new_support_imgs.append(sup_img)
                new_support_masks.append(sup_mask[int(sup_box[1]):int(sup_box[1]+sup_box[3]), int(sup_box[0]):int(sup_box[0]+sup_box[2])])

            support_imgs = new_support_imgs
            support_masks = new_support_masks

        return query_img, query_mask, support_imgs, support_masks, query_name, support_names, class_sample, org_qry_imsize
    

def build(image_set, args):
    img_size = 640
    transform = transforms.Compose([
        transforms.Resize(size=(img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    dataset = DatasetPACOPart(datapath=args.data_root, fold=args.fold, transform=transform,
                 shot=args.shots, split=image_set)

    return dataset
