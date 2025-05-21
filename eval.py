import os
import copy
import argparse
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import cv2
import time
import random
from PIL import Image
from typing import Dict, List, Tuple
from collections import OrderedDict, defaultdict
from typing import Sequence

from utils.dataloader import get_im_gt_name_dict, create_dataloaders, RandomHFlip, Resize, LargeScaleJitter, DualAug, ResizeVOS, DefaultBundle
from utils.dataset import CustomConcatDataset
from utils.lr_sched import adjust_learning_rate
from utils.loss_mask import loss_masks
from utils.dataloader import transforms, DistributedSampler, DataLoader, custom_collate_fn, EvalDataProcessor
import utils.misc as misc
from utils.logger import get_logger
from detectron2.layers import ROIAlign
from utils.fss import DatasetFSS, DatasetCOCO, SemCOCO, SemADE
from utils.lvis import DatasetLVIS
from utils.fss_inst import InstCOCO, get_inst_aug
from utils.paco_part import DatasetPACOPart
from utils.pascal_part import DatasetPASCALPart
from utils.pascal_voc_cd import DatasetPASCALCD
from model.segic import build_model


def get_args_parser():
    parser = argparse.ArgumentParser('HQ-SAM', add_help=False)

    parser.add_argument("--output", type=str, required=True, 
                        help="Path to the directory where masks and checkpoints will be output")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="The device to run generation on.")

    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--use_cos_sched', action='store_true')
    parser.add_argument('--warmup_epochs', default=0, type=float)
    parser.add_argument('--min_lr', default=1e-7, type=float)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--lr_drop_epoch', default=10, type=int)
    parser.add_argument('--shots', default=1, type=int)
    parser.add_argument('--max_epoch_num', default=12, type=int)
    parser.add_argument('--input_size', default=[1024,1024], type=list)
    parser.add_argument('--batch_size_train', default=4, type=int)
    parser.add_argument('--batch_size_valid', default=1, type=int)
    parser.add_argument('--model_save_fre', default=1, type=int)

    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--rank', default=0, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', type=int, help='local rank for dist')
    parser.add_argument('--find_unused_params', action='store_true')

    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--input_keys', type=str, nargs='+', default=['box','point','noise_mask'])
    parser.add_argument('--eval_keys', type=str, nargs='+', default=['box'])
    parser.add_argument('--eval_datasets', type=str, default=None)
    parser.add_argument('--n_point', type=int, default=1)
    parser.add_argument('--noised_inst', action='store_true')
    parser.add_argument('--use_dual_aug', action='store_true')
    parser.add_argument('--use_simm_prompt', action='store_true')
    parser.add_argument('--use_ref_decoder', action='store_true')
    parser.add_argument('--use_ref_refine_img', action='store_true')
    parser.add_argument('--use_bbox_head', action='store_true')
    parser.add_argument('--use_ref_keypoint', action='store_true')
    parser.add_argument('--use_corr_prompt', action='store_true')
    parser.add_argument('--open_ft', action='store_true')

    parser.add_argument('--use_dift', action='store_true')
    parser.add_argument('--encoder_model', type=str, default='dift')
    parser.add_argument('--dinov2_model', type=str, default='l')
    parser.add_argument('--use_inst_proj', action='store_true')
    parser.add_argument('--diff_text_prompt_ratio', default=1., type=float)
    parser.add_argument('--use_keypoint', action='store_true')
    parser.add_argument('--num_keypoint', default=1, type=int)
    parser.add_argument('--no_text_eval', action='store_true')
    parser.add_argument('--no_text', action='store_true')
    parser.add_argument('--eval_vos', action='store_true')
    parser.add_argument('--vos_dataset', type=str, default='davis17')
    parser.add_argument('--vos_gt_root', type=str, default='data/DAVIS/2017/Annotations/480p/')

    parser.add_argument('--eval_semseg', action='store_true')
    parser.add_argument('--custom_eval', action='store_true')
    parser.add_argument('--custom_eval_with_instance', action='store_true')
    parser.add_argument('--no_sim_prompt', action='store_true')
    parser.add_argument('--reverse_context', action='store_true')

    parser.add_argument('--use_aug_inst', action='store_true')
    parser.add_argument('--use_neg_aug_inst', action='store_true')
    parser.add_argument('--add_neg_prompt', action='store_true')
    parser.add_argument('--que_len', default=128, type=int)
    parser.add_argument('--max_inst_used', default=10, type=int)

    parser.add_argument('--use_inst_train', action='store_true')
    parser.add_argument('--max_inst', type=int, default=5)
    parser.add_argument('--use_cross_inst_prompt', action='store_true')
    
    parser.add_argument('--inst_datasets', nargs='+', default=[])
    parser.add_argument('--sem_datasets', nargs='+', default=[])
    parser.add_argument('--eval_sem_datasets', nargs='+', default=['ade'])
    parser.add_argument('--force_input_size', default=None, type=int)
    parser.add_argument('--use_task_indicator', action='store_true')
    parser.add_argument('--inst_for_simm', default=0, type=int)
    parser.add_argument('--tau_simm', default=1, type=float)
    parser.add_argument('--dataset_ratio', default=None, type=float, nargs='+')
    parser.add_argument('--samples_per_epoch', default=160000, type=int)

    parser.add_argument('--visualize', action='store_true')
    parser.add_argument("--restore-model", type=str,
                        help="The path to the hq_decoder training checkpoint for evaluation")

    return parser.parse_args()


def main(args):

    misc.init_distributed_mode(args)
    assert args.input_size[0] == args.input_size[1]
    model = build_model(args)
    os.makedirs(args.output, exist_ok=True)
    n_parameters_tot = sum(p.numel() for p in model.parameters())
    print('number of params:', n_parameters_tot)

    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    logger.info('world size: {}'.format(args.world_size))
    logger.info('rank: {}'.format(args.rank))
    logger.info('local_rank: {}'.format(args.local_rank))
    logger.info("args: " + str(args) + '\n')

    seed = args.seed + misc.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


    if torch.cuda.is_available():
        model.cuda()
    args.gpu = 0
    # model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=args.find_unused_params)

    ### --- Step 1: Train or Valid dataset ---
    args.eval = True
    aug_list_eval = [Resize(args.input_size)]

    logger.info("--- create valid dataloader ---")

    valid_dataloaders = []
    aug_list_eval = [Resize(args.input_size)]
    aug_eval = transforms.Compose(aug_list_eval)
    valid_datasets = []
    
    if args.eval_datasets is None :
        valid_datasets = [DatasetCOCO('data', idx, aug_eval, 'test', args.shots, False) for idx in range(4)]
        #valid_datasets.append(DatasetFSS('data', None, aug_eval, 'test', 1, False))
    elif args.eval_datasets.startswith('lvis'):
        fold = int(args.eval_datasets.split('_')[-1])
        valid_datasets = [DatasetLVIS('data', fold, aug_eval, 'val', 1, False)]
    elif args.eval_datasets.startswith('coco'):
        fold = int(args.eval_datasets.split('_')[-1])
        valid_datasets = [DatasetCOCO('data', fold, aug_eval, 'test', 1, False)]
    elif args.eval_datasets.startswith('pascalcd'):
        fold = int(args.eval_datasets.split('_')[-1])
        valid_datasets = [DatasetPASCALCD('data', fold, aug_eval, 'test', 1)]
    elif args.eval_datasets.startswith('fss'):
        valid_datasets = [(DatasetFSS('data', None, aug_eval, 'test', 1, False))]
    elif args.eval_datasets.startswith('paco_part'):
        fold = args.eval_datasets.split('_')[-1]
        fold = int(fold)
        valid_datasets = [DatasetPACOPart('data', fold, aug_eval, 'test', 1, False)]
    elif args.eval_datasets.startswith('pascal_part'):
        fold = args.eval_datasets.split('_')[-1]
        fold = int(fold)
        valid_datasets = [DatasetPASCALPart('data', fold, aug_eval, 'test', 1, False)]
    for valid_dataset in valid_datasets :
        valid_dataloaders.append(DataLoader(valid_dataset, args.batch_size_valid, drop_last=False, num_workers=4,
                                            collate_fn=custom_collate_fn))

    logger.info("{} valid dataloaders created".format(len(valid_dataloaders)))
    
    if not args.restore_model and (args.eval_vos or args.eval_semseg or args.auto_resume):
        output_dir = args.output
        ckpt_list = [x for x in os.listdir(output_dir) if x.startswith('epoch_')]
        ckpt_list = sorted(ckpt_list, key=lambda x: int(x.split('.')[0].split('_')[-1]), reverse=True)
        if len(ckpt_list) :
            args.restore_model = os.path.join(output_dir, ckpt_list[0])
            if args.auto_resume:
                args.start_epoch = int(ckpt_list[0].split('.')[0].split('_')[-1]) + 1

    if args.restore_model:
        logger.info("restore model from: {}".format(args.restore_model))
        if torch.cuda.is_available():
            _info = model.load_state_dict(torch.load(args.restore_model), strict=False)
            print(_info)
        else:
            model.load_state_dict(torch.load(args.restore_model,map_location="cpu"))

    evaluate(args, model, valid_dataloaders, args.visualize)

    
def compute_iou(preds, target, return_inter_union=False):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds

    if return_inter_union :
        inter_all, union_all = 0, 0
        for i in range(0,len(preds)):
            inter, union = misc.mask_iter_union(postprocess_preds[i],target[i])
            inter_all += inter
            union_all += union
        return inter_all, union_all
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.mask_iou(postprocess_preds[i],target[i])
    return iou / len(preds)

def compute_boundary_iou(preds, target):
    assert target.shape[1] == 1, 'only support one mask per image now'
    if(preds.shape[2]!=target.shape[2] or preds.shape[3]!=target.shape[3]):
        postprocess_preds = F.interpolate(preds, size=target.size()[2:], mode='bilinear', align_corners=False)
    else:
        postprocess_preds = preds
    iou = 0
    for i in range(0,len(preds)):
        iou = iou + misc.boundary_iou(target[i],postprocess_preds[i])
    return iou / len(preds)


def evaluate(args, model, valid_dataloaders, visualize=False):

    logger = get_logger(name='mmdet', log_file=os.path.join(args.output, 'log.txt'), file_mode='a')
    model.eval()
    logger.info("Validating...")
    test_stats = {}

    from utils.meter import AverageMeter

    for k in range(len(valid_dataloaders)):
        metric_logger = misc.MetricLogger(delimiter="  ")
        valid_dataloader = valid_dataloaders[k]
        logger.info('valid_dataloader len: {}'.format(len(valid_dataloader)))
        eval_meter = AverageMeter(valid_dataloader.dataset.class_ids, logger)

        # count = 0
        tot_el  = 0
        from time import time
        step = 0
        for data_val in metric_logger.log_every(valid_dataloader,50, logger=logger):
            step +=1
            s = time()
            masks_hq, bbox_preds, loss, loss_dict = model(data_val, inference=True)
            labels_ori = data_val['ori_label']
            if isinstance(labels_ori, (list, tuple)):
                labels_ori = torch.cat(labels_ori)[:, None]
            if torch.cuda.is_available():
                labels_ori = labels_ori.cuda()
            iou = compute_iou(masks_hq,labels_ori)
            boundary_iou = compute_boundary_iou(masks_hq,labels_ori)
            tot_el += time() -s
            inter, union = compute_iou(masks_hq,labels_ori, True)
            eval_meter.update(inter.cuda(), union.cuda(), data_val['class_id'][0].cuda(), loss=None)
            if step % 50 == 0:
                avg_lat = tot_el / step*2
                avg_fps = 1 / avg_lat
                print(f'fps = {avg_fps:.1f}')
            loss_dict = {"val_iou_"+str(k): iou, "val_boundary_iou_"+str(k): boundary_iou}
            loss_dict_reduced = misc.reduce_dict(loss_dict)
            metric_logger.update(**loss_dict_reduced)

        eval_meter.write_result()
        logger.info('============================')
        # gather the stats from all processes
        metric_logger.synchronize_between_processes()
        logger.info("Averaged stats: {}".format(metric_logger))
        resstat = {k: meter.global_avg for k, meter in metric_logger.meters.items() if meter.count > 0}
        test_stats.update(resstat)


    return test_stats


if __name__ == "__main__":

    args = get_args_parser()

    main(args)
