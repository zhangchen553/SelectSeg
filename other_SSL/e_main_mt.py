# import the necessary packages
from utils.train_utils import AverageMeter, init_log, OnlineMeanStd
from utils.dataset import SegmentationDataset, NpyDataset
from utils.model import get_model
from utils.optim_weight_ema import EMAWeightOptimizer
from utils.custom_loss import sigmoid_rampup
import a_config_mt as a_config
from e_main import val, CallBacks

# Tool package for ML
# tqdm means "progress"
# 可以生成一个进度条，同时显示预计完成时间和速度
from tqdm import tqdm
from time import time
import numpy as np
import os
import random
import math

import segmentation_models_pytorch as smp
import argparse

import torch
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.backends.cudnn as cudnn


# Define the learning rate scheduler with a warm-up function
def warmup_loss(epoch, warmup_epochs, sup_loss1, sup_loss2, cps_loss):
    if epoch < warmup_epochs:
        return sup_loss1 + sup_loss2
    else:
        return sup_loss1 + sup_loss2 + 0.4*cps_loss  # Your base learning rate after warm-up


def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return sigmoid_rampup(epoch, 100)


def semi_train_step(model_t, model_s, optim_t, optim_s, loss_fn, criteria, sup_loader, unsup_loader,
                    accumulation_steps, scaler, epoch, max_epochs, device=a_config.DEVICE):
    model_t.train()
    model_s.train()
    train_logs = init_log()

    bar = tqdm(sup_loader, dynamic_ncols=True, leave=False)
    unsup_loader_iter = iter(unsup_loader)

    torch.cuda.empty_cache()
    start = time()
    for idx, sup_data in enumerate(bar):
        try:
            unsup_data = next(unsup_loader_iter)
        except StopIteration:
            unsup_loader_iter = iter(unsup_loader)
            unsup_data = next(unsup_loader_iter)

        imgs, masks = sup_data
        imgs, masks = imgs.to(device), masks.to(device)
        unsup_imgs, _ = unsup_data
        unsup_imgs = unsup_imgs.to(device)

        with autocast():
            with torch.no_grad():
                sup_output_t = model_t(imgs).detach()
                unsup_output_t = model_t(unsup_imgs).detach()

            sup_output_s = model_s(imgs)
            unsup_output_s = model_s(unsup_imgs)

            sup_preds_t = torch.sigmoid(sup_output_t)
            sup_preds_s = torch.sigmoid(sup_output_s)
            unsup_preds_t = torch.sigmoid(unsup_output_t)
            unsup_pred_s = torch.sigmoid(unsup_output_s)

            pred_t = torch.cat([sup_preds_t, unsup_preds_t], dim=0)
            pred_s = torch.cat([sup_preds_s, unsup_pred_s], dim=0)

            if loss_fn == 'Focal':
                sup_loss = criteria(sup_preds_s, masks)
            else:
                sup_loss = criteria(sup_output_s, masks)
                # loss = 0.5*criteria(output, masks) + 0.5*BCE_criteria(output, masks)

            consistency_weight = get_current_consistency_weight(epoch)
            consistent_loss = torch.mean((pred_s - pred_t) ** 2)
            unsup_loss = consistent_loss*consistency_weight

            loss = sup_loss + unsup_loss

        batch_size = imgs.size(0)

        scaler.scale(loss).backward()

        if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(sup_loader)):
            scaler.step(optim_s)
            scaler.update()
            optim_s.zero_grad()
            optim_t.step()

        tp, fp, fn, tn = smp.metrics.get_stats(sup_preds_t, masks.int(), mode='binary', threshold=0.5)
        train_logs['loss'].update(loss.item(), batch_size)
        train_logs['sup_loss'].update(sup_loss.item(), batch_size)
        train_logs['unsup_loss'].update(unsup_loss.item())
        train_logs['time'].update(time() - start)
        train_logs['dice'].update(
            smp.metrics.f1_score(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy(), batch_size)
        train_logs['iou'].update(
            smp.metrics.iou_score(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy(), batch_size)
        train_logs['acc'].update(
            smp.metrics.accuracy(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy(), batch_size)
        train_logs['precision'].update(
            smp.metrics.precision(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy(), batch_size)
        train_logs['recall'].update(
            smp.metrics.sensitivity(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy(), batch_size)

        bar.set_description(
            f" Training Epoch: [{epoch}/{max_epochs}] Loss: {'{:.3f}'.format(train_logs['loss'].avg)}"
            f" Sup L: {'{:.3f}'.format(train_logs['sup_loss'].avg)}"
            f" Unsup L: {'{:.3f}'.format(train_logs['unsup_loss'].avg)}"
            f" Dice: {'{:.3f}'.format(train_logs['dice'].avg)} IoU: {'{:.3f}'.format(train_logs['iou'].avg)}"
            f" Acc: {'{:.3f}'.format(train_logs['acc'].avg)} "
            f" Prec: {'{:.3f}'.format(train_logs['precision'].avg)}"
            f" Rec: {'{:.3f}'.format(train_logs['recall'].avg)}")
    return train_logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=a_config.DATASET)
    parser.add_argument('--backbone_name', type=str, default=a_config.BACKBONE)
    parser.add_argument('--decoder_name', type=str, default=a_config.DECODER)
    parser.add_argument('--aug_transforms', type=str, default=a_config.AUG)
    parser.add_argument('--loss_fn', type=str, default=a_config.LOSSFN)
    parser.add_argument('--init_lr', type=float, default=a_config.INIT_LR)
    parser.add_argument('--seed', type=int, default=a_config.SEED)
    parser.add_argument('--portion', type=int, default=a_config.sup_portion)
    parser.add_argument('--var_min', type=float, default=10)
    parser.add_argument('--var_max', type=float, default=400)
    parser.add_argument('--c_ratio', type=int, default=a_config.c_ratio)
    parser.add_argument('--device', type=str, default=a_config.DEVICE)

    args = parser.parse_args()

    dataset_name = args.dataset
    backbone_name = args.backbone_name
    decoder_name = args.decoder_name
    aug_transforms = args.aug_transforms
    loss_fn = args.loss_fn
    init_lr = args.init_lr
    portion = args.portion
    seed = args.seed
    var_min = args.var_min
    var_max = args.var_max
    device = args.device

    batch_size = a_config.BATCH_SIZE

    TRAIN_MASK_PATH = a_config.TRAIN_MASK_PATH
    data_output_path = a_config.data_output_path
    save_folder = a_config.save_folder

    sup_trainImages = []
    sup_trainMasks = []
    unsup_trainImages = []
    unsup_trainMasks = []

    if dataset_name == 'CRACK500':
        c_ratio = args.c_ratio
        data_output_path = a_config.STORAGE_PATH + '/DATASET/' + dataset_name + \
                           '/data_split_txt_' + str(c_ratio) + 'pct/'
        TRAIN_MASK_PATH = a_config.DATA_PATH + '/train_crop_mask_' + str(c_ratio) + 'pct/'
        save_folder = '/Results_' + dataset_name + '/' + 'mt_' + str(c_ratio) + 'pct/'

        print('Data path:', TRAIN_MASK_PATH)

    partial_type = ''
    with open(data_output_path + 'sup_' + dataset_name + '_train_' + str(portion) + 'pct'
              + partial_type + '.txt') as file:
        for line in file:
            sup_trainImages.append(a_config.TRAIN_IMG_PATH + line.strip())
            sup_trainMasks.append(TRAIN_MASK_PATH + line.strip())

    with open(data_output_path + 'unsup_' + dataset_name + '_train_' + str(100 - portion) + 'pct'
              + partial_type + '.txt') as file:
        for line in file:
            unsup_trainImages.append(a_config.TRAIN_IMG_PATH + line.strip())
            unsup_trainMasks.append(TRAIN_MASK_PATH + line.strip())

    save_path = a_config.STORAGE_PATH + save_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    earlyStopEpoch = 50

    # cps set true for increase the inconsistency loss
    model_t, model_name_t, result_name1, img_trans1 = get_model(backbone_name, decoder_name, dataset_name,
                                                                portion=portion, seed=seed, lr=init_lr,
                                                                prefix='t_', lossfn=loss_fn,
                                                                h=a_config.INPUT_IMAGE_HEIGHT,
                                                                w=a_config.INPUT_IMAGE_WIDTH,
                                                                aug_type=aug_transforms, semi=True, cps=True)

    model_s, model_name2, result_name2, img_trans2 = get_model(backbone_name, decoder_name, dataset_name,
                                                               portion=portion, seed=5, lr=init_lr,
                                                               prefix='s_', lossfn=loss_fn,
                                                               h=a_config.INPUT_IMAGE_HEIGHT,
                                                               w=a_config.INPUT_IMAGE_WIDTH,
                                                               aug_type=aug_transforms, semi=True, cps=True)

    print(model_name_t, '\n', model_name2)
    model_t = model_t.to(device)
    model_s = model_s.to(device)

    # For teacher model
    for p in model_t.parameters():
        p.requires_grad = False

    writer = SummaryWriter(comment=result_name1[:-4])

    sup_trainDS = SegmentationDataset(imagePaths=sup_trainImages, maskPaths=sup_trainMasks,
                                      img_trans=img_trans1, aug_trans=aug_transforms)
    unsup_trainDs = SegmentationDataset(imagePaths=unsup_trainImages, maskPaths=unsup_trainMasks,
                                        img_trans=img_trans2, aug_trans=aug_transforms)

    valDS = SegmentationDataset(imagePaths=a_config.valImages, maskPaths=a_config.valMasks,
                                img_trans=img_trans1)

    print(f"[INFO] found {len(sup_trainDS)} examples in the supervised training set...")
    print(f"[INFO] found {len(unsup_trainDs)} examples in the unsupervised training set...")
    print(f"[INFO] found {len(valDS)} examples in the val set...")

    # Smapler make sure the number of samples in each dataset is the same
    # Determine the maximum number of samples in both datasets
    max_num_samples = max(len(sup_trainDS), len(unsup_trainDs))

    # Create weighted samplers for both datasets
    labeled_weights = [1.0 / len(sup_trainDS)] * len(sup_trainDS)
    unlabeled_weights = [1.0 / len(unsup_trainDs)] * len(unsup_trainDs)

    labeled_sampler = WeightedRandomSampler(labeled_weights, max_num_samples, replacement=True)
    unlabeled_sampler = WeightedRandomSampler(unlabeled_weights, max_num_samples, replacement=True)

    sup_trainLoader = DataLoader(sup_trainDS, # shuffle=True,
                                 batch_size=batch_size,
                                 pin_memory=a_config.PIN_MEMORY,
                                 num_workers=a_config.WORKER,
                                 sampler=labeled_sampler)  #
    unsup_trainLoader = DataLoader(unsup_trainDs, # shuffle=True,
                                  batch_size=batch_size,
                                  pin_memory=a_config.PIN_MEMORY,
                                  num_workers=a_config.WORKER,
                                  sampler=unlabeled_sampler)  #

    # The shape of val images may not all the same, so evaluate 1 by 1
    # ie batch size = 1
    valLoader = DataLoader(valDS,
                           batch_size=8,
                           pin_memory=a_config.PIN_MEMORY,
                           num_workers=a_config.WORKER)  # os.cpu_count()

    try:
        a_config.testMasks
    except NameError:
        a_config.testMasks = None

    # Just for counting number of test data points
    if a_config.testMasks:
        testDS = SegmentationDataset(imagePaths=a_config.testImages, maskPaths=a_config.testMasks, img_trans=img_trans1)
        print(f"[INFO] found {len(testDS)} examples in the test set...")
        testLoader = DataLoader(testDS,
                                batch_size=8,
                                pin_memory=a_config.PIN_MEMORY,
                                num_workers=a_config.WORKER)  # os.cpu_count()
    else:
        testLoader = None

    if loss_fn == 'Dice':
        criteria = smp.losses.DiceLoss('binary') #DiceLoss()
    if loss_fn == 'Focal':
        criteria = smp.losses.FocalLoss('binary')

    cps_criteria = smp.losses.DiceLoss('binary')

    if loss_fn == 'BCE':
        criteria = nn.BCEWithLogitsLoss()
    if loss_fn == 'wBCE':
        criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7]).to(device))

    epochs = a_config.MAX_epochs
    optimizer_t = EMAWeightOptimizer(model_t, model_s, 0.99)  # Moving average optimizer
    optimizer_s = optim.AdamW(model_s.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(),lr=init_lr,betas=(0.9, 0.999),eps=1e-08)
    # optimizer = optim.SGD(model.parameters(),lr=init_lr, momentum=0.9, weight_decay = 1e-4)

    encoder = []
    decoder = []
    for name, param in model_s.named_parameters():
        if 'encoder' in name:
            encoder.append(param)
        else:
            decoder.append(param)
    optimizer_s = optim.AdamW([{'params': encoder}, {'params': decoder}],
                              lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    optimizer_s.param_groups[0]['lr'] = init_lr
    optimizer_s.param_groups[1]['lr'] = init_lr

    scheduler1 = PolynomialLR(optimizer_s, total_iters=epochs, power=1.0)

    accumulation_steps = 4
    best_dice = 0.15
    scaler = GradScaler()

    cb = CallBacks(best_dice)

    results = {"train_loss": [], "train_dice": [], "train_iou": [], 'train_acc': [],
               "train_precision": [], "train_recall": [],
               "val_loss": [], "val_dice": [], "val_iou": [], "val_acc": [],
               "val_precision": [], "val_recall": [],
               "test_loss": [], "test_dice": [], "test_iou": [], "test_acc": [],
               "test_precision": [], "test_recall": []}

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # else:
    #     model_path = save_path + model_name1
        # if os.path.exists(model_path):
        #     model.load_state_dict(torch.load(model_path, map_location=device))

    for epoch in range(1, epochs + 1):


        train_logs = semi_train_step(model_t, model_s, optimizer_t, optimizer_s, loss_fn,
                                     criteria, sup_trainLoader, unsup_trainLoader,
                                     accumulation_steps, scaler, epoch, epochs, device=device)
        scheduler1.step()
        # print('{:.1E}'.format(optimizer.param_groups[0]['lr']))
        val_logs = val(model_t, loss_fn, criteria, valLoader, epoch, epochs, device=device)

        for stage in ['train_', 'val_', 'test_']:  # skip val for now
            for metrics in ['loss', 'dice', 'iou', 'acc', 'precision', 'recall']:
                if stage == 'train_':
                    results[stage+metrics].append(train_logs[metrics].avg)
                    writer.add_scalar(metrics+'/train', train_logs[metrics].avg, epoch)
                if stage == 'val_':
                    results[stage+metrics].append(val_logs[metrics].avg)
                    writer.add_scalar(metrics+'/val', val_logs[metrics].avg, epoch)

        cb.saveBestModel(val_logs['dice'].avg, model_t, save_path, model_name_t, result_name1, epoch)
        if cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch):
            print("Early stopping")
            break

    writer.close()

    # ----------------------------- Results on Test Best Model --------------------------------------------------

    model_path = save_path + model_name_t
    result_path = save_path+result_name1

    # np.save(result_path,results)
    model_best = model_t
    model_best = model_best.to(device)
    print("loading best model...")
    model_best.load_state_dict(torch.load(model_path, map_location=device))

    valid_logs = val(model_best, loss_fn, criteria, valLoader, 1, 1, device=device)
    tests_logs = val(model_best, loss_fn, criteria, testLoader, 1, 1, split="Testing", device=device)