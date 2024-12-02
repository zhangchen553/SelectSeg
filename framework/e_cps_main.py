# import the necessary packages
import a_config_cps as a_config
from utils.train_utils import AverageMeter, init_log, OnlineMeanStd
from utils.dataset import SegmentationDataset, NpyDataset
from utils.model import get_model
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
from torch.amp import GradScaler, autocast
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


def unsup_train_step(model1, model2, optim1, optim2, loss_fn, criteria, cps_criteria, sup_loader, unsup_loader,
                     accumulation_steps, scaler, epoch, max_epochs, device=a_config.DEVICE, cps_weight=0.5):
    model1.train()
    model2.train()
    train_logs = init_log()

    bar = tqdm(sup_loader, dynamic_ncols=True, leave=False)
    unsup_loader_iter = iter(unsup_loader)

    torch.cuda.empty_cache()
    start = time()
    with torch.enable_grad():
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

            with autocast(device_type='cuda'):
                sup_output1 = model1(imgs)
                sup_output2 = model2(imgs)
                sup_preds1 = torch.sigmoid(sup_output1)
                sup_preds2 = torch.sigmoid(sup_output2)

                unsup_output1 = model1(unsup_imgs)
                unsup_output2 = model2(unsup_imgs)               
                unsup_preds1 = torch.sigmoid(unsup_output1)
                unsup_pred2 = torch.sigmoid(unsup_output2)

                pred1 = torch.cat([sup_preds1, unsup_preds1], dim=0)
                pred2 = torch.cat([sup_preds2, unsup_pred2], dim=0)
                pred_mask1 = (pred1 > 0.5).type(torch.float32)
                pred_mask2 = (pred2 > 0.5).type(torch.float32)

                cps_loss = cps_criteria(torch.cat([sup_output1, unsup_output1], dim=0), pred_mask2) + \
                           cps_criteria(torch.cat([sup_output2, unsup_output2], dim=0), pred_mask1)

                if loss_fn == 'Focal':
                    sup_loss1 = criteria(sup_preds1, masks)
                    sup_loss2 = criteria(sup_preds2, masks)
                else:
                    sup_loss1 = criteria(sup_output1, masks)
                    sup_loss2 = criteria(sup_output2, masks)
                    # loss = 0.5*criteria(output, masks) + 0.5*BCE_criteria(output, masks)

                # warm up training
                # a = cps_weight
                # w_cps = a*(epoch/60)**2
                # w_cps = min(w_cps, a)
                # weight by confidence
                confident_mask = ((pred1>0.95) & (pred2>0.95)).type(torch.float32)
                pred_mask = ((pred1>0.5) | (pred2>0.5)).type(torch.float32)
                w_cps = confident_mask.sum()/(pred_mask).sum()
                cps_loss = w_cps*cps_loss
                # w_cps = 0.5
                loss = sup_loss1 + sup_loss2 + cps_loss

            batch_size = imgs.size(0)

            scaler.scale(loss).backward()

            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(sup_loader)):

                scaler.step(optim1)
                scaler.step(optim2)
                scaler.update()
                optim1.zero_grad()
                optim2.zero_grad()

            tp, fp, fn, tn = smp.metrics.get_stats(sup_preds1, masks.int(), mode='binary', threshold=0.5)
            train_logs['loss'].update(loss.item(), batch_size)
            train_logs['sup_loss1'].update(sup_loss1.item(), batch_size)
            train_logs['sup_loss2'].update(sup_loss2.item(), batch_size)
            train_logs['cps_loss'].update(cps_loss.item(), batch_size)
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
                f" Sup L1: {'{:.3f}'.format(train_logs['sup_loss1'].avg)}"
                f" Sup L2: {'{:.3f}'.format(train_logs['sup_loss2'].avg)}" 
                f" CPS L: {'{:.3f}'.format(train_logs['cps_loss'].avg)}"
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
    s_seed = seed + 4
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
        data_output_path = (a_config.STORAGE_PATH + '/Results_' + dataset_name +
                            '/fully_supervise_' + str(c_ratio) + 'pct/' + 'data_split_txt/')
        TRAIN_MASK_PATH = a_config.DATA_PATH + '/train_crop_mask_' + str(c_ratio) + 'pct/'
        save_folder = '/Results_' + dataset_name + '/' + 'cps_' + str(c_ratio) + 'pct/'

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

    earlyStopEpoch = 1  #

    model1, model_name1, result_name1, img_trans1 = get_model(backbone_name, decoder_name, dataset_name,
                                                              portion=portion, seed=seed, lr=init_lr,
                                                              h=a_config.INPUT_IMAGE_HEIGHT, w=a_config.INPUT_IMAGE_WIDTH,
                                                              aug_type=aug_transforms, semi=True, cps=True)

    model2, model_name2, result_name2, img_trans2 = get_model(backbone_name, decoder_name, dataset_name,
                                                              portion=portion, seed=s_seed, lr=init_lr,
                                                              h=a_config.INPUT_IMAGE_HEIGHT, w=a_config.INPUT_IMAGE_WIDTH,
                                                              aug_type=aug_transforms, semi=True, cps=True)

    print(model_name1, '\n', model_name2)
    model1 = model1.to(device)
    model2 = model2.to(device)

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
    optimizer1 = optim.AdamW(model1.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    optimizer2 = optim.AdamW(model2.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(),lr=init_lr,betas=(0.9, 0.999),eps=1e-08)
    # optimizer = optim.SGD(model.parameters(),lr=init_lr, momentum=0.9, weight_decay = 1e-4)

    encoder = []
    decoder = []
    for name, param in model1.named_parameters():
        if 'encoder' in name:
            encoder.append(param)
        else:
            decoder.append(param)
    optimizer1 = optim.AdamW([{'params': encoder}, {'params': decoder}],
                             lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    optimizer1.param_groups[0]['lr'] = init_lr
    optimizer1.param_groups[1]['lr'] = init_lr

    scheduler1 = PolynomialLR(optimizer1, total_iters=epochs, power=1.0)
    scheduler2 = PolynomialLR(optimizer2, total_iters=epochs, power=1.0)

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


        train_logs = unsup_train_step(model1, model2, optimizer1, optimizer2, loss_fn,
                                      criteria, cps_criteria,
                                      sup_trainLoader, unsup_trainLoader,
                                      accumulation_steps, scaler, epoch, epochs, device=device)
        scheduler1.step()
        scheduler2.step()
        # print('{:.1E}'.format(optimizer.param_groups[0]['lr']))
        val_logs = val(model1, loss_fn, criteria, valLoader, epoch, epochs, device=device)
        tests_logs = val(model1, loss_fn, criteria, testLoader, epoch, epochs, split="Testing", device=device)

        for stage in ['train_', 'val_', 'test_']:  # skip val for now
            for metrics in ['loss', 'dice', 'iou', 'acc', 'precision', 'recall']:
                if stage == 'train_':
                    results[stage+metrics].append(train_logs[metrics].avg)
                    writer.add_scalar(metrics+'/train', train_logs[metrics].avg, epoch)
                if stage == 'val_':
                    results[stage+metrics].append(val_logs[metrics].avg)
                    writer.add_scalar(metrics+'/val', val_logs[metrics].avg, epoch)

        cb.saveBestModel(val_logs['dice'].avg, model1, save_path, model_name1, result_name1, epoch)
        if cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch):
            print("Early stopping")
            break

    writer.close()

    # ----------------------------- Results on Test Best Model --------------------------------------------------

    model_path = save_path+model_name1
    result_path = save_path+result_name1

    # np.save(result_path,results)
    model_best = model1
    model_best = model_best.to(device)
    print("loading best model...")
    model_best.load_state_dict(torch.load(model_path, map_location=device))

    valid_logs = val(model_best, loss_fn, criteria, valLoader, 1, 1, device=device)
    tests_logs = val(model_best, loss_fn, criteria, testLoader, 1, 1, split="Testing", device=device)
