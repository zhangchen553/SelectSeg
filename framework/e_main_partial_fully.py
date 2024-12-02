# import the necessary packages
# the dataset also need to be changed in the a_config.py or the get_model function as the get model function
# resize the image to the setting in the a_config.py
import a_config_partial_fully as a_config
from utils.train_utils import AverageMeter, init_log
from utils.dataset import SegmentationDataset, NpyDataset
from utils.model import get_model
from e_main import val, CallBacks
import cv2
import glob

# Tool package for ML
# tqdm means "progress"
# 可以生成一个进度条，同时显示预计完成时间和速度
from tqdm import tqdm
import matplotlib.pyplot as plt
from time import time
import numpy as np
import os
import random

import segmentation_models_pytorch as smp
import argparse

import torch
import torch.optim as optim
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import PolynomialLR
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.backends.cudnn as cudnn

# import ssl
# ssl._create_default_https_context = ssl._create_unverified_context # solove the certificate verify


def train_step(model, optim, loss_fn, criteria, loader,
               accumulation_steps, scaler, epoch, max_epochs, device='cuda:0'):
    model.train()
    train_logs = init_log()
    bar = tqdm(loader, dynamic_ncols=True, leave=False)
    torch.cuda.empty_cache()
    start = time()
    with torch.enable_grad():
        for idx, data in enumerate(bar):
            imgs, masks = data
            imgs, masks = imgs.to(device), masks.to(device)

            with autocast(device_type='cuda'):
                output = model(imgs)
                # output = output.squeeze(1)
                op_preds = torch.sigmoid(output)
                # masks = masks.squeeze(1)

                if loss_fn == 'Focal':
                    loss = criteria(op_preds, masks)
                else:
                    loss = criteria(output, masks)
                    # loss = 0.5*criteria(output, masks) + 0.5*BCE_criteria(output, masks)

            batch_size = imgs.size(0)

            scaler.scale(loss).backward()

            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(loader)):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            tp, fp, fn, tn = smp.metrics.get_stats(op_preds, masks.int(), mode='binary', threshold=0.5)
            train_logs['loss'].update(loss.item(), batch_size)
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
                f"Training Epoch: [{epoch}/{max_epochs}] Loss: {'{:.3f}'.format(train_logs['loss'].avg)}"
                f" Dice: {'{:.3f}'.format(train_logs['dice'].avg)} IoU: {'{:.3f}'.format(train_logs['iou'].avg)}"
                f" Accuracy: {'{:.3f}'.format(train_logs['acc'].avg)} "
                f"Precision: {'{:.3f}'.format(train_logs['precision'].avg)}"
                f" Recall: {'{:.3f}'.format(train_logs['recall'].avg)}")
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
    parser.add_argument('--device', type=str, default=a_config.DEVICE)
    parser.add_argument('--prefix', type=str, default='')
    parser.add_argument('--c_ratio', type=int, default=50)
    parser.add_argument('--r_metric', type=str, default='')

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
    prefix = args.prefix
    rank_metric = args.r_metric

    batch_size = a_config.BATCH_SIZE

    TRAIN_IMG_PATH = a_config.TRAIN_IMG_PATH
    TRAIN_MASK_PATH = a_config.TRAIN_MASK_PATH

    data_output_path = a_config.data_output_path
    save_folder = a_config.save_folder

    if dataset_name == 'CRACK500':
        c_ratio = args.c_ratio
        data_output_path = (a_config.STORAGE_PATH + '/Results_' + dataset_name +
                            '/fully_supervise_' + str(c_ratio) + 'pct/' + 'data_split_txt/')
        TRAIN_MASK_PATH = a_config.DATA_PATH + '/train_crop_mask_' + str(c_ratio) + 'pct/'
        save_folder = '/Results_' + dataset_name + '/' + 'partial_fully_supervise_' + str(c_ratio) + 'pct/'
        print('Data path:', TRAIN_MASK_PATH)

    sup_trainImages = []
    sup_trainMasks = []
    with open(data_output_path + 'sup_' + dataset_name + '_train_' +
              str(portion) + 'pct' + rank_metric + '.txt') as file:
        for line in file:
            sup_trainImages.append(TRAIN_IMG_PATH + line.strip())
            sup_trainMasks.append(TRAIN_MASK_PATH + line.strip())
    trainImages = sup_trainImages
    trainMasks = sup_trainMasks

    save_path = a_config.STORAGE_PATH + save_folder
    print('save path:', save_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cudnn.benchmark = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    earlyStopEpoch = 50

    model, model_name, result_name, img_trans = get_model(backbone_name, decoder_name, dataset_name, portion=portion,
                                                          lr=init_lr, lossfn=loss_fn, prefix=prefix,
                                                          h=a_config.INPUT_IMAGE_HEIGHT, w=a_config.INPUT_IMAGE_WIDTH,
                                                          seed=seed, aug_type=aug_transforms)
    print(model_name)
    model = model.to(device)

    writer = SummaryWriter(comment=result_name[:-4])

    # if a_config.DATASET == 'CRACK500':
    #     imgs = np.load(a_config.DATA_PATH + '/images_train.npy', allow_pickle=True)
    #     masks = np.load(a_config.DATA_PATH + '/masks_train.npy', allow_pickle=True)
    #     trainDS = NpyDataset(data=imgs, targets=masks, img_trans=img_trans, aug_trans=aug_transforms,
    #                          var_min=var_min, var_max=var_max)
    # else:
    trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
                                  img_trans=img_trans, aug_trans=aug_transforms, var_min=var_min, var_max=var_max)

    valDS = SegmentationDataset(imagePaths=a_config.valImages, maskPaths=a_config.valMasks,
                                img_trans=img_trans)

    print(f"[INFO] found {len(trainDS)} examples in the training set...")
    print(f"[INFO] found {len(valDS)} examples in the val set...")

    # It represents a Python iterable over a dataset
    # Passing train and val dataset to the Pytorch DataLoader class
    # Reshuffled the data at every epoch
    # num_workers is number of threads (number of cpu)
    trainLoader = DataLoader(trainDS, shuffle=True,
                             batch_size=batch_size,
                             pin_memory=a_config.PIN_MEMORY,
                             num_workers=a_config.WORKER)  # if os.cpu_count() 页面文件太小，无法完成操作


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
        testDS = SegmentationDataset(imagePaths=a_config.testImages, maskPaths=a_config.testMasks, img_trans=img_trans)
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

    BCE_criteria = nn.BCEWithLogitsLoss()

    if loss_fn == 'BCE':
        criteria = BCE_criteria
    if loss_fn == 'wBCE':
        criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7]).to(device))

    epochs = a_config.MAX_epochs
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(),lr=init_lr,betas=(0.9, 0.999),eps=1e-08)
    # optimizer = optim.SGD(model.parameters(),lr=init_lr, momentum=0.9, weight_decay = 1e-4)

    encoder = []
    decoder = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder.append(param)
        else:
            decoder.append(param)
    optimizer = optim.AdamW([{'params': encoder}, {'params': decoder}],
                            lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    optimizer.param_groups[0]['lr'] = init_lr
    optimizer.param_groups[1]['lr'] = init_lr

    scheduler = PolynomialLR(optimizer, total_iters=epochs, power=1.0)

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


    model_path = save_path + model_name
        # if os.path.exists(model_path):
        #     model.load_state_dict(torch.load(model_path, map_location=device))

    for epoch in range(1, epochs + 1):
        train_logs = train_step(model, optimizer, loss_fn, criteria, trainLoader,
                                accumulation_steps, scaler, epoch, epochs, device=device)
        scheduler.step()
        # print('{:.1E}'.format(optimizer.param_groups[0]['lr']))
        val_logs = val(model, loss_fn, criteria, valLoader, epoch, epochs, device=device)

        for stage in ['train_', 'val_']:  # skip val for now
            for metrics in ['loss', 'dice', 'iou', 'acc', 'precision', 'recall']:
                if stage == 'train_':
                    results[stage+metrics].append(train_logs[metrics].avg)
                    writer.add_scalar(metrics+'/train', train_logs[metrics].avg, epoch)
                if stage == 'val_':
                    results[stage+metrics].append(val_logs[metrics].avg)
                    writer.add_scalar(metrics+'/val', val_logs[metrics].avg, epoch)

        cb.saveBestModel(val_logs['dice'].avg, model, save_path, model_name, result_name, epoch)
        if cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch):
            print("Early stopping")
            break

    writer.close()

    # ----------------------------- Results on Test Best Model --------------------------------------------------

    model_path = save_path+model_name
    result_path = save_path+result_name

    # np.save(result_path,results)
    model_best = model
    model_best = model_best.to(device)
    print("loading best model...")
    model_best.load_state_dict(torch.load(model_path, map_location=device))

    valid_logs = val(model_best, loss_fn, criteria, valLoader, 1, 1, device=device)
    tests_logs = val(model_best, loss_fn, criteria, testLoader, 1, 1, split="Testing", device=device)

    # # Plot the losses and accuracy
    # Plot style
    plt.style.use("ggplot")
    # Create a new figure, or activate an existing figure
    plt.figure()
    plt.plot(results["train_loss"], label="train_loss")
    plt.plot(results["val_loss"], label="val_loss")
    plt.title("Loss on Dataset")
    # find position of lowest validation loss
    minposs = results["val_loss"].index(min(results["val_loss"])) + 1
    plt.axvline(minposs, linestyle='--', color='r', label='Early Stopping Checkpoint')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.ylim(0, 1)  # consistent scale
    plt.xlim(0, len(results["train_loss"]) + 1)  # consistent scale
    plt.legend(loc="upper left")
    # plt.show()



