# import the necessary packages
from utils.train_utils import init_log
from utils.dataset import SegmentationDataset
from utils.model import get_model
import a_config_fixmatch as a_config
from e_main import val, CallBacks

# Tool package for ML
# tqdm means "progress"
# 可以生成一个进度条，同时显示预计完成时间和速度
from tqdm import tqdm
from time import time
import numpy as np
import os
import random

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


def interleave(x, size):
    s = list(x.shape)  # [batch_size, 3, 512, 512], s[1:] = [3, 512, 512]
    return x.reshape([-1, size] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


# Define a function to initialize the weights of a specific layer
def de_interleave(x, size):
    s = list(x.shape)
    return x.reshape([size, -1] + s[1:]).transpose(0, 1).reshape([-1] + s[1:])


def fixmatch_train_step(model, optim, loss_fn, criteria, sup_loader, unsup_loader,
                        accumulation_steps, scaler, epoch, max_epochs, device=a_config.DEVICE):
    model.train()
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
            unsup_imgs_weak, unsup_imgs_strong = unsup_data
            unsup_imgs_weak, unsup_imgs_strong = unsup_imgs_weak.to(device), unsup_imgs_strong.to(device)

            unsup_bs = unsup_imgs_weak.size(0)

            with autocast():
                sup_output = model(imgs)
                op_preds = torch.sigmoid(sup_output)

                unsup_weak_output = model(unsup_imgs_weak)
                unsup_strong_output = model(unsup_imgs_strong)

                unsup_weak_output = unsup_weak_output.detach()  # no gradient from the weak prediction

                unsup_weak_preds = torch.sigmoid(unsup_weak_output)
                psedo_mask = (unsup_weak_preds > 0.5).type(torch.float32)
                unsup_strong_output_probs = torch.sigmoid(unsup_strong_output)

                # # normalize the unsupervised output to [0,1]
                # probs_min = torch.min(unsup_weak_output.flatten(1), dim=1)[0].reshape(-1, 1, 1, 1)  # [batch_size, 1, 1]
                # probs_max = torch.max(unsup_weak_output.flatten(1), dim=1)[0].reshape(-1, 1, 1, 1)
                # unsup_weak_normlize_output = (unsup_weak_output - probs_min) / (probs_max - probs_min + 1e-7)

                # # close to 0 consider as background, close to 1 consider as foreground both are confident
                mask_reference = unsup_weak_preds  # or unsup_weak_normlize_output
                unsup_img_confidence_fg_mask = (mask_reference > 0.95).type(torch.float32)
                # unsup_img_confidence_bg_mask = torch.zeros_like(mask_reference)
                # unsup_img_confidence_bg_mask = (mask_reference < 0.05).type(torch.float32)
                # unsup_img_confidence_mask = unsup_img_confidence_bg_mask + unsup_img_confidence_fg_mask

                # # Count the confident pixels for background and foreground
                # bg_loss_counted = unsup_img_confidence_bg_mask.sum()/unsup_img_confidence_mask.numel()
                # fg_loss_counted = unsup_img_confidence_fg_mask.sum()/unsup_img_confidence_mask.numel()

                # # Batch level confidence
                # unsup_batch_level_confidence = torch.mean(unsup_img_confidence_mask)

                # # Ignore the pixels that are not confident, mask need to reshape to match the smp code
                # ignore_mask = abs(unsup_img_confidence_mask-1)

                # # Different criteria for unsupervised loss
                # unsup_criteria = smp.losses.DiceLoss('binary',
                #                                      ignore_index=ignore_mask.view(unsup_bs, 1, -1))
                # unsup_criteria = nn.BCEWithLogitsLoss(pos_weight=unsup_img_confidence_mask)
                # unsup_loss = unsup_criteria(unsup_strong_output, psedo_mask)
                unsup_loss = criteria(unsup_strong_output, psedo_mask)*unsup_img_confidence_fg_mask
                unsup_loss = unsup_loss.sum()/psedo_mask.sum()

                if loss_fn == 'Focal':
                    sup_loss = criteria(op_preds, masks)
                else:
                    sup_loss = criteria(sup_output, masks)
                    # loss = 0.5*criteria(output, masks) + 0.5*BCE_criteria(output, masks)

                # warm up training
                # w = 1
                # a = 1
                # w = a*(epoch/40)**2
                # w = min(w, a)
                # w = unsup_batch_level_confidence
                # w = 0  # w = w*(bg_loss_counted + fg_loss_counted)
                # w = unsup_img_confidence_fg_mask.sum()/(psedo_mask.sum() + 1e-7)
                loss = sup_loss + unsup_loss

            batch_size = imgs.size(0)  # 不影响结果后面再改

            scaler.scale(loss).backward()

            if ((idx + 1) % accumulation_steps == 0) or (idx + 1 == len(sup_loader)):
                scaler.step(optim)
                scaler.update()
                optim.zero_grad()

            tp, fp, fn, tn = smp.metrics.get_stats(op_preds, masks.int(), mode='binary', threshold=0.5)
            train_logs['loss'].update(loss.item(), batch_size)
            train_logs['sup_loss'].update(sup_loss.item(), batch_size)
            train_logs['unsup_loss'].update(unsup_loss.item(), batch_size)
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
                f" Unsup L: {'{:.3f}'.format(train_logs['unsup_loss'].avg)}"  # w: {'{:.3f}'.format(w)} "
                # f" ({'{:.3f}'.format(bg_loss_counted)}bg+{'{:.3f}'.format(fg_loss_counted)}fg)"
                f" Dice: {'{:.3f}'.format(train_logs['dice'].avg)} IoU: {'{:.3f}'.format(train_logs['iou'].avg)}"
                f" Acc: {'{:.3f}'.format(train_logs['acc'].avg)} "
                f" P: {'{:.3f}'.format(train_logs['precision'].avg)}"
                f" R: {'{:.3f}'.format(train_logs['recall'].avg)}")
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
    parser.add_argument('--c_ratio', type=int, default=10)

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
    fm_aug = 'randaug_fm'  # 'uniaug_fm' 'randaug_fm'

    batch_size = a_config.BATCH_SIZE

    save_folder = a_config.save_folder

    sup_trainImages = a_config.sup_trainImages
    sup_trainMasks = a_config.sup_trainMasks
    unsup_trainImages = a_config.unsup_trainImages
    unsup_trainMasks = a_config.unsup_trainMasks

    if dataset_name == 'CRACK500':
        sup_trainImages = []
        sup_trainMasks = []
        unsup_trainImages = []
        unsup_trainMasks = []
        c_ratio = args.c_ratio
        data_output_path = a_config.STORAGE_PATH + '/DATASET/' + dataset_name + \
                           '/data_split_txt_' + str(c_ratio) + 'pct/'
        TRAIN_MASK_PATH = a_config.DATA_PATH + '/train_crop_mask_' + str(c_ratio) + 'pct/'
        save_folder = '/Results_' + dataset_name + '/' + 'fixmatch_' + str(c_ratio) + 'pct/'

        partial_type = ''
        with open(data_output_path + 'sup_' + dataset_name + '_train_' + str(portion) + 'pct'
                  + partial_type + '.txt') as file:
            for line in file:
                sup_trainImages.append(a_config.TRAIN_IMG_PATH + line.strip())
                sup_trainMasks.append(TRAIN_MASK_PATH + line.strip())

        with open(data_output_path + 'unsup_' + dataset_name + '_train_' + str(100-portion) + 'pct'
                  + partial_type + '.txt') as file:
            for line in file:
                unsup_trainImages.append(a_config.TRAIN_IMG_PATH + line.strip())
                unsup_trainMasks.append(TRAIN_MASK_PATH + line.strip())

        print('Data path:', TRAIN_MASK_PATH)

    save_path = a_config.STORAGE_PATH + save_folder

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    cudnn.benchmark = True
    cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    earlyStopEpoch = 50

    model, model_name, result_name, img_trans = get_model(backbone_name, decoder_name, dataset_name,
                                                          portion=portion, lr=init_lr, lossfn=loss_fn,
                                                          h=a_config.INPUT_IMAGE_HEIGHT, w=a_config.INPUT_IMAGE_WIDTH,
                                                          seed=seed, aug_type=aug_transforms, semi=True)

    print(model_name)
    model = model.to(device)

    writer = SummaryWriter(comment=result_name[:-4])

    sup_trainDS = SegmentationDataset(imagePaths=sup_trainImages, maskPaths=sup_trainMasks,
                                      img_trans=img_trans, aug_trans=aug_transforms)

    unsup_trainDs = SegmentationDataset(imagePaths=unsup_trainImages, maskPaths=unsup_trainMasks,
                                        img_trans=img_trans, fixmatch=True, fm_aug=fm_aug)

    valDS = SegmentationDataset(imagePaths=a_config.valImages, maskPaths=a_config.valMasks,
                                img_trans=img_trans)

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

    sup_trainLoader = DataLoader(sup_trainDS, #  shuffle=True,
                                 batch_size=batch_size,
                                 pin_memory=a_config.PIN_MEMORY,
                                 num_workers=a_config.WORKER,
                                 sampler=labeled_sampler)  #
    unsup_trainLoader = DataLoader(unsup_trainDs, #  shuffle=True,
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
        testDS = SegmentationDataset(imagePaths=a_config.testImages, maskPaths=a_config.testMasks, img_trans=img_trans)
        print(f"[INFO] found {len(testDS)} examples in the test set...")
        testLoader = DataLoader(testDS,
                                batch_size=8,
                                pin_memory=a_config.PIN_MEMORY,
                                num_workers=a_config.WORKER)  # os.cpu_count()
    else:
        testLoader = None

    criteria = smp.losses.DiceLoss('binary')  # DiceLoss()

    if loss_fn == 'Focal':
        criteria = smp.losses.FocalLoss('binary')
    elif loss_fn == 'BCE':
        criteria = nn.BCEWithLogitsLoss()
    elif loss_fn == 'wBCE':
        criteria = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([7]).to(device))

    epochs = a_config.MAX_epochs
    optimizer = optim.AdamW(model.parameters(), lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # optimizer = optim.Adam(model.parameters(),lr=init_lr, betas=(0.9, 0.999),eps=1e-08)
    # optimizer = optim.SGD(model.parameters(), lr=init_lr, momentum=0.9, weight_decay = 1e-4)

    encoder = []
    decoder = []
    for name, param in model.named_parameters():
        if 'encoder' in name:
            encoder.append(param)
        else:
            decoder.append(param)

    # No need to set different learning rate for encoder and decoder
    # optimizer = optim.AdamW([{'params': encoder}, {'params': decoder}],
    #                         lr=init_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
    # optimizer.param_groups[0]['lr'] = init_lr
    # optimizer.param_groups[1]['lr'] = init_lr

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

    # if not os.path.exists(save_path):
    #     os.makedirs(save_path)
    # else:
    #     model_path = save_path + model_name
    #     # if os.path.exists(model_path):
    #     #     model.load_state_dict(torch.load(model_path, map_location=device))

    for epoch in range(1, epochs + 1):
        train_logs = fixmatch_train_step(model, optimizer, loss_fn,
                                         criteria, sup_trainLoader, unsup_trainLoader,
                                         accumulation_steps, scaler, epoch, epochs, device=device)
        scheduler.step()
        # print('{:.1E}'.format(optimizer.param_groups[0]['lr']))
        val_logs = val(model, loss_fn, criteria, valLoader, epoch, epochs, device=device)

        for stage in ['train_', 'val_']:  # skip val for now
            for metrics in ['loss', 'dice', 'iou', 'acc', 'precision', 'recall']:
                if stage == 'train_':
                    results[stage + metrics].append(train_logs[metrics].avg)
                    writer.add_scalar(metrics + '/train', train_logs[metrics].avg, epoch)
                if stage == 'val_':
                    results[stage + metrics].append(val_logs[metrics].avg)
                    writer.add_scalar(metrics + '/val', val_logs[metrics].avg, epoch)

        cb.saveBestModel(val_logs['dice'].avg, model, save_path, model_name, result_name, epoch)
        if cb.earlyStoping(val_logs['dice'].avg, earlyStopEpoch):
            print("Early stopping")
            break

    writer.close()

    # ----------------------------- Results on Test Best Model --------------------------------------------------

    model_path = save_path + model_name
    result_path = save_path + result_name

    # np.save(result_path,results)
    model_best = model
    model_best = model_best.to(device)
    print("loading best model...")
    model_best.load_state_dict(torch.load(model_path, map_location=device))

    valid_logs = val(model_best, loss_fn, criteria, valLoader, 1, 1, device=device)
    tests_logs = val(model_best, loss_fn, criteria, testLoader, 1, 1, split="Testing", device=device)
    print('\n')

