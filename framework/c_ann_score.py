import a_config
import torch
import cv2
import numpy as np
from utils.model import get_model
from g_prediction_and_variance import make_predictions
import segmentation_models_pytorch as smp
from tqdm import tqdm
import json
import os
import glob
import argparse


# This script is used to calculate the uncertainty
# of all datapoint according to the models with different seeds
# This script also generate of confusion matrix at pixel level

def dilated_mask(gtMask):
    gtMask = torch.from_numpy(np.expand_dims(gtMask, axis=(0, 1))).to(device)
    y_dilated = torch.zeros_like(gtMask)
    yy = gtMask.squeeze().cpu().numpy()
    kernel = np.ones((a_config.SLACK, a_config.SLACK), np.uint8)
    mask = cv2.dilate(yy, kernel, iterations=1)
    # Binarize mask after dilation
    mask = np.where(mask > 0, 1., 0.)
    y_dilated[0, 0, :, :] = torch.from_numpy(mask).to(device)

    return y_dilated, gtMask


def assemble_f1(confidence_maps, y_dilated, gtMask, thres=0.5):
    avg_confi_map = np.average(confidence_maps, axis=0)
    gtMask_np = gtMask
    error_map = gtMask_np.squeeze().cpu().numpy() / 255 - avg_confi_map
    obj_mse = np.square(error_map[error_map > 0])
    bg_mse = np.square(error_map[error_map < 0])
    mse_vectors = [obj_mse, bg_mse]
    avg_confi_map = torch.from_numpy(np.expand_dims(avg_confi_map, axis=(0, 1))).to(device)
    tp, fp, _, _ = smp.metrics.get_stats(avg_confi_map, y_dilated.int(), mode='binary', threshold=thres)
    _, _, fn, tn = smp.metrics.get_stats(avg_confi_map, gtMask.int(), mode='binary', threshold=thres)
    F1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy()

    return F1, error_map, mse_vectors


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=a_config.DATASET)
    parser.add_argument('--backbone_name', type=str, default=a_config.BACKBONE)
    parser.add_argument('--decoder_name', type=str, default=a_config.DECODER)
    parser.add_argument('--device', type=str, default=a_config.DEVICE)
    parser.add_argument('--c_ratio', type=int, default=a_config.c_ratio)
    parser.add_argument('--portion', type=int, default=100)
    parser.add_argument('--semi', type=bool, default=False)
    parser.add_argument('--flag', type=str, default='train_')
    args = parser.parse_args()

    prefix = ''
    save_new_ranking = True
    retry = True  # if evaluate the dataset model again
    flag = args.flag

    dataset_name = args.dataset
    device = args.device
    backbone_name = args.backbone_name
    decoder_name = args.decoder_name
    lr = a_config.INIT_LR

    portion = args.portion
    c_ratio = args.c_ratio
    semi = args.semi
    seeds = [1, 2, 3, 4]

    # For ranking data points
    save_folder = '/Results_' + dataset_name + '/' + 'fully_supervise/'
    # data_output_path = a_config.STORAGE_PATH + '/DATASET/'+dataset_name+'/data_split_txt/'
    # Save test results
    data_output_path = a_config.STORAGE_PATH + '/Results_' + dataset_name + '/cps/'
    label_output_path = a_config.STORAGE_PATH + '/DATASET/' + dataset_name + '/train_crop_pmask_soft/'
    confi_map_output_path = a_config.STORAGE_PATH + '/DATASET/' + dataset_name + '/test_confi_map/'
    var_map_output_path = a_config.STORAGE_PATH + '/DATASET/' + dataset_name + '/test_var_map/'

    if dataset_name == 'CRACK500':
        data_output_path = a_config.STORAGE_PATH + '/Results_CRACK500/cps_' + str(c_ratio) + 'pct/'
        label_output_path = a_config.STORAGE_PATH \
                            + '/DATASET/CRACK500/train_crop_pmask_soft_' + str(c_ratio) + 'pct/'
        confi_map_output_path = a_config.STORAGE_PATH \
                                + '/DATASET/CRACK500/test_confi_map_' + str(c_ratio) + 'pct/'
        var_map_output_path = a_config.STORAGE_PATH \
                                + '/DATASET/CRACK500/test_var_map_' + str(c_ratio) + 'pct/'

    # Check part of the dataset accuracy supervise par
    check_part_data = False
    if check_part_data:
        image_names = data_output_path + 'sup_HKCrack_train_50pct_f1_f1_var.txt'
        with open(image_names, 'r') as f:
            image_names = f.readlines()

        image_paths = [a_config.TRAIN_IMG_PATH + name.strip() for name in image_names]
        mask_paths = [a_config.TRAIN_MASK_PATH + name.strip() for name in image_names]

    accu_F1 = 0
    count = 0
    thres = 0.5
    confusion_matrix = np.zeros((2, 2))  # [[tp, fp], [fn, tn]]

    if flag == 'train_':
        imagePaths = a_config.trainImages
        GTPaths = a_config.trainMasks
        if dataset_name == 'CRACK500':
            save_folder = '/Results_' + dataset_name + '/' + 'fully_supervise_' + str(c_ratio) + 'pct/'
            data_output_path = (a_config.STORAGE_PATH + '/Results_' + dataset_name +
                                '/fully_supervise_' + str(c_ratio) + 'pct/')
            TRAIN_MASK_PATH = a_config.DATA_PATH + '/train_crop_mask_' + str(c_ratio) + 'pct/'
            trainMasks = glob.glob(TRAIN_MASK_PATH + '*.png')
            GTPaths = sorted(trainMasks)
            print(TRAIN_MASK_PATH)
    elif flag == 'split':
        imagePaths = image_paths
        GTPaths = mask_paths
    elif flag == 'test_':
        imagePaths = a_config.testImages  # testImages
        GTPaths = a_config.testMasks  # testMasks
        # For Deep assemble results and calculate uncertainty of cps
        if dataset_name == 'CRACK500':
            seeds = [1, 2, 3, 4]
            lr = 2e-4
            portion = 40
            semi = True
            save_folder = '/Results_CRACK500/cps_' + str(c_ratio) + 'pct/'
        if dataset_name == 'HKCrack':
            seeds = [1, 2, 3, 4]
            lr = 1.3e-4
            portion = 60
            semi = True
            save_folder = '/Results_HKCrack/cps/'


    model_path = a_config.STORAGE_PATH + save_folder

    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)
    if not os.path.exists(label_output_path):
        os.makedirs(label_output_path)
    if not os.path.exists(confi_map_output_path):
        os.makedirs(confi_map_output_path)
    if not os.path.exists(var_map_output_path):
        os.makedirs(var_map_output_path)

    if retry:
        avg_pixel_vars = []
        avg_fg_pixel_vars = []
        F1_vars = []
        avg_n_confi_scores = []
        avg_confi_scores = []
        avg_fg_confi_scores = []
        mses = []
        paths = []
        F1s = []

        models = []
        for seed in seeds:
            model, model_name, result_name, img_trans = get_model(backbone_name, decoder_name, a_config.DATASET,
                                                                  lr=lr, lossfn='Dice', portion=portion,
                                                                  seed=seed, aug_type='base', semi=semi)
            model = model.to(device)
            model_dir = model_path + model_name
            model.load_state_dict(torch.load(model_dir, map_location=device))

            print(model_name)
            models.append(model)

        for idx, (img_path, gt_path) in enumerate(zip(imagePaths, tqdm(GTPaths))):

            paths.append(img_path)
            confidence_maps = []
            score_maps = []
            seed_F1s = []

            for model in models:
                output_list, confi_map, score_map = make_predictions(model, img_path, gt_path, device=device)
                orig, gtMask, predMask = output_list
                confidence_maps.append(confi_map)
                score_maps.append(score_map)

                confi_map = torch.from_numpy(np.expand_dims(confi_map, axis=(0, 1))).to(device)
                score_map = torch.from_numpy(np.expand_dims(score_map, axis=(0, 1))).to(device)

                y_dilated, gtMask = dilated_mask(gtMask)

                tp, fp, _, _ = smp.metrics.get_stats(confi_map, y_dilated.int(), mode='binary', threshold=thres)
                _, _, fn, tn = smp.metrics.get_stats(confi_map, gtMask.int(), mode='binary', threshold=thres)

                seed_F1 = smp.metrics.f1_score(tp, fp, fn, tn, reduction='macro-imagewise').cpu().numpy()
                seed_F1s.append(seed_F1)

                confusion_matrix[0, 0] += tp
                confusion_matrix[0, 1] += fp
                confusion_matrix[1, 0] += fn
                confusion_matrix[1, 1] += tn

            # ----------- first uncertainty: pixel variance and foreground pixel variance ------- smaller better
            var_map = np.var(confidence_maps, axis=0)
            avg_pixel_var = np.average(var_map)
            avg_fg_pixel_var = np.sum(var_map) / (np.sum(np.average(confidence_maps, axis=0) > 0.5) + 1e-6)
            avg_pixel_vars.append(avg_pixel_var)
            avg_fg_pixel_vars.append(avg_fg_pixel_var)

            # ------------ second uncertainty: dice score variance ------------- smaller better
            F1_var = np.var(seed_F1s)
            F1_vars.append(F1_var.tolist())

            # ------------- third uncertainty: average confidence -------------- bigger better
            normalized_confi_map = abs(np.average(confidence_maps, axis=0) - 0.5)
            avg_n_confi_score = np.average(normalized_confi_map)
            avg_confi_score = np.average(np.average(confidence_maps, axis=0))
            avg_fg_confi_score = (np.sum(np.average(confidence_maps, axis=0))/
                                  (np.sum(np.average(confidence_maps, axis=0) > 0.5) + 1e-6))
            avg_confi_scores.append(avg_confi_score)
            avg_fg_confi_scores.append(avg_fg_confi_score)
            avg_n_confi_scores.append(avg_n_confi_score)

            # Assemble F1 score
            F1, error_map, mse_vectors = assemble_f1(confidence_maps, y_dilated, gtMask)

            count += 1
            accu_F1 += F1
            F1s.append(F1.tolist())

            # ---------save mse for plot f1 variance vs mse ----------------------
            mse = np.average(np.square(error_map))
            mses.append(mse.tolist())

            # ------- create soft label by assemble confidence map --------------
            if flag == 'train_': # save soft label for training, accidentally overwrite with test mask
                soft_label = np.average(confidence_maps, axis=0)
                cv2.imwrite(label_output_path + os.path.basename(img_path), soft_label * 255)
            else:
                confidence_map = np.average(confidence_maps, axis=0)
                cv2.imwrite(confi_map_output_path + os.path.basename(img_path), confidence_map * 255)
                cv2.imwrite(var_map_output_path + os.path.basename(img_path), var_map * 255)

        # Normalize confusion matrix
        confusion_matrix = confusion_matrix / np.sum(confusion_matrix)

        print(confusion_matrix)

        F1 = accu_F1 / count
        print('Original F1: ', F1)

        if save_new_ranking:
            with open(data_output_path + 'c_' + prefix + dataset_name + '_' + flag + 'f1_vs_f1_var_mses.json',
                      'w') as f:
                json.dump([F1s, F1_vars, mses, paths], f)
            np.save(data_output_path + 'c_' + prefix + dataset_name + '_' + flag + 'pixel_var_and_n_confi.npy',
                    [avg_pixel_vars, avg_n_confi_scores, avg_fg_pixel_vars, avg_fg_confi_scores, avg_confi_scores])
