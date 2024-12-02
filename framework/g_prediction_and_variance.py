# USAGE
# python g_prediction_and_variance.py
from torch.backends import cudnn

# import the necessary packages
import a_config
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
import cv2
import albumentations as A
import math
from utils.model import get_model
import segmentation_models_pytorch as smp


# plot the variance map of the prediction from model with different seeds

def prepare_plot(img_list, corruptions, num_cols=4):
    num_imgs = len(img_list)
    num_rows = math.ceil(num_imgs / num_cols)
    figure = plt.figure()

    for i in range(num_imgs):
        figure.add_subplot(num_rows, num_cols, i + 1)
        plt.imshow(img_list[i])
        plt.axis('off')
        plt.title(corruptions[i])

    return figure


def make_predictions(model, imagePath, maskPath, device='cuda:1'):
    model.eval()

    img_trans = A.Compose([A.Resize(a_config.INPUT_IMAGE_HEIGHT, a_config.INPUT_IMAGE_WIDTH),
                           A.Normalize(a_config.MEAN, a_config.STD, p=1.0)])
    with torch.no_grad():
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        orig = image.copy()

        gtMask = cv2.imread(maskPath, 0)
        gtMask = cv2.resize(gtMask, (a_config.INPUT_IMAGE_WIDTH,
                                     a_config.INPUT_IMAGE_HEIGHT))

        # Input image has the shape [128, 128, 3]
        # Model accepts four-dimensional inputs of the format
        # [batch_dimension, channel_dimension, height, width]
        # torch.from_numpy() convert our image to a PyTorch tensor
        # .to(a_config.DEVICE) flash it to the current device
        image = img_trans(image=image)['image']
        image = np.transpose(image, (2, 0, 1))
        image = np.expand_dims(image, 0)
        image = torch.from_numpy(image).to(device)

        # .squeeze remove the dimension with size 1
        # Mask now is [height, width]
        predMask = model(image).squeeze()
        score_map = predMask.cpu().numpy()
        predMask = torch.sigmoid(predMask)
        predMask = predMask.cpu().numpy()
        confi_map = predMask.copy()

        predMask = (predMask > 0.5) * 255
        predMask = predMask.astype(np.uint8)

        image_list = [orig, gtMask, predMask]

        return image_list, confi_map, score_map


if __name__ == '__main__':

    plt.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams['font.size'] = 12
    # plt.rcParams['image.cmap'] = 'Blues'

    imagePaths = a_config.testImages
    GTPaths = a_config.testMasks
    dataset_name = a_config.DATASET

    backbone_name = 'mit_b2'  # , 'vgg16' 'resnet50', 'mit_b2'
    decoder_name = 'FPN'  # 'FPN', 'Unet', 'DeepLabV3Plus'
    seeds = [0, 1, 2]  # 0, 1, 2

    device = 'cuda:0'

    if dataset_name == 'CRACK500':
        c_ratio = 50
        fsl_save_path = a_config.STORAGE_PATH + '/Results_CRACK500/fully_supervise_' + str(c_ratio) + 'pct/'
        partial_fsl_save_path = a_config.STORAGE_PATH + '/Results_CRACK500/partial_fully_supervise_' \
                                + str(c_ratio) + 'pct/'
        cps_save_path = a_config.STORAGE_PATH + '/Results_CRACK500/cps_' + str(c_ratio) + 'pct/'
        fsl_srl_model_seeds = [3, 2, 1]
        fsl_srl_model_lr = [5e-5, 2.4e-4, 3.4e-4]
        portions = [100, 40, 40]
        semi_or_nots = [False, False, True]
        fsl_srl_cherry_idx = [5, 13, 18, 27, 39, 43]  # picked
    else:
        fsl_save_path = a_config.STORAGE_PATH + '/Results_HKCrack/fully_supervise/'
        partial_fsl_save_path = a_config.STORAGE_PATH + '/Results_HKCrack/partial_fully_supervise/'
        cps_save_path = a_config.STORAGE_PATH + '/Results_HKCrack/cps/'
        fsl_srl_model_seeds = [1, 1, 1]
        fsl_srl_model_lr = [5e-5, 1.5e-4, 1.3e-4]
        portions = [100, 60, 60]
        semi_or_nots = [False, False, True]
        fsl_srl_cherry_idx = [40, 92, 141, 150]  # , 176, 478]

    fsl_srl_model_pahts = [fsl_save_path, partial_fsl_save_path, cps_save_path]

    var_namelist = ['Image\nOrigin', 'Image C', "GT", 'seed0', 'seed1', 'seed2', 'variance']  # , 'Aug Ensemble'
    fsl_srl_namelist = ['Image', "Ground Truth", 'FSL', 'P-FSL', 'DRL']

    # CRACK500
    # fsl_srl_cherry_idx = [5, 13, 18, 27, 31, 33, 38, 39, 43, 45, 46, 60, 66, 74, 77, 101, 108, 128, 142, 172]
    # HKCrack
    # fsl_srl_cherry_idx = [39, 40, 48, 49, 56, 65, 92, 96, 100, 107, 137, 141, 145, 149, 150, 165, 166, 176, 183,
    #                       185, 206, 220, 247, 265, 297, 349, 369, 392, 394, 408, 473, 478, 480]
    basline_srl_cherry_idx = []

    # ------------- cherry pick Plot for compare fsl and srl ----------------------------
    # 搜到了202 for crack500

    # for n_search in range(1):  # set range(1) the loop for display
    #     fig_fsl_srl, fsl_srl_axes = plt.subplots(6, 5, figsize=(10, 12))
    #     for j, (lr, seed, path, portion, semi_or_not) in enumerate(zip(fsl_srl_model_lr, fsl_srl_model_seeds,
    #                                                     fsl_srl_model_pahts, portions, semi_or_nots)):
    #         model, model_name, _, _ = get_model(backbone_name, decoder_name, a_config.DATASET, lr=lr,
    #                                             portion=portion, seed=seed, aug_type='base', semi=semi_or_not)
    #         print(model_name)
    #
    #         model = model.to(device)
    #         model.load_state_dict(torch.load(path + model_name, map_location=device))
    #
    #         for i, idx in enumerate(fsl_srl_cherry_idx):  # display cherry picked images
    #         # for i, idx in enumerate(range(6)):  # search 6 by 6
    #         #     idx = n_search * 6 + i
    #             print(idx)
    #
    #             img = cv2.imread(imagePaths[idx])
    #             img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #             img = cv2.resize(img, (a_config.INPUT_IMAGE_WIDTH, a_config.INPUT_IMAGE_HEIGHT))
    #
    #             gtMask = cv2.imread(GTPaths[idx], 0)
    #             gtMask = cv2.resize(gtMask, (a_config.INPUT_IMAGE_WIDTH,
    #                                          a_config.INPUT_IMAGE_HEIGHT))
    #
    #             image_list, confi_map, _ = make_predictions(model, imagePaths[idx], GTPaths[idx], device=device)
    #
    #             fsl_srl_axes[i, 0].imshow(img)
    #             fsl_srl_axes[i, 1].imshow(gtMask, cmap='gray')
    #             fsl_srl_axes[i, j+2].imshow(image_list[2], cmap='gray')
    #
    #     for i, ax in enumerate(fsl_srl_axes.flatten()):
    #         ax.axis('off')
    #         if i <= 4:
    #             ax.set_title(fsl_srl_namelist[i], font={'family': 'Times New Roman', 'size': 20})
    #     fig_fsl_srl.tight_layout()
    #     plt.show()

    # ------------- cherry pick Plot for compare drl and baselines ----------------------------
    baseline_srl_namelist = ['Image', "Ground Truth", 'Baseline', 'SCE', 'LS', 'Soft Label',
                             'ADELE', 'SFL', 'T-Loss', 'SelectSeg(Ours)']
    prefixs = ['', '', 'smoothing_',  'distill_', 'adele_', 'SFL_', '', '']
    semi_or_nots = [False, False, False, False, False, False, False, True]
    loss_fns = ['Dice', 'SCE', 'BCE', 'BCE', 'Dice', 'BCE', 'TLoss', 'Dice']

    display_title = True

    if dataset_name == 'CRACK500':
        c_ratio = 50
        cps_path = a_config.STORAGE_PATH + '/Results_CRACK500/cps_' + str(c_ratio) + 'pct/'
        path = a_config.STORAGE_PATH + '/Results_CRACK500/fully_supervise_' + str(c_ratio) + 'pct/'
        portions = [100, 100, 100, 100, 100, 100, 100, 40]
        seeds = [3, 3, 2, 2, 1, 1, 1, 1]
        lrs = [5e-5, 4e-4, 8e-4, 6e-4, 2e-4, 2e-4, 2e-4, 3.4e-4]
        drl_cherry_idx = [45, 60, 77, 101]  #, 108, 172]
    else:
        path = a_config.STORAGE_PATH + '/Results_HKCrack/fully_supervise/'
        cps_path = a_config.STORAGE_PATH + '/Results_HKCrack/cps/'
        portions = [100, 100, 100, 100, 100, 100, 100, 60]
        seeds = [1, 2, 3, 3, 2, 1, 3, 1]
        lrs = [5e-5, 1e-4, 2e-4, 2e-4, 1e-4, 1e-4, 1e-4, 1.3e-4]
        drl_cherry_idx = [176, 183, 220, 478]  # , 392, 394]
    paths = [path, path, path, path, path, path, path, cps_path]

    for n_search in range(1):  # set range(1) the loop for display
        fig_fsl_srl, fsl_srl_axes = plt.subplots(4, 10, figsize=(23, 12))
        for j, (lr, seed, portion, path, semi_or_not) in enumerate(zip(lrs, seeds, portions, paths, semi_or_nots)):
            model, model_name, _, _ = get_model(backbone_name, decoder_name, a_config.DATASET, lr=lr, prefix=prefixs[j],
                                                portion=portion, seed=seed, aug_type='base', semi=semi_or_not,
                                                lossfn=loss_fns[j])
            print(model_name)

            model = model.to(device)
            model.load_state_dict(torch.load(path + model_name, map_location=device))

            for i, idx in enumerate(drl_cherry_idx):  # display cherry picked images
            # for i, idx in enumerate(range(6)):  # search 6 by 6
            #     idx = n_search * 6 + i
                print(idx)

                img = cv2.imread(imagePaths[idx])
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (a_config.INPUT_IMAGE_WIDTH, a_config.INPUT_IMAGE_HEIGHT))

                gtMask = cv2.imread(GTPaths[idx], 0)
                gtMask = cv2.resize(gtMask, (a_config.INPUT_IMAGE_WIDTH,
                                             a_config.INPUT_IMAGE_HEIGHT))

                image_list, confi_map, _ = make_predictions(model, imagePaths[idx], GTPaths[idx], device=device)

                img = np.rot90(img)
                gtMask = np.rot90(gtMask)
                image_list[2] = np.rot90(image_list[2])

                fsl_srl_axes[i, -1].imshow(img)
                fsl_srl_axes[i, -2].imshow(gtMask, cmap='gray')
                fsl_srl_axes[i, -j-3].imshow(image_list[2], cmap='gray')

        for i, ax in enumerate(fsl_srl_axes.flatten()):
            ax.axis('off')
            if i <= 9 and display_title:
                ax.set_title(baseline_srl_namelist[-i-1], font={'family': 'Times New Roman', 'size': 26}, rotation=-90)
        fig_fsl_srl.tight_layout()
        plt.show()

    # prediction variance plot for each data points
    # ------------- Plot for the variance of prediction ----------------------------

    confidence_maps = []
    clean_confidence_maps = []

    for seed in seeds:
        model, model_name, _, _ = get_model(backbone_name, decoder_name, a_config.DATASET, lr=lr,
                                            portion=portion, seed=seed, aug_type='base', semi=semi_or_not)

        # ---------------------- variance of confidence map ------------------------------------

        image_list, clean_confi_map, _ = make_predictions(model, imagePaths[idx], GTPaths[idx])

        clean_confidence_maps.append(clean_confi_map)
        # ---------------------------------------------------------------------------------------

    fig_clean_maps, clean_map_axes = plt.subplots(1, 9)

    # -------------------fig_clean_maps-----------
    for ax in clean_map_axes:
        ax.axis('off')

    variance = np.var(clean_confidence_maps, axis=0)
    avg_confidence_map = np.average(clean_confidence_maps, axis=0)
    assemble_mask = (avg_confidence_map > 0.5) * 255
    error_map = assemble_mask != gtMask

    clean_map_axes[0].imshow(img_orig)
    clean_map_axes[0].set_title(var_namelist[0])
    clean_map_axes[1].imshow(gtMask)
    clean_map_axes[1].set_title(var_namelist[2])

    for i, map in enumerate(clean_confidence_maps):
        clean_map_axes[i + 2].imshow(map)
        clean_map_axes[i + 2].set_title(var_namelist[i + 3])

    clean_map_axes[5].imshow(avg_confidence_map - 0.5)
    clean_map_axes[5].set_title('Assemble confidence')
    clean_map_axes[6].imshow(assemble_mask)
    clean_map_axes[6].set_title('Assemble mask')
    clean_map_axes[7].imshow(error_map)
    clean_map_axes[7].set_title('error map')
    clean_map_axes[8].imshow(variance)
    clean_map_axes[8].set_title(var_namelist[6])

    fig_clean_maps.suptitle(backbone_name + decoder_name)

    plt.show()
