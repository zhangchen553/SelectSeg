import a_config
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from scipy.stats import spearmanr
import argparse


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=a_config.DATASET)
    parser.add_argument('--backbone_name', type=str, default=a_config.BACKBONE)
    parser.add_argument('--decoder_name', type=str, default=a_config.DECODER)
    parser.add_argument('--device', type=str, default=a_config.DEVICE)
    parser.add_argument('--c_ratio', type=int, default=a_config.c_ratio)
    parser.add_argument('--flag', type=str, default='test_')
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

    c_ratio = args.c_ratio

    imagePaths = a_config.testImages

    data_output_path = a_config.STORAGE_PATH + '/Results_' + dataset_name + '/cps/'
    if dataset_name == 'CRACK500':
        data_output_path = a_config.STORAGE_PATH + '/Results_CRACK500/cps_' + str(c_ratio) + 'pct/'

    # ------------------ Rank and calculate F1 -------------------
    with open(data_output_path + 'c_' + dataset_name + '_' + flag + 'f1_vs_f1_var_mses.json') as f:
        f1_vs_f1_var_mse = json.load(f)
    pixel_var_and_n_confi = np.load(data_output_path + 'c_' + dataset_name + '_' + flag + 'pixel_var_and_n_confi.npy')

    f1s = np.array(f1_vs_f1_var_mse[0])
    f1_vars = np.array(f1_vs_f1_var_mse[1])
    mses = np.array(f1_vs_f1_var_mse[2])
    avg_pixel_vars = pixel_var_and_n_confi[0]
    avg_fg_pixel_vars = pixel_var_and_n_confi[2]
    n_confi = pixel_var_and_n_confi[1]
    avg_fg_confi = pixel_var_and_n_confi[3]
    avg_confi = pixel_var_and_n_confi[4]

    sorted_f1 = sorted(f1s)
    pixel_vars_sorted_f1s = f1s[np.argsort(avg_pixel_vars)[::-1]]
    avg_fg_pixel_vars_sorted_f1s = f1s[np.argsort(avg_fg_pixel_vars)[::-1]]
    f1_vars_sorted_f1s = f1s[np.argsort(f1_vars)]
    avg_n_confi_scores_sorted_f1s = f1s[np.argsort(n_confi)]
    avg_fg_confis_sorted_f1s = f1s[np.argsort(avg_fg_confi)]
    avg_confis_sorted_f1s = f1s[np.argsort(avg_confi)]

    fg_var_paths = np.array(f1_vs_f1_var_mse[3])[np.argsort(avg_fg_pixel_vars)[::-1]]

    f1_paths = np.array(f1_vs_f1_var_mse[3])[np.argsort(f1s)]
    print('high and low f1 paths')
    print([os.path.basename(i) for i in f1_paths[-10:]])  # high f1
    print([os.path.basename(i) for i in f1_paths[10:40]])  # low f1

    # correlation between the uncertainties and f1 score
    # corre_pixel_var = np.corrcoef(f1s, avg_pixel_vars)[0, 1]
    # corre_fg_pixel_var = np.corrcoef(f1s, avg_fg_pixel_vars)[0, 1]
    # corre_avg_confi = np.corrcoef(f1s, avg_confi)[0, 1]
    # corre_avg_fg_confi = np.corrcoef(f1s, avg_fg_confi)[0, 1]
    # print('Correlation between pixel variance and F1: ', corre_pixel_var)
    # print('Correlation between fg pixel variance and F1: ', corre_fg_pixel_var)
    # print('Correlation between average confidence and F1: ', corre_avg_confi)
    # print('Correlation between fg average confidence and F1: ', corre_avg_fg_confi)

    percentages = [0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    avg_pixel_var_f1s = []
    fg_pixel_vars_f1s = []
    avg_f1_var_f1s = []
    avg_n_confi_f1s = []
    avg_fg_confi_f1s = []
    avg_confi_f1s = []
    lower_baseline = []
    upper_baseline = []

    orig_f1 = np.average(f1s)  # Change accordingly !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    print(orig_f1)

    # Assemble after softmax then average, a little higher than assemble score (resnet, mit)
    # CRACK500: 0.761994, 0.781301
    # deepcrack: 0.8700682, 0.879995
    # topo: 0.685592, 0.7059749
    # hk crack: _, 0.733712

    for pct in percentages:
        split = round(pct * len(imagePaths))
        avg_pixel_var_f1 = np.average(pixel_vars_sorted_f1s[split:])
        avg_fg_pixel_var_f1 = np.average(avg_fg_pixel_vars_sorted_f1s[split:])
        avg_f1_var_f1 = np.average(f1_vars_sorted_f1s[:-split])
        avg_n_confi_f1 = np.average(avg_n_confi_scores_sorted_f1s[split:])
        avg_fg_confi_f1 = np.average(avg_fg_confis_sorted_f1s[split:])
        avg_confi_f1 = np.average(avg_confis_sorted_f1s[split:])
        avg_sorted_f1 = np.average(sorted_f1[split:])

        # # find path of the ranking of the worst cases
        # if pct <= 0.5:
        #     fg_var_path = fg_var_paths[split:]
        #     print([os.path.basename(i) for i in fg_var_path])

        corre_pixel_var, _ = spearmanr(pixel_vars_sorted_f1s[split:], np.sort(avg_pixel_vars)[::-1][split:])
        corre_fg_pixel_var, _ = spearmanr(avg_fg_pixel_vars_sorted_f1s[split:], np.sort(avg_fg_pixel_vars)[::-1][split:])
        corre_avg_confi, _ = spearmanr(avg_confis_sorted_f1s[split:], np.sort(avg_confi)[split:])
        corre_avg_fg_confi, _ = spearmanr(avg_fg_confis_sorted_f1s[split:], np.sort(avg_fg_confi)[split:])

        print(pct * 100, '% samples')
        print('Pixel variance F1: ', avg_pixel_var_f1)
        print('F1 variance F1: ', avg_f1_var_f1)
        print('Sorted F1: ', avg_sorted_f1)
        print('Correlation between pixel variance and F1: ', corre_pixel_var)
        print('Correlation between fg pixel variance and F1: ', corre_fg_pixel_var)
        print('Correlation between average confidence and F1: ', corre_avg_confi)
        print('Correlation between fg average confidence and F1: ', corre_avg_fg_confi)

        avg_pixel_var_f1s.append(avg_pixel_var_f1 * (1 - pct) + pct)
        fg_pixel_vars_f1s.append(avg_fg_pixel_var_f1 * (1 - pct) + pct)
        avg_f1_var_f1s.append(avg_f1_var_f1 * (1 - pct) + pct)
        avg_n_confi_f1s.append(avg_n_confi_f1 * (1 - pct) + pct)
        avg_fg_confi_f1s.append(avg_fg_confi_f1 * (1 - pct) + pct)
        avg_confi_f1s.append(avg_confi_f1 * (1 - pct) + pct)

        lower_baseline.append(orig_f1 * (1 - pct) + pct)
        upper_baseline.append(avg_sorted_f1 * (1 - pct) + pct)

    coverage = (1-np.array(percentages)) * 100
    fig_pec_f1, ax_pec_f1 = plt.subplots(figsize=(10, 9))
    # ax_pec_f1.plot(coverage, avg_n_confi_f1s, ':', label='Baseline fg+bg', color='blue')
    ax_pec_f1.axhline(orig_f1,  label='Reference', color='black')
    ax_pec_f1.plot(coverage, lower_baseline, '--', label='Random', color='black')
    ax_pec_f1.plot(coverage, avg_confi_f1s, '--', label='Confidence', color='red')
    ax_pec_f1.plot(coverage, avg_fg_confi_f1s, label='Confidence foreground', color='red')
    ax_pec_f1.plot(coverage, avg_pixel_var_f1s, '--', label='Variance', color='green')
    ax_pec_f1.plot(coverage, fg_pixel_vars_f1s, label='Variance foreground', color='green')
    # ax_pec_f1.plot(percentages, avg_f1_var_f1s, label='F1 variance')
    print(lower_baseline)
    print(avg_confi_f1s)
    print(avg_fg_confi_f1s)
    print(avg_pixel_var_f1s)
    print(fg_pixel_vars_f1s)
    # ax_pec_f1.plot(coverage, upper_baseline, '--', label='Oracle', color='red')
    # ax_pec_f1.plot(percentages,
    #                [0.754402968287468, 0.7568937349319458, 0.7644602239131928, 0.7741764545440674, 0.7966354846954347,
    #                 0.8689453601837158], '--', label='dropout pixel var')
    # ax_pec_f1.plot(percentages, [0.7575772178173065, 0.7622562408447265, 0.7735552370548249, 0.7928709149360657,
    #                              0.824559497833252, 0.8964112997055054], '--', label='dropout f1 var')

    ax_pec_f1.set_xlabel('Coverage (%)', fontsize=18)  # Ratio of images labeled by human
    ax_pec_f1.set_ylabel('F1 score', fontsize=18)
    ax_pec_f1.grid()
    ax_pec_f1.legend()
    ax_pec_f1.set_xlim(50, 100)
    ax_pec_f1.set_xticks(np.arange(50, 105, 5))
    if dataset_name == 'CRACK500':
        ax_pec_f1.set_ylim(0.75, 1.0)
        ax_pec_f1.set_yticks(np.arange(0.75, 1.025, 0.025))
        # ax_pec_f1.set_title(dataset_name)
    elif dataset_name == 'HKCrack':
        ax_pec_f1.set_ylim(0.70, 0.95)
        ax_pec_f1.set_yticks(np.arange(0.7, 1.025, 0.025))
        # ax_pec_f1.set_title('Façade Crack ')

    fig_error_f1var, ax_error_f1var = plt.subplots(figsize=(5, 5))

    ax_error_f1var.scatter(f1_vars, mses, s=10)
    # ax_error_f1var.scatter(variance[error_map > 0], mse_vectors[0], label='crack', s=10)
    # ax_error_var.scatter(variance[error_map < 0], mse_vectors[1], label='bg', s=10)

    ax_error_f1var.set_xlabel('Variance')
    ax_error_f1var.set_ylabel('Error')
    ax_error_f1var.set_title(dataset_name)

    # save the plot
    save_path = a_config.STORAGE_PATH + '/Results_' + dataset_name + '/cps/'
    if dataset_name == 'CRACK500':
        save_path = a_config.STORAGE_PATH + '/Results_CRACK500/cps_' + str(c_ratio) + 'pct/'
    fig_pec_f1.savefig(save_path + 'coverage_f1.png')

    plt.show()

    print(np.argsort(f1_vars)[-16:])

