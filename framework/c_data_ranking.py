import os.path
import a_config

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import argparse
import json


def identifying_treating_outliers(df, col, remove_or_fill_with_quartile):
    q1 = df[col].quantile(0.25)

    q3 = df[col].quantile(0.75)

    iqr = q3 - q1

    lower_fence = q1 - 1.5 * (iqr)

    upper_fence = q3 + 1.5 * (iqr)

    print('Lower Fence;', lower_fence)

    print('Upper Fence:', upper_fence)

    print('Total number of outliers are left:', df[df[col] - upper_fence].shape[0])

    if remove_or_fill_with_quartile == "drop":

        df.drop(df.loc[df[col] < lower_fence].index, inplace=True)

        df.drop(df.loc[df[col] > upper_fence].index, inplace=True)

    elif remove_or_fill_with_quartile == "fill":

        df[col] = np.where(df[col] < lower_fence, lower_fence, df[col])

        df[col] = np.where(df[col] > upper_fence, upper_fence, df[col])


def get_indices_of_points_inside_ellipse(points, center, major_axis, minor_axis):
    cx, cy = center

    # Calculate the values of the equation of the ellipse for all points
    ellipse_equation = (((points[0] - cx) ** 2) / (major_axis ** 2)) + (((points[1] - cy) ** 2) / (minor_axis ** 2))

    # Get the indices of points that lie inside the ellipse
    inside_indices = np.where(ellipse_equation <= 1)[0]
    outside_indices = np.where(ellipse_equation > 1)[0]

    return inside_indices, outside_indices
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=a_config.DATASET)
    parser.add_argument('--c_ratio', type=int, default=a_config.c_ratio)
    parser.add_argument('--flag', type=str, default='train_')
    parser.add_argument('--portions', nargs='+', type=int, default=[90, 80, 70, 60, 50, 40, 30, 20, 10])
    args = parser.parse_args()

    prefix = ''
    flag = args.flag
    dataset_name = args.dataset
    data_output_path = a_config.STORAGE_PATH + '/Results_' + dataset_name + '/fully_supervise/data_split_txt/'
    score_path = a_config.STORAGE_PATH + '/Results_' + dataset_name + '/fully_supervise/'

    if dataset_name == 'CRACK500':
        c_ratio = args.c_ratio
        score_path = (a_config.STORAGE_PATH + '/Results_' + dataset_name +
                      '/fully_supervise_' + str(c_ratio) + 'pct/')
        data_output_path = (a_config.STORAGE_PATH + '/Results_' + dataset_name +
                            '/fully_supervise_' + str(c_ratio) + 'pct/' + 'data_split_txt/')

    if not os.path.exists(data_output_path):
        os.makedirs(data_output_path)

    with open(score_path + 'c_' + dataset_name + '_' + flag + 'f1_vs_f1_var_mses.json') as f:
        f1_vs_f1_var_mse = json.load(f)

    f1s = np.array(f1_vs_f1_var_mse[0])
    one_minus_f1s = 1 - f1s
    f1_vars = np.array(f1_vs_f1_var_mse[1])
    one_minus_f1s_f1_vars = [one_minus_f1s, f1_vars]
    mses = np.array(f1_vs_f1_var_mse[2])
    paths = np.array(f1_vs_f1_var_mse[3])

    num_data = len(paths)

    fig_error_f1var, ax_error_f1var = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig_f1_f1var_hist, ax_f1_f1var_hist = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    fig_f1_f1var_scatter, ax_f1_f1var_scatter = plt.subplots(figsize=(6, 6))

    # -------- (0) f1 and f1 variance histogram and scatter ---------------

    # Plot a histogram of the data
    ax_f1_f1var_hist[0, 0].hist(1 - f1s, bins=20, align='left', edgecolor='black')
    ax_f1_f1var_hist[0, 0].set_xlabel('1 - F1')
    ax_f1_f1var_hist[0, 0].set_ylabel('Frequency')

    # Plot a histogram of the data
    ax_f1_f1var_hist[0, 1].hist(f1_vars, bins=20, align='left', edgecolor='black')
    ax_f1_f1var_hist[0, 1].set_xlabel('F1 variance')
    ax_f1_f1var_hist[0, 1].set_ylabel('Frequency')

    # Plot a scatter of the data
    ax_f1_f1var_scatter.scatter(1 - f1s, f1_vars, s=2)
    ax_f1_f1var_scatter.set_xlim(0)
    ax_f1_f1var_scatter.set_ylim(0)
    ax_f1_f1var_scatter.set_xlabel('1 - F1')
    ax_f1_f1var_scatter.set_ylabel('F1 variance')

    # -------- (1) by percentage----------------
    portions = args.portions
    st_f1_vars_idx = np.argsort(f1_vars)
    st_mse_idx = np.argsort(mses)
    st_one_minus_f1_idx = np.argsort(1 - f1s)

    num_good_indices_f1_f1_var = [np.arange(num_data)]
    num_good_indices_mse_f1_var = [np.arange(num_data)]

    for portion in portions:
        split = round(portion / 100 * num_data)
        f1_vars_thes = f1_vars[st_f1_vars_idx[split]]
        one_minus_f1_thes = one_minus_f1s[st_one_minus_f1_idx[split]]
        mse_thes = mses[st_mse_idx[split]]

        for pct in range(5, 100):
            major_axis =  one_minus_f1s[st_one_minus_f1_idx[round(pct / 100 * num_data)]]
            minor_axis = f1_vars[st_f1_vars_idx[round(pct / 100 * num_data)]]
            good_f1_f1_var_indices, bad_f1_f1_var_indices = get_indices_of_points_inside_ellipse(one_minus_f1s_f1_vars,
                                                                                                 [0, 0],
                                                                                                 major_axis,
                                                                                                 minor_axis)


            if len(good_f1_f1_var_indices)/num_data >= portion / 100:
                ellipse = Ellipse((0, 0), major_axis, minor_axis, fill=False, color='blue')
                ax_f1_f1var_scatter.add_patch(ellipse)
                break

        for pct in range(5, 100):
            major_axis = mses[st_mse_idx[round(pct / 100 * num_data)]]
            minor_axis = f1_vars[st_f1_vars_idx[round(pct / 100 * num_data)]]
            good_f1_var_mse_indices, bad_f1_var_mse_indices = get_indices_of_points_inside_ellipse([mses, f1_vars],
                                                                                                    [0, 0],
                                                                                                    major_axis,
                                                                                                    minor_axis)

            if len(good_f1_var_mse_indices)/num_data >= portion / 100:
                # ellipse = Ellipse((0, 0), major_axis, minor_axis, fill=False, color='blue')
                # ax_f1_f1var_scatter.add_patch(ellipse)
                break


        num_good_indices_f1_f1_var.append(good_f1_f1_var_indices)
        num_good_indices_mse_f1_var.append(good_f1_var_mse_indices)


        good_f1_indices = np.where(1 - f1s < one_minus_f1_thes)[0]
        bad_f1_indices = np.where(1 - f1s >= one_minus_f1_thes)[0]

        good_f1_var_indices = np.where(f1_vars < f1_vars_thes)[0]
        bad_f1_var_indices = np.where(f1_vars >= f1_vars_thes)[0]

        good_mse_indices = np.where(mses < mse_thes)[0]
        bad_mse_indices = np.where(mses >= mse_thes)[0]

        with open(data_output_path + 'sup_' + dataset_name + '_' + flag + str(portion) + 'pct' + '.txt', 'w') as f:
            for text in paths[good_f1_var_mse_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'unsup_' + dataset_name + '_' + flag + str(100 - portion) + 'pct' + '.txt', 'w') as f:
            for text in paths[bad_f1_var_mse_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'sup_' + dataset_name + '_' + flag + str(portion) + 'pct_f1_base.txt', 'w') as f:
            for text in paths[good_f1_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'unsup_' + dataset_name + '_' + flag + str(100 - portion) + 'pct_f1_base.txt', 'w') as f:
            for text in paths[bad_f1_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'sup_' + dataset_name + '_' + flag + str(portion) + 'pct_f1_var_base.txt', 'w') as f:
            for text in paths[good_f1_var_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'unsup_' + dataset_name + '_' + flag + str(100 - portion) + 'pct_f1_var_base.txt', 'w') as f:
            for text in paths[bad_f1_var_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'sup_' + dataset_name + '_' + flag + str(portion) + 'pct_mse_base.txt', 'w') as f:
            for text in paths[good_mse_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'unsup_' + dataset_name + '_' + flag + str(100 - portion) + 'pct_mse_base.txt', 'w') as f:
            for text in paths[bad_mse_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'sup_' + dataset_name + '_' + flag + str(portion) + 'pct_f1_f1_var.txt', 'w') as f:
            for text in paths[good_f1_f1_var_indices]:
                f.write(os.path.basename(text) + '\n')
        with open(data_output_path + 'unsup_' + dataset_name + '_' + flag + str(100 - portion) + 'pct_f1_f1_var.txt', 'w') as f:
            for text in paths[bad_f1_f1_var_indices]:
                f.write(os.path.basename(text) + '\n')

    ax_error_f1var[0, 0].scatter(f1_vars[good_f1_var_mse_indices], mses[good_f1_var_mse_indices], s=2, label="Inliers", color="blue")
    ax_error_f1var[0, 0].scatter(f1_vars[bad_f1_var_mse_indices], mses[bad_f1_var_mse_indices], s=6, label="Outliers", color="red")
    ax_error_f1var[0, 0].set_xlabel('Variance')
    ax_error_f1var[0, 0].set_ylabel('Error')
    ax_error_f1var[0, 0].legend(fontsize=15, title_fontsize=15)

    num_good_indices_f1_f1_var = np.array(num_good_indices_f1_f1_var, dtype=object)

    np.save(data_output_path + 'c_' + dataset_name + '_' + flag + 'index.npy', num_good_indices_f1_f1_var)

    print([len(i) for i in num_good_indices_f1_f1_var])
    print([len(i) for i in num_good_indices_mse_f1_var])

