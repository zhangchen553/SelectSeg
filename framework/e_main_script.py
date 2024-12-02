import os

arch = 'mit_b2'
decoder = 'FPN'
aug = 'base'
errors = []
seeds = [1, 2, 3, 4]  # 0, 1, 2, 3
loss_fn = 'Dice'
c_ratios = [50]
lr = 2e-4


common_cmd_name = "python e_main.py --decoder_name {} --backbone_name {}" \
                  " --aug_transforms {} --loss_fn {} --init_lr {} --seed {} --portion {} " \
                  "--device {} --c_ratio {}"

for c_ratio in c_ratios:
    for seed in seeds:
        cmd = common_cmd_name.format(decoder, arch, aug, loss_fn, lr, seed, 100, 'cuda:0', c_ratio)
        return_code = os.system(cmd)
        if return_code != 0:
            errors.append(f"Error running command: {cmd}")

