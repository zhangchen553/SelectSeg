import torch
import a_config as a_config
from e_main import val
from utils.dataset import SegmentationDataset
from utils.model import get_model
from torch.utils.data import DataLoader
import albumentations as A
import torch.backends.cudnn as cudnn
import segmentation_models_pytorch as smp
import argparse
import cv2

if __name__ == '__main__':

    torch.cuda.empty_cache()
    cudnn.benchmark = True

    flag = 'test'

    parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type=str, default=a_config.DATASET)
    parser.add_argument('--backbone_name', type=str, default=a_config.BACKBONE)
    parser.add_argument('--decoder_name', type=str, default=a_config.DECODER)
    parser.add_argument('--aug_transforms', type=str, default=a_config.AUG)
    parser.add_argument('--loss_fn', type=str, default=a_config.LOSSFN)
    parser.add_argument('--init_lr', type=float, default=1.5e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--portion', type=float, default=95.)
    parser.add_argument('--test_portion', type=int, default=100)
    parser.add_argument('--c_ratio', type=int, default=10)

    args = parser.parse_args()

    # dataset_name = args.dataset
    backbone_name = args.backbone_name
    decoder_name = args.decoder_name
    aug_transforms = args.aug_transforms
    augmentation = a_config.AUG  # 'base'
    loss_fn = args.loss_fn  # 'BCE'
    # init_lr = args.init_lr
    seed = args.seed
    portions = [70]  # , 60]100
    test_portion = args.test_portion
    prefix = ''  # 'smooth_'  # smoothing_
    init_lrs = [2e-4]  # , 1.3e-4] Fully: HKCrack: 1.5e-4, CRACK500: 10, 20=5e-4, 50=5e-5
    # [4e-4, 4.4e-4, 3.7e-4, 4.4e-4, 1.1e-4, 2e-4, 2e-4, 2e-4, 2e-4]
    # [4e-4, 4.4e-4, 3.7e-4, 4.4e-4, 1.1e-4, 2e-4, 2e-4, 2e-4, 2e-4]
    c_ratio = args.c_ratio
    semi = True
    train_mode = 'cps'  # 'partial_fully_supervise' 'fixmatch' cps，'cps_new' 'fully_supervise'

    dataset_name = a_config.DATASET
    if dataset_name == 'CRACK500':
        # save_folder = '/Results_' + dataset_name  + '/' + 'fixmatch/'  # partial_fully_supervise/fixmatch/
        save_folder = '/Results_' + dataset_name + '/' + train_mode + '_' + str(c_ratio) + 'pct/'
    else:
        save_folder = '/Results_' + dataset_name + '/' + train_mode + '/'
    device = 'cuda:0'

    save_path = a_config.STORAGE_PATH + save_folder
    print(save_path)
    seeds = [2, 3, 4]

    model_list = []
    model_names = []

    img_trans = A.Compose([A.Resize(a_config.INPUT_IMAGE_HEIGHT, a_config.INPUT_IMAGE_WIDTH,
                                    interpolation=cv2.INTER_NEAREST)])
    testDS = SegmentationDataset(imagePaths=a_config.testImages, maskPaths=a_config.testMasks, img_trans=img_trans)
    print(f"[INFO] found {len(testDS)} examples in the test set...")
    testLoader = DataLoader(testDS,
                            batch_size=8,
                            pin_memory=a_config.PIN_MEMORY,
                            num_workers=a_config.WORKER)  # os.cpu_count()
    criteria = smp.losses.DiceLoss('binary')

    for portion, init_lr in zip(portions, init_lrs):
    # for portion in portions:
        for seed in seeds:
            model, model_name, result_name, _ = get_model(backbone_name, decoder_name, dataset_name, portion=portion,
                                                          lr=init_lr, lossfn=loss_fn, prefix=prefix,
                                                          seed=seed, aug_type=augmentation, semi=semi)
            # model_name = str(test_portion) + '_' + model_name
            # result_name = str(test_portion) + '_' + result_name

            model = model.to(device)
            print("[INFO] loading up model...")

            model_dir = save_path + model_name
            try:
                model.load_state_dict(torch.load(model_dir, map_location=device))
            except:
                print(f"[INFO] model {model_name} not found in {model_dir}")
                continue

            model_list.append(model)
            model_names.append(model_name)
            print(model_name)

            # This metric is only for one model is not assemble
            metric = val(model, loss_fn, criteria, testLoader, 1, 1, device=device, split='Testing')

            # print(metric)

        # f1_metrics = []
        # for i, index in enumerate(a_config.test_index[:len(a_config.train_test_portion)]):
        #     print(f"[INFO] found {len(index)} examples in the test set...")
        #     metric = evaluate_c(model_list, model_names,
        #                         np.array(a_config.testImages)[index], np.array(a_config.testMasks)[index], clean=True)
        #     print(metric)
        #     f1_metrics.append(metric[3])

        # This metrics is for assemble models
        # metric = evaluate_c(model_list, model_names,  # for single model [model], [model_name]
        #                     a_config.testImages, a_config.testMasks, clean=True, device=device)
        # print(metric)

    # Remove seed info in the string
    # np.save(save_path + result_name[:-10] + '_test_portions_f1.npy',
    # np.array([a_config.train_test_portion, f1_metrics]))

