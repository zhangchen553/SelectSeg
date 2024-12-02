import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):

        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}\n')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).'
                            f'  Saving model ...\n')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


class AverageMeter(object):
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def init_log():
    
    log = {
      'loss': AverageMeter(),
       'sup_loss': AverageMeter(),
      'sup_loss1': AverageMeter(),
      'sup_loss2': AverageMeter(),
      'unsup_loss': AverageMeter(),
      'cps_loss': AverageMeter(),
      'time': AverageMeter(),
      'iou': AverageMeter(),
      'dice': AverageMeter(),
      'acc': AverageMeter(),
      'precision': AverageMeter(),
      'recall': AverageMeter(),
      'f1': AverageMeter()
    }
    return log


class OnlineMeanStd:
    def __init__(self):
        pass

    def __call__(self, loader, method='strong'):
        """
        Calculate mean and std of a dataset in lazy mode (online)
        On mode strong, batch size will be discarded because we use batch_size=1 to minimize leaps.
        :param dataset: Dataset object corresponding to your dataset
        :param batch_size: higher size, more accurate approximation
        :param method: weak: fast but less accurate, strong: slow but very accurate - recommended = strong
        :return: A tuple of (mean, std) with size of (3,)
        """

        if method == 'weak':
            mean = 0.
            std = 0.
            nb_samples = 0.
            for data, y in loader:
                batch_samples = data.size(0)
                data = data.view(batch_samples, data.size(1), -1)
                mean += data.mean(2).sum(0)
                std += data.std(2).sum(0)
                nb_samples += batch_samples

            mean /= nb_samples
            std /= nb_samples

            return mean, std

        elif method == 'strong':
            cnt = 0
            fst_moment = torch.empty(3)
            snd_moment = torch.empty(3)

            for data, y in loader:
                b, c, h, w = data.shape
                nb_pixels = b * h * w
                sum_ = torch.sum(data, dim=[0, 2, 3])
                sum_of_square = torch.sum(data ** 2, dim=[0, 2, 3])
                fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
                snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

                cnt += nb_pixels

            return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)


# For beco, generate boundary mask
# https://github.com/ShenghaiRong/BECO/blob/main/datasets/transforms/imgmix.py
# modified by me for batch processing
def morphological_operation(tensor, operation, kernel_size, iterations=1):
    """ Helper function to perform dilation or erosion using PyTorch """
    # Create a kernel for dilation or erosion
    kernel = torch.ones((1, 1, kernel_size, kernel_size), device=tensor.device, dtype=tensor.dtype)
    padding = kernel_size // 2

    for _ in range(iterations):
        if operation == 'erosion':
            tensor = F.pad(tensor, (padding, padding, padding, padding), mode='constant',
                           value=1)  # Pad with 1s for erosion
            tensor = 1 - F.conv2d(1 - tensor, kernel, padding=0, stride=1, groups=tensor.shape[1])
        elif operation == 'dilation':
            tensor = F.conv2d(tensor, kernel, padding=padding, stride=1, groups=tensor.shape[1])

        # Ensure the tensor remains binary
        tensor = (tensor > 0).float()

    return tensor


def getBoundary(batch_masks, size=2, iterations=1):
    if torch.unique(batch_masks).numel() == 1:
        return torch.zeros_like(batch_masks), torch.zeros_like(batch_masks)

    pixels = 2 * size + 1  # Kernel size

    # Morphological operations
    eroded_masks = morphological_operation(batch_masks, 'erosion', pixels, iterations)
    dilated_masks = morphological_operation(batch_masks, 'dilation', pixels, iterations)

    # Calculate boundaries and insides
    boundaries = (dilated_masks - eroded_masks) > 0  # Ensure boundary is binary
    return boundaries.float(), eroded_masks


# For self filtering learning
def remove_high_loss_samples(model, train_loader, loss_function, mfr, cre, crt,
                             b_size=1, device='cuda', flag='train'):
    """
    Remove training samples based on their loss for batch size = 1,

    Parameters:
    - model: Trained model.
    - train_loader: DataLoader for the training dataset. shuffle=False is recommended.
    - loss_function: Loss function used during training.
    - device: Device to perform computations ('cuda' or 'cpu').
    - mfr: Maximum Filtering Rate.
    - cre: Current Convergence Rate.
    - crt: Convergence Rate Threshold.

    Returns:
    - DataLoader with the filtered dataset.
    """
    model.eval()  # Set the model to evaluation mode
    losses = []

    bar = tqdm(train_loader, desc='Computing losses', leave=False)

    # Compute losses for each sample in the dataset
    with torch.no_grad():
        for idx, data in enumerate(bar):
            imgs, target = data
            imgs, target = imgs.to(device), target.to(device)
            output = model(imgs)
            loss = loss_function(output, target)
            losses.append((loss.item(), idx))

    # Sort samples by loss (high to low)
    losses.sort(reverse=True, key=lambda x: x[0])

    # Calculate the number of samples to remove
    num_samples_to_remove = int(len(losses) * mfr * (1 - cre / crt))

    # Identify indices of samples to remove
    indices_to_remove = [idx for _, idx in losses[:num_samples_to_remove]]

    print(f'For {flag} dataset: Removing {num_samples_to_remove} samples with high loss')

    # Filter out the high-loss samples
    remaining_indices = [idx for idx in range(len(train_loader.dataset)) if idx not in indices_to_remove]

    # Create a new DataLoader with the remaining samples
    filtered_dataset = Subset(train_loader.dataset, remaining_indices)
    filtered_loader = DataLoader(filtered_dataset, batch_size=b_size, shuffle=True, pin_memory=True, num_workers=4)

    return filtered_loader, filtered_dataset


if __name__ == '__main__':
    # test getBoundary of a batch of masks, one circle and one square
    batch_masks = torch.zeros((2, 1, 17, 17))
    batch_masks[0, 0, 3:15, 3:15] = 1
    batch_masks[1, 0, 4:6, 4:6] = 1
    print(batch_masks)
    boundaries, eroded_masks = getBoundary(batch_masks, size=1, iterations=1)
    print(boundaries)
