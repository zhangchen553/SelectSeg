import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


# Weight for consistency loss of mean teacher
def sigmoid_rampup(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


class SCELoss(torch.nn.Module):
    def __init__(self, alpha=2.3, beta=0.1, num_classes=10):  # 2.3, 0.1
        super(SCELoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.cross_entropy = torch.nn.BCEWithLogitsLoss()

    def forward(self, pred, labels):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = labels
        # to avoid log(0), A=log(1e-4) = -4,
        label_one_hot = torch.clamp(label_one_hot, min=1e-6, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        return loss


class SCLLoss(torch.nn.Module):
    def __init__(self):  # 2.3, 0.1
        super(SCLLoss, self).__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.cross_entropy = torch.nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, pred, labels):
        weights = torch.sigmoid(pred)
        weights[weights > 0.8] = 1.0
        weights = torch.clamp(weights, min=1e-10, max=1.0).detach()

        # CCE
        ce = self.cross_entropy(pred, labels)
        ce = torch.mean(ce * (1-weights))

        # RCE
        pred = torch.sigmoid(pred)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = labels
        # to avoid log(0), A=log(1e-4) = -4,
        label_one_hot = torch.clamp(label_one_hot, min=1e-6, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))
        rce = torch.mean(rce * weights)

        # Loss
        loss = ce + rce
        return loss

def dice_coef(SR, GT, threshold=0.5):
    # DC : Dice Coefficient
    SR = torch.sigmoid(SR)
    SR = SR > threshold
    GT = GT == torch.max(GT)

    inter = torch.sum((SR.long()+GT.long()) == 2, dim=(-1, -2))
    DC = 2*inter/(torch.sum(SR, dim=(-1, -2))+torch.sum(GT, dim=(-1, -2)) + 1e-6)
    DC = torch.sum(DC)/len(DC)

    return DC


def dice_coef_loss(y_pred, y_true):
    return 1.0-dice_coef(y_pred, y_true)


class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice_loss = 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return dice_loss


class ConLoss(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(ConLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.bmm(feat_q.reshape(-1, 1, dim), feat_k.reshape(-1, dim, 1))
        l_pos = l_pos.view(-1, 1)

        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


class contrastive_loss_sup(torch.nn.Module):
    def __init__(self, temperature=0.07, base_temperature=0.07):
        """
        Contrastive Learning for Unpaired Image-to-Image Translation
        models/patchnce.py
        """
        super(contrastive_loss_sup, self).__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.nce_includes_all_negatives_from_minibatch = False
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mask_dtype = torch.bool

    def forward(self, feat_q, feat_k):
        assert feat_q.size() == feat_k.size(), (feat_q.size(), feat_k.size())
        batch_size = feat_q.shape[0]
        dim = feat_q.shape[1]
        width = feat_q.shape[2]
        feat_q = feat_q.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_k = feat_k.view(batch_size, dim, -1).permute(0, 2, 1)
        feat_q = F.normalize(feat_q, dim=-1, p=1)
        feat_k = F.normalize(feat_k, dim=-1, p=1)
        feat_k = feat_k.detach()

        # pos logit
        l_pos = torch.zeros((batch_size * 4096, 1)).cuda()
        # neg logit
        if self.nce_includes_all_negatives_from_minibatch:
            # reshape features as if they are all negatives of minibatch of size 1.
            batch_dim_for_bmm = 1
        else:
            batch_dim_for_bmm = batch_size

        # reshape features to batch size
        feat_q = feat_q.reshape(batch_dim_for_bmm, -1, dim)
        feat_k = feat_k.reshape(batch_dim_for_bmm, -1, dim)
        npatches = feat_q.size(1)
        l_neg_curbatch = torch.bmm(feat_q, feat_k.transpose(2, 1))

        diagonal = torch.eye(npatches, device=feat_q.device, dtype=self.mask_dtype)[None, :, :]

        l_neg_curbatch.masked_fill_(diagonal, -10.0)
        l_neg = l_neg_curbatch.view(-1, npatches)

        out = torch.cat((l_pos, l_neg), dim=1) / self.temperature

        loss = self.cross_entropy_loss(out, torch.zeros(out.size(0), dtype=torch.long,
                                                        device=feat_q.device))

        return loss


# https://github.com/Digital-Dermatology/t-loss/blob/main/tloss.py
# Modified the configuration to match the one in the config.py
class TLoss(nn.Module):
    def __init__(
            self,
            config,
            nu: float = 1.0,
            epsilon: float = 1e-8,
            reduction: str = "mean",
    ):
        """
        Implementation of the TLoss.

        Args:
            config: Configuration object for the loss.
            nu (float): Value of nu.
            epsilon (float): Value of epsilon.
            reduction (str): Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'.
                             'none': no reduction will be applied,
                             'mean': the sum of the output will be divided by the number of elements in the output,
                             'sum': the output will be summed.
        """
        super().__init__()
        self.config = config
        self.D = torch.tensor(
            (self.config.image_size * self.config.image_size),
            dtype=torch.float,
            device=config.device,
        )

        self.lambdas = torch.ones(
            (self.config.image_size, self.config.image_size),
            dtype=torch.float,
            device=config.device,
        )
        self.nu = nn.Parameter(
            torch.tensor(nu, dtype=torch.float, device=config.device)
        )
        self.epsilon = torch.tensor(epsilon, dtype=torch.float, device=config.device)
        self.reduction = reduction

    def forward(
            self, input_tensor: torch.Tensor, target_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            input_tensor (torch.Tensor): Model's prediction, size (B x W x H).
            target_tensor (torch.Tensor): Ground truth, size (B x W x H).

        Returns:
            torch.Tensor: Total loss value.
        """

        delta_i = input_tensor - target_tensor
        sum_nu_epsilon = torch.exp(self.nu) + self.epsilon
        first_term = -torch.lgamma((sum_nu_epsilon + self.D) / 2)
        second_term = torch.lgamma(sum_nu_epsilon / 2)
        third_term = -0.5 * torch.sum(self.lambdas + self.epsilon)
        fourth_term = (self.D / 2) * torch.log(torch.tensor(np.pi))
        fifth_term = (self.D / 2) * (self.nu + self.epsilon)

        delta_squared = torch.pow(delta_i, 2)
        lambdas_exp = torch.exp(self.lambdas + self.epsilon)
        numerator = delta_squared * lambdas_exp
        numerator = torch.sum(numerator, dim=(1, 2))

        fraction = numerator / sum_nu_epsilon
        sixth_term = ((sum_nu_epsilon + self.D) / 2) * torch.log(1 + fraction)

        total_losses = (
                first_term
                + second_term
                + third_term
                + fourth_term
                + fifth_term
                + sixth_term
        )

        if self.reduction == "mean":
            return total_losses.mean()
        elif self.reduction == "sum":
            return total_losses.sum()
        elif self.reduction == "none":
            return total_losses
        else:
            raise ValueError(
                f"The reduction method '{self.reduction}' is not implemented."
            )

if __name__ == '__main__':
    loss = contrastive_loss_sup()
