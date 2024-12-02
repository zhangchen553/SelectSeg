import torch

# SR : Segmentation Result
# GT : Ground Truth
# tensor(True + True) = tensor(True) not work for this tool
# need transfer boolean to int
# True + True = 2


def get_accuracy(SR,GT,threshold=0.5):
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    GT = GT > 0.5
    corr = torch.sum(SR==GT)
    tensor_size = SR.size(0)*SR.size(1)*SR.size(2)*SR.size(3)
    acc = float(corr)/float(tensor_size)

    return acc


def get_sensitivity(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    GT = GT > 0.5

    # TP : True Positive
    # FN : False Negative
    TP = ((SR==1).long()+(GT==1).long())==2
    FN = ((SR==0).long()+(GT==1).long())==2

    # print((SR==1)[0]+(GT==1)[0], ((SR==1)[0]+(GT==1)[0])==2)

    SE = float(torch.sum(TP))/(float(torch.sum(TP+FN)) + 1e-6)
    
    return SE


def get_specificity(SR,GT,threshold=0.5):
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    GT = GT > 0.5

    # TN : True Negative
    # FP : False Positive
    TN = ((SR==0).long()+(GT==0).long())==2
    FP = ((SR==1).long()+(GT==0).long())==2

    SP = float(torch.sum(TN))/(float(torch.sum(TN+FP)) + 1e-6)
    
    return SP


def get_precision(SR,GT,threshold=0.5):
    SR = SR > threshold
    # GT = GT == torch.max(GT)
    GT = GT > 0.5

    # TP : True Positive
    # FP : False Positive
    TP = ((SR==1).long()+(GT==1).long())==2
    FP = ((SR==1).long()+(GT==0).long())==2

    PC = float(torch.sum(TP))/(float(torch.sum(TP+FP)) + 1e-6)

    return PC


def get_F1(SR,GT,threshold=0.5):
    # Sensitivity == Recall
    SE = get_sensitivity(SR,GT,threshold=threshold)
    PC = get_precision(SR,GT,threshold=threshold)

    F1 = 2*SE*PC/(SE+PC + 1e-6)

    return F1


def get_JS(SR,GT,threshold=0.5):
    # JS : Jaccard similarity
    SR = SR > threshold
    GT = GT > 0.5
    
    Inter = torch.sum((SR.long()+GT.long())==2)
    Union = torch.sum((SR.long()+GT.long())>=1)
    
    JS = float(Inter)/(float(Union) + 1e-6)
    
    return JS


def get_DC(SR,GT,threshold=0.5):
    # DC : Dice Coefficient
    SR = SR > threshold
    GT = GT > 0.5

    Inter = torch.sum((SR.long()+GT.long())==2)
    DC = float(2*Inter)/(float(torch.sum(SR)+torch.sum(GT)) + 1e-6)

    return DC













def compute_dice2(pred, gt):
  pred = ((pred) >= .5).float()
  dice_score = (2 * (pred * gt).sum()) / ((pred + gt).sum() + 1e-8)
  
  return dice_score


def get_IoU(outputs, labels):
  EPS = 1e-6
  outputs = (outputs> 0.5).int()
  labels = (labels > 0.5).int()   
  intersection = (outputs & labels).float().sum((1, 2))
  union = (outputs | labels).float().sum((1, 2))

  iou = (intersection + EPS) / (union + EPS)  # We smooth our devision to avoid 0/0
  return iou.mean()

def accuracy(preds, label):
    preds = (preds > 0.5).int()
    label = (label > 0.5).int()   
    valid = (label >= 0)
    acc_sum = (valid * (preds == label)).sum()
    valid_sum = valid.sum()
    acc = float(acc_sum) / (valid_sum + 1e-10)
    return acc

def precision_recall_f1(preds, label):
  epsilon = 1e-7
  y_true = (label > 0.5).int()
  y_pred = (preds > 0.5).int()
  tol_pix = (label >= 0).int()
  tp = (y_true * y_pred).sum().to(torch.float32)
  tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
  fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
  fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
  precision = tp / (tp + fp + epsilon)
  recall = tp / (tp + fn + epsilon)
  f1 = 2* (precision*recall) / (precision + recall + epsilon)
  return precision, recall, f1

def confusion_mat(preds, label):
  epsilon = 1e-7
  y_true = (label > 0.5).int()
  y_pred = (preds > 0.5).int()
  tol_pix = (label >= 0).int()
  tp = (y_true * y_pred).sum().to(torch.float32)
  tn = ((1 - y_true) * (1 - y_pred)).sum().to(torch.float32)
  fp = ((1 - y_true) * y_pred).sum().to(torch.float32)
  fn = (y_true * (1 - y_pred)).sum().to(torch.float32)
  return tp, tn, fp, fn

