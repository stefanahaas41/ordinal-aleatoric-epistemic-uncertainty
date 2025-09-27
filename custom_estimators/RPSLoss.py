import torch
from torch import nn
import torch.nn.functional as F


class RPSLoss(nn.Module):

    def __init__(self):
        super().__init__()


    def forward(self,y_pred,y_true):

        y_true_one_hot = F.one_hot(y_true,num_classes=y_pred.shape[1])

        #y_pred = F.softmax(y_pred,dim=1)

        pred_cdf = torch.cumsum(y_pred,dim=1)
        true_cdf = torch.cumsum(y_true_one_hot, dim=1)

        rps = torch.sum((pred_cdf - true_cdf)**2, dim=1)

        return rps.mean()
