import torch
import torch.nn as nn


class FilteredJaccardLoss(nn.Module):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epsilon=1e-8):
        """
        Calculate filtered Jaccard loss
        :param y_pred: tensor shaped (batch_size x H x W ) with probability for each pixel
        :param y_true: tensor shaped (batch_size x H x W) of true class for each pixel
        :param epsilon: const for stability
        :return:
        """

        if y_true.sum() == 0:
            i = ((1 - y_true) * (1 - y_pred)).sum().float()
            u = ((1 - y_true) + (1 - y_pred)).sum().float()
            loss = 1. - (i / (u - i + epsilon))
        else:
            i = (y_true * y_pred).sum().float()
            u = (y_true + y_pred).sum().float()
            loss = 1. - (i / (u - i + epsilon))

        return loss
