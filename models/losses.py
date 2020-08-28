import torch
import torch.nn as nn

from DenseCRFLoss import DenseCRFLoss, denormalizeimage


class FilteredJaccardLoss(nn.Module):

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, epsilon=1e-8):
        """
        Calculate filtered Jaccard loss
        :param y_pred: tensor shaped (batch_size x H x W) with probability for each pixel
        :param y_true: tensor shaped (batch_size x H x W) of true class for each pixel
        :param epsilon: const for stability
        :return:
        """

        if len(y_pred.shape) == 4:
            y_pred = y_pred[:, 1, :, :]
        if y_true.sum() == 0:
            i = ((1 - y_true) * (1 - y_pred)).sum().float()
            u = ((1 - y_true) + (1 - y_pred)).sum().float()
            loss = 1. - (i / (u - i + epsilon))
        else:
            i = (y_true * y_pred).sum().float()
            u = (y_true + y_pred).sum().float()
            loss = 1. - (i / (u - i + epsilon))

        return loss


class WeaklyLoss(nn.Module):
    def __init__(self, dense_crf_weight, ignore_index=255):
        super().__init__()
        self.ignore_index = ignore_index
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.jaccard_loss = FilteredJaccardLoss()
        self.dense_crf_loss = DenseCRFLoss(weight=dense_crf_weight, sigma_rgb=15.0, sigma_xy=80.0, scale_factor=1.0)
        self.softmax = nn.Softmax(dim=1)
        self.batch_average = True

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()

        if isinstance(target, torch.cuda.FloatTensor):
            self.ce_loss = self.ce_loss.cuda()

        loss = self.ce_loss(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor, x: torch.Tensor):
        # the loss is celoss + regularization
        loss = self.CrossEntropyLoss(y_pred, y_true)

        denormalized_image = denormalizeimage(x, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        probs = self.softmax(y_pred)
        croppings = (y_true != 254).float()

        regularization = self.dense_crf_loss(denormalized_image, probs, croppings)
        if isinstance(y_pred, torch.cuda.FloatTensor):
            regularization = regularization.cuda()

        return loss + regularization
