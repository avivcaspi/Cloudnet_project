import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import os


class Cloud95Dataset(Dataset):
    """95 - Cloud dataset."""

    def __init__(self, csv_file, root_dir, transform=None, train=True, use_nir=True):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.patches_name = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.nir = use_nir

    def __len__(self):
        return len(self.patches_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        train_test_str = 'train' if self.train else 'test'
        red_str = f'{train_test_str}_red/red_'
        blue_str = f'{train_test_str}_blue/blue_'
        green_str = f'{train_test_str}_green/green_'
        nir_str = f'{train_test_str}_nir/nir_'
        gt_str = f'{train_test_str}_gt/gt_'
        gt_img = None
        red_img_name = os.path.join(self.root_dir,
                                    red_str + self.patches_name.iloc[idx, 0] + '.TIF')
        blue_img_name = os.path.join(self.root_dir,
                                     blue_str + self.patches_name.iloc[idx, 0] + '.TIF')
        green_img_name = os.path.join(self.root_dir,
                                      green_str + self.patches_name.iloc[idx, 0] + '.TIF')

        if self.train:
            gt_img_name = os.path.join(self.root_dir,
                                       gt_str + self.patches_name.iloc[idx, 0] + '.TIF')
            gt_img = io.imread(gt_img_name) / 255

        red_img = np.expand_dims(io.imread(red_img_name) / 65535, axis=-1)
        green_img = np.expand_dims(io.imread(green_img_name) / 65535, axis=-1)
        blue_img = np.expand_dims(io.imread(blue_img_name) / 65535, axis=-1)

        image = np.concatenate([red_img, green_img, blue_img], axis=-1)
        if self.nir:
            nir_img_name = os.path.join(self.root_dir,
                                        nir_str + self.patches_name.iloc[idx, 0] + '.TIF')
            nir_img = np.expand_dims(io.imread(nir_img_name) / 65535, axis=-1)
            image = np.concatenate([image, nir_img], axis=-1)

        sample = {'image': image, 'gt': gt_img, 'patch_name': self.patches_name.iloc[idx, 0]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SwinysegDataset(Dataset):
    """swinyseg dataset."""

    def __init__(self, csv_file, root_dir, transform=None, weakly=False):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        original_patches_name = pd.read_csv(csv_file)
        patches_name = original_patches_name.copy()
        for i in range(1, 6):
            extra_patches = original_patches_name.copy()
            extra_patches['Name'] = original_patches_name['Name'].str.replace('.jpg', f'_{i}.jpg')
            patches_name = pd.concat([patches_name, extra_patches])
        self.patches_name = patches_name
        self.root_dir = root_dir
        self.transform = transform
        self.weakly = weakly

    def __len__(self):
        return len(self.patches_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_folder = 'images/'
        gt_folder = 'GTmaps/'
        full_folder = 'WeaklyFull/'
        if self.weakly:
            gt_folder = 'WeaklyGT/'
        img_name = os.path.join(self.root_dir,
                                img_folder + self.patches_name.iloc[idx, 0])
        gt_map_name = os.path.join(self.root_dir,
                                   gt_folder + self.patches_name.iloc[idx, 0].replace('jpg', 'png'))
        weakly_full = None
        if self.weakly:
            full_map_name = os.path.join(self.root_dir,
                                         full_folder + self.patches_name.iloc[idx, 0].replace('jpg', 'png'))
            weakly_full = io.imread(full_map_name)
        image = io.imread(img_name) / 255
        gt_img = io.imread(gt_map_name)
        if not self.weakly:
            gt_img /= 255

        sample = {'image': image, 'gt': gt_img, 'full_weakly': weakly_full, 'patch_name': self.patches_name.iloc[idx, 0]}

        if self.transform:
            sample = self.transform(sample)

        return sample


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))

        if gt is not None:
            gt = cv2.resize(gt, dsize=(new_h, new_w), interpolation=cv2.INTER_NEAREST)

        sample['image'] = img
        sample['gt'] = gt
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        sample['image'] = torch.from_numpy(image)
        sample['gt'] = torch.from_numpy(gt) if gt is not None else None

        return sample


def show_image_gt_batch(image, gt, pred=None):
    """Show image with gt"""
    batch_size = image.shape[0]
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((0, 2, 3, 1))
    if pred is not None and len(pred.shape) == 4:
        pred = pred[:, 1, :, :]

    fig, ax = plt.subplots(batch_size, 3, figsize=(20, batch_size * 5))
    if batch_size == 1:
        ax[0].imshow(image[0, :, :, :3])
        ax[0].set_title('Image')
        ax[1].imshow(gt[0], cmap='gray')
        ax[1].set_title('gt')
        if pred is not None:
            ax[2].imshow(pred[0], cmap='gray')
            ax[2].set_title('Pred')
    else:
        ax[0, 0].set_title('Image')
        ax[0, 1].set_title('gt')
        if pred is not None:
            ax[0, 2].set_title('Pred')
        for i in range(batch_size):
            ax[i, 0].imshow(image[i, :, :, :3])
            ax[i, 1].imshow(gt[i], cmap='gray')
            if pred is not None:
                ax[i, 2].imshow(pred[i], cmap='gray')

    plt.show()


def show_image_inference_batch(image, pred, gt=None):
    """Show image with gt"""
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((0, 2, 3, 1))
    if len(pred.shape) == 4:
        pred = pred[:, 1, :, :]

    batch_size = image.shape[0]
    num_cols = 2 if gt is None else 3
    fig, axes = plt.subplots(batch_size, num_cols, figsize=(12, batch_size * 5))

    if batch_size == 1:
        axes = axes.reshape([1, num_cols])

    titles = ['Image', 'Pred', 'GT']
    for i in range(num_cols):
        axes[0, i].set_title(titles[i])

    for i in range(batch_size):
        axes[i, 0].imshow(image[i, :, :, :3])
        axes[i, 1].imshow(pred[i], cmap='gray')
        if gt is not None:
            axes[i, 2].imshow(gt[i], cmap='gray')

    plt.setp(axes, xticks=[], yticks=[])
    plt.show()


def gt_to_onehot(gt_image):
    onehot = torch.zeros((gt_image.shape[0], 2, *gt_image.shape[1:]))
    onehot[:, 1, :, :] = gt_image
    onehot[:, 0, :, :] = 1 - gt_image
    return onehot


if __name__ == '__main__':
    cloud95_dataset = SwinysegDataset(
        csv_file='../data/swinyseg/metadata.csv',
        root_dir='../data/swinyseg/',
        transform=ToTensor(), weakly=True
    )

    x = cloud95_dataset[1]
    dataloader = DataLoader(cloud95_dataset, batch_size=3,
                            shuffle=False)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size(),
              sample_batched['gt'].size())
        print(gt_to_onehot(sample_batched['gt']).size())
        show_image_gt_batch(sample_batched['image'], sample_batched['gt'])
