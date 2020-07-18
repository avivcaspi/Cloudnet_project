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

    def __init__(self, csv_file, root_dir, transform=None):
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

    def __len__(self):
        return len(self.patches_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        red_str = 'train_red/red_'
        blue_str = 'train_blue/blue_'
        green_str = 'train_green/green_'
        nir_str = 'train_nir/nir_'
        gt_str = 'train_gt/gt_'
        red_img_name = os.path.join(self.root_dir,
                                    red_str + self.patches_name.iloc[idx, 0] + '.TIF')
        blue_img_name = os.path.join(self.root_dir,
                                     blue_str + self.patches_name.iloc[idx, 0] + '.TIF')
        green_img_name = os.path.join(self.root_dir,
                                      green_str + self.patches_name.iloc[idx, 0] + '.TIF')
        nir_img_name = os.path.join(self.root_dir,
                                    nir_str + self.patches_name.iloc[idx, 0] + '.TIF')
        gt_img_name = os.path.join(self.root_dir,
                                   gt_str + self.patches_name.iloc[idx, 0] + '.TIF')
        red_img = np.expand_dims(io.imread(red_img_name) / 65535, axis=-1)
        green_img = np.expand_dims(io.imread(green_img_name) / 65535, axis=-1)
        blue_img = np.expand_dims(io.imread(blue_img_name) / 65535, axis=-1)
        nir_img = np.expand_dims(io.imread(nir_img_name) / 65535, axis=-1)
        gt_img = io.imread(gt_img_name) / 255

        image = np.concatenate([red_img, green_img, blue_img, nir_img], axis=-1)

        sample = {'image': image, 'gt': gt_img}

        if self.transform:
            sample = self.transform(sample)

        return sample


class SwinysegDataset(Dataset):
    """swinyseg dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
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

    def __len__(self):
        return len(self.patches_name)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_folder = 'images/'
        gt_folder = 'GTmaps/'
        img_name = os.path.join(self.root_dir,
                                    img_folder + self.patches_name.iloc[idx, 0])
        gt_map_name = os.path.join(self.root_dir,
                                   gt_folder + self.patches_name.iloc[idx, 0].replace('jpg', 'png'))
        image = io.imread(img_name) / 255
        gt_img = io.imread(gt_map_name) / 255

        sample = {'image': image, 'gt': gt_img}

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

        gt = transform.resize(gt, (new_h, new_w))

        return {'image': img, 'gt': gt}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, gt = sample['image'], sample['gt']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'gt': torch.from_numpy(gt)}


def show_image_gt_batch(image, gt, pred=None):
    """Show image with gt"""
    batch_size = image.shape[0]
    if isinstance(image, torch.Tensor):
        image = image.numpy().transpose((0, 2, 3, 1))

    fig, ax = plt.subplots(batch_size, 3, figsize=(20, batch_size * 5))
    if batch_size == 1:
        ax[0].imshow(image[0, :, :, :3])
        ax[1].imshow(gt[0], cmap='gray')
        if pred is not None:
            ax[2].imshow(pred[0], cmap='gray')
    else:
        for i in range(batch_size):
            ax[i, 0].imshow(image[i, :, :, :3])
            ax[i, 1].imshow(gt[i], cmap='gray')
            if pred is not None:
                ax[i, 2].imshow(pred[i], cmap='gray')
    plt.show()


if __name__ == '__main__':
    cloud95_dataset = SwinysegDataset(
        csv_file='../data/swinyseg/metadata.csv',
        root_dir='../data/swinyseg/',
        transform=ToTensor()
        )

    x = cloud95_dataset[1]
    dataloader = DataLoader(cloud95_dataset, batch_size=3,
                            shuffle=False)

    for i_batch, sample_batched in enumerate(dataloader):

        print(i_batch, sample_batched['image'].size(),
              sample_batched['gt'].size())
        show_image_gt_batch(sample_batched['image'], sample_batched['gt'])




