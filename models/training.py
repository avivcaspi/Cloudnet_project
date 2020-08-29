import time
from IPython.display import clear_output
import torch
import numpy as np
import torch.nn as nn
from cloud_net_plus import CloudNetPlus
from losses import FilteredJaccardLoss, WeaklyLoss
from dataset import SwinysegDataset, Cloud95Dataset, ToTensor, Rescale, show_image_gt_batch, show_image_inference_batch
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage import io, transform
import cv2


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1, weakly=False):
    start = time.time()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    if dev is 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    print(f'Using device {device}')
    model.type(dtype)

    train_loss, valid_loss = [], []
    train_acc, valid_acc = [], []

    best_acc = 0.0
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.3, patience=3, verbose=True, min_lr=1e-8)

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)
        flag = True
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set training mode = true
                dataloader = train_dl
            else:
                model.train(False)  # Set model to evaluate mode
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0

            # iterate over data
            for sample_batched in dataloader:
                x = sample_batched['image'].type(dtype)
                y = sample_batched['gt'].type(dtype)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()
                    outputs = model(x)

                    args = [outputs, y]
                    if weakly:
                        args.append(x)
                    loss = loss_fn(*args)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        args = [outputs, y.long()]
                        if weakly:
                            args.append(x)
                        loss = loss_fn(*args)

                # stats - whatever is the phase
                acc = acc_fn(outputs, y)

                running_acc += acc * dataloader.batch_size
                running_loss += loss.item() * dataloader.batch_size

                if epoch % 10 == 0 and flag:
                    with torch.no_grad():
                        outputs = model(x)
                        if weakly:
                            softmax = nn.Softmax(dim=1)
                            outputs = softmax(outputs)
                        # TODO show weakly gt in different colors, and add full gt for reference
                        show_image_gt_batch(x.cpu(), y.cpu(), outputs.cpu())
                        flag = False

                if step % 100 == 0:
                    # TODO add accuracy to the real gt
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss.item(), acc,
                                                                                          torch.cuda.memory_allocated()
                                                                                          / 1024 / 1024))

                    # print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
            train_acc.append(epoch_acc) if phase == 'train' else valid_acc.append(epoch_acc)

            if phase == 'valid':
                lr_scheduler.step(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    saved_state = dict(model_state=model.state_dict())
    torch.save(saved_state, 'saved_state')
    print(f'*** Saved checkpoint ***')
    print(f'Finding best threshold:')
    # find_best_threshold(model, valid_dl)
    return train_loss, valid_loss, train_acc, valid_acc


def acc_metric(pred, y, threshold=0.5):
    dtype = pred.dtype
    if len(pred.shape) == 4:
        mask = torch.argmax(pred, 1)
    else:
        mask = pred.clone().detach()
        mask[pred >= threshold] = 1
        mask[pred < threshold] = 0
    return (mask == y.type(dtype)).float().mean()


def weakly_acc(pred, y):
    dtype = pred.dtype
    mask = torch.argmax(pred, 1)
    relevant = (y != 255).float()
    return ((mask == y.type(dtype)) * relevant).float().sum() / relevant.sum()


def validation_acc(model: torch.nn.Module, valid_dl: DataLoader, threshold=0.5):
    is_cuda = next(model.parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor

    model.train(False)
    running_acc = 0.0

    for sample_batched in valid_dl:
        x = sample_batched['image'].type(dtype)
        y = sample_batched['gt'].type(dtype)

        with torch.no_grad():
            outputs = model(x)
        acc = acc_metric(outputs, y, threshold)

        running_acc += acc * valid_dl.batch_size

    total_acc = running_acc / len(valid_dl.dataset)
    return total_acc


def find_best_threshold(model: torch.nn.Module, valid_dl: DataLoader):
    thresholds = list(np.linspace(0.2, 0.8, 7))
    best_acc = 0.0
    best_t = 0.0
    for t in thresholds:
        acc = validation_acc(model, valid_dl, t)
        if best_acc < acc:
            best_acc = acc
            best_t = t
        print(f'Threshold = {t},  accuracy = {acc}')
    return best_t


def inference(model: nn.Module, images: torch.Tensor, saved_state=None, gt=None):
    if saved_state is not None:
        model.load_state_dict(saved_state['model_state'])

    if images.ndim == 3:
        images = images.unsqueeze(0)
    if gt is not None and gt.ndim == 3:
        gt = gt.cpu()

    model.eval()
    with torch.no_grad():
        output = model(images)

    show_image_inference_batch(images.cpu(), output.cpu(), gt=gt)


def train_network():
    # Training phase
    cloud_net = CloudNetPlus(3, 6, residual=True, softmax=True)
    print(cloud_net)
    num_params = sum(p.numel() for p in cloud_net.parameters())
    print(f'# of parameters: {num_params}')

    dataset = Cloud95Dataset(csv_file='../data/95-cloud_train/training_patches_95-cloud_nonempty.csv',
                             root_dir='../data/95-cloud_train/',
                             transform=transforms.Compose([Rescale(192), ToTensor()]),
                             use_nir=False)
    length = len(dataset)
    train_size = int(0.85 * length)
    print(f'train set size is : {train_size}')
    print(f'validation set size is : {length - train_size}')

    train_ds, valid_ds = torch.utils.data.random_split(dataset, (train_size, length - train_size))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=True, num_workers=2)

    loss_fn = FilteredJaccardLoss()
    opt = torch.optim.Adam(cloud_net.parameters(), lr=1e-3)

    train_loss, valid_loss, train_acc, valid_acc = train(cloud_net, train_dl, valid_dl, loss_fn, opt, acc_metric,
                                                         epochs=50)
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(valid_acc, label='Valid accuracy')
    plt.legend()
    plt.show()


def train_network_weakly():
    # Training phase
    cloud_net = CloudNetPlus(3, 6, residual=True, softmax=False, sigmoid=False)
    print(cloud_net)
    num_params = sum(p.numel() for p in cloud_net.parameters())
    print(f'# of parameters: {num_params}')

    dataset = SwinysegDataset(csv_file='../data/swinyseg/metadata.csv',
                              root_dir='../data/swinyseg/',
                              transform=transforms.Compose([Rescale(192), ToTensor()]),
                              weakly=True)
    length = len(dataset)
    train_size = int(0.85 * length)
    print(f'train set size is : {train_size}')
    print(f'validation set size is : {length - train_size}')

    train_ds, valid_ds = torch.utils.data.random_split(dataset, (train_size, length - train_size))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=4)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=True, num_workers=2)

    loss_fn = WeaklyLoss(dense_crf_weight=1e-9, ignore_index=255)
    opt = torch.optim.Adam(cloud_net.parameters(), lr=1e-4)

    train_loss, valid_loss, train_acc, valid_acc = train(cloud_net, train_dl, valid_dl, loss_fn, opt, weakly_acc,
                                                         epochs=80, weakly=True)
    plt.figure(figsize=(10, 8))
    plt.plot(train_loss, label='Train loss')
    plt.plot(valid_loss, label='Valid loss')
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 8))
    plt.plot(train_acc, label='Train accuracy')
    plt.plot(valid_acc, label='Valid accuracy')
    plt.legend()
    plt.show()


def get_dtype():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    if dev is 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # print(f'Using device {device}')
    return dtype


def show_inference(num_imgs=4, gt=False, print_patches=False):
    dtype = get_dtype()

    model = CloudNetPlus(3, 6, residual=True, softmax=True).type(dtype)
    saved_state = torch.load('saved_state_swinyseg_new', map_location='cpu')

    dataset = SwinysegDataset(csv_file='../data/swinyseg/metadata.csv',
                              root_dir='../data/swinyseg/',
                              transform=transforms.Compose([Rescale(256), ToTensor()]))
    dl = DataLoader(dataset, batch_size=num_imgs, shuffle=True, num_workers=4)

    batch = next(iter(dl))
    imgs = batch['image'].type(dtype)
    gt_images = None
    if gt:
        gt_images = batch['gt'].type(dtype)

    if print_patches:
        print(batch['patch_name'])

    inference(model, imgs, saved_state=saved_state, gt=gt_images)


def from_video():
    dtype = get_dtype()
    model = CloudNetPlus(3, 6, residual=True, softmax=True).type(dtype)
    saved_state = torch.load('saved_state_swinyseg_new', map_location='cpu')
    model.load_state_dict(saved_state['model_state'])
    model.type(dtype).eval()

    cap = cv2.VideoCapture(0)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        frame = frame / 255
        frame_scaled = frame[:256, :256]
        frame_T = frame_scaled.transpose((2, 0, 1))
        frame_tensor = torch.from_numpy(frame_T).unsqueeze(0).type(dtype)
        with torch.no_grad():
            output = model(frame_tensor)[0, 0, :, :].unsqueeze(0)
            output = cv2.cvtColor(output.cpu().numpy().transpose(1, 2, 0), cv2.COLOR_GRAY2RGB)

        # Display the resulting frame
        cv2.imshow('frame', np.concatenate((output, frame_scaled), axis=1))

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # show_inference(num_imgs=4, gt=True, print_patches=True)
    train_network_weakly()
    '''model = CloudNetPlus(3, 6, residual=True, softmax=True)
    saved_state = torch.load('saved_state_95-cloud-3d', map_location='cpu')
    model.load_state_dict(saved_state['model_state'])
    model.type(torch.cuda.FloatTensor)

    x = batch.type(torch.cuda.FloatTensor)

    with torch.no_grad():
        output = model(x)

    show_image_inference_batch(batch, output.cpu())'''
