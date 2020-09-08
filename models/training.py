import time
from IPython.display import clear_output
import torch
import numpy as np
import torch.nn as nn
from cloud_net_plus import CloudNetPlus
from losses import FilteredJaccardLoss, WeaklyLoss
from dataset import SwinysegDataset, Cloud95Dataset, ToTensor, Rescale, show_image_gt_batch, show_image_inference_batch, \
    show_image_gt_batch_weakly, get_dataloaders
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage import io, transform
import cv2


class Trainer:
    def __init__(self, weakly_training, **kwargs):
        # Using cuda if possible
        self.dtype = get_dtype()

        self.weakly = weakly_training
        self.use_softmax = not weakly_training

        # Model
        net_params = kwargs['network_parameters']  # in_channels and depth
        net_params['softmax'] = self.use_softmax
        self.network_parameters = net_params
        self.model = CloudNetPlus(**self.network_parameters)

        # Dataset and data loaders
        dataset_params = kwargs['dataset_parameters']  # dataset name , csv path, root path, transform, batch_size
        dataset_params['weakly'] = self.weakly
        dataset_params['use_nir'] = True if net_params['input_channels'] == 4 else False
        self.train_dl, self.valid_dl = get_dataloaders(**dataset_params)

        # Optimizer and schedule
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=kwargs['lr'])
        self.lr_scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.3, patience=3, verbose=True,
                                              min_lr=1e-8)

        # Loss function
        if self.weakly:
            self.loss_fn = WeaklyLoss(dense_crf_weight=kwargs['dense_loss_weight'], ignore_index=255)
        else:
            self.loss_fn = FilteredJaccardLoss()

        # Accuracy function
        if self.weakly:
            self.acc_fn = weakly_acc
        else:
            self.acc_fn = acc_metric

        # Extras
        if self.weakly:
            self.softmax = nn.Softmax(dim=1)

    def train(self, epochs=1):
        start = time.time()

        self.model.type(self.dtype)

        train_loss, valid_loss = [], []
        train_acc, valid_acc = [], []
        train_orig_acc, valid_orig_acc = [], []

        best_acc = 0.0

        for epoch in range(1, epochs + 1):
            print('Epoch {}/{}'.format(epoch, epochs))
            print('-' * 10)
            flag = True
            for phase in ['train', 'valid']:
                if phase == 'train':
                    self.model.train(True)  # Set training mode = true
                    dataloader = self.train_dl
                else:
                    self.model.train(False)  # Set model to evaluate mode
                    dataloader = self.valid_dl

                running_loss = 0.0
                running_acc = 0.0
                running_orig_acc = 0.0

                step = 0

                # iterate over data
                for sample_batched in dataloader:
                    x = sample_batched['image'].type(self.dtype)
                    y = sample_batched['gt'].type(self.dtype)
                    y_orig = sample_batched['orig_gt'].type(self.dtype) if self.weakly else None
                    step += 1

                    # forward pass
                    if phase == 'train':
                        # zero the gradients
                        self.optimizer.zero_grad()
                        outputs = self.model(x)

                        args = [outputs, y]
                        if self.weakly:
                            args.append(x)
                        loss = self.loss_fn(*args)

                        # the backward pass frees the graph memory, so there is no
                        # need for torch.no_grad in this training pass
                        loss.backward()
                        self.optimizer.step()
                        # scheduler.step()

                    else:
                        with torch.no_grad():
                            outputs = self.model(x)
                            args = [outputs, y.long()]
                            if self.weakly:
                                args.append(x)
                            loss = self.loss_fn(*args)

                    # stats - whatever is the phase
                    acc = self.acc_fn(outputs, y)
                    orig_acc = 0.0
                    if self.weakly:
                        orig_acc = acc_metric(self.softmax(outputs), y_orig)
                        running_orig_acc += orig_acc * dataloader.batch_size

                    running_acc += acc * dataloader.batch_size
                    running_loss += loss.item() * dataloader.batch_size

                    if epoch % 10 == 0 and flag:
                        with torch.no_grad():
                            outputs = self.model(x)
                            if self.weakly:
                                outputs = self.softmax(outputs)
                                show_image_gt_batch_weakly(x.cpu(), y.cpu(), y_orig.cpu(), outputs.cpu())
                            else:
                                show_image_gt_batch(x.cpu(), y.cpu(), outputs.cpu())
                            flag = False

                    if step % 100 == 0:
                        # clear_output(wait=True)
                        print(f'Current step: {step}  Loss: {loss.item()}  Acc: {acc} '
                              f'Orig acc: {orig_acc} AllocMem (Mb): '
                              f'{torch.cuda.memory_allocated() / 1024 / 1024}')

                        # print(torch.cuda.memory_summary())

                epoch_loss = running_loss / len(dataloader.dataset)
                epoch_acc = running_acc / len(dataloader.dataset)
                epoch_orig_acc = running_orig_acc / len(dataloader.dataset)

                clear_output(wait=True)
                print('Epoch {}/{}'.format(epoch, epochs))
                print('-' * 10)
                print('{} Loss: {:.4f} Acc: {}, Orig acc: {}'.format(phase, epoch_loss, epoch_acc, epoch_orig_acc))
                print('-' * 10)

                train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)
                train_acc.append(epoch_acc) if phase == 'train' else valid_acc.append(epoch_acc)
                train_orig_acc.append(epoch_orig_acc) if phase == 'train' else valid_orig_acc.append(epoch_orig_acc)

                if phase == 'valid':
                    self.lr_scheduler.step(epoch_loss)

        time_elapsed = time.time() - start
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        saved_state = dict(model_state=self.model.state_dict())
        torch.save(saved_state, 'saved_state')
        print(f'*** Saved checkpoint ***')
        # print(f'Finding best threshold:')
        # find_best_threshold(model, valid_dl)
        return train_loss, valid_loss, train_acc, valid_acc, train_orig_acc, valid_orig_acc


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


def test_acc(model: torch.nn.Module, test_dl: DataLoader, threshold=0.5, use_softmax=False):
    is_cuda = next(model.parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    softmax = nn.Softmax(dim=1)

    model.train(False)
    running_acc = 0.0

    for sample_batched in test_dl:
        x = sample_batched['image'].type(dtype)
        y = sample_batched['gt'].type(dtype)

        with torch.no_grad():
            outputs = model(x)
            if use_softmax:
                outputs = softmax(outputs)
        acc = acc_metric(outputs, y, threshold)

        running_acc += acc * test_dl.batch_size

    total_acc = running_acc / len(test_dl.dataset)
    return total_acc


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


def train_network(weakly=False):
    network_parameters = {'inception_depth': 6,
                          'input_channels': 3}
    dataset_parameters = {'name': 'swinyseg',
                          'csv_file': '../data/swinyseg/metadata_train.csv',
                          'root_dir': '../data/swinyseg/',
                          'transform': transforms.Compose([Rescale(192), ToTensor()]),
                          'batch_size': 16}
    dense_loss_weight = 3e-10
    lr = 1e-3
    kwarg = {'lr': lr,
             'dense_loss_weight': dense_loss_weight,
             'network_parameters': network_parameters,
             'dataset_parameters': dataset_parameters}

    trainer = Trainer(weakly_training=weakly, **kwarg)

    train_loss, valid_loss, train_acc, valid_acc, train_orig_acc, valid_orig_acc = trainer.train(epochs=50)

    get_model_accuracy_swinyseg(trainer.model, use_softmax=weakly)

    plot_graph(train_loss, valid_loss, 'loss', f'../plots/losses with denseloss {dense_loss_weight}.png')
    plot_graph(train_acc, valid_acc, 'accuracy', f'../plots/ accuracy with denseloss {dense_loss_weight}.png')
    plot_graph(train_orig_acc, valid_orig_acc, 'orig accuracy',
               f'../plots/orig accuracy with denseloss {dense_loss_weight}.png')


def get_dtype():
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev)
    if dev is 'cuda':
        dtype = torch.cuda.FloatTensor
    else:
        dtype = torch.FloatTensor
    # print(f'Using device {device}')
    return dtype


def plot_graph(train_data, valid_data, data_type, destination):
    plt.figure(figsize=(10, 8))
    plt.plot(train_data, label=f'Train {data_type}')
    plt.plot(valid_data, label=f'Valid {data_type}')
    plt.legend()
    plt.savefig(destination)
    plt.show()


def evaluate_performance():
    # Calculate TP TN FP FN for all images in test set
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0


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


def get_model_accuracy_swinyseg(model=None, use_softmax=True):
    dataset = SwinysegDataset('../data/swinyseg/metadata_test.csv', '../data/swinyseg/',
                              transform=transforms.Compose([Rescale(192), ToTensor()]), train=False)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    if model is None:
        model = CloudNetPlus(3, 6, residual=True, softmax=False)
        model_saved_state = 'saved_state_weakly_denseloss_1e-9_88_96%'
        saved_state = torch.load(model_saved_state, map_location='cpu')
        model.load_state_dict(saved_state['model_state'])
    model.type(torch.cuda.FloatTensor)
    acc = test_acc(model, dl, use_softmax=use_softmax)
    print(f'Test accuracy is {acc * 100}%')


if __name__ == "__main__":
    # show_inference(num_imgs=4, gt=True, print_patches=True)
    train_network(False)
    '''model = CloudNetPlus(3, 6, residual=True, softmax=True)
    saved_state = torch.load('saved_state_95-cloud-3d', map_location='cpu')
    model.load_state_dict(saved_state['model_state'])
    model.type(torch.cuda.FloatTensor)

    x = batch.type(torch.cuda.FloatTensor)

    with torch.no_grad():
        output = model(x)

    show_image_inference_batch(batch, output.cpu())'''
