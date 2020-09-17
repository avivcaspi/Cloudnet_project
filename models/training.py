import time
from IPython.display import clear_output
import torch
import numpy as np
import torch.nn as nn
from cloud_net_plus import CloudNetPlus
from losses import FilteredJaccardLoss, WeaklyLoss, CrossEntropyLoss
from dataset import SwinysegDataset, Cloud95Dataset, ToTensor, Rescale, show_image_gt_batch, show_image_inference_batch, \
    show_image_gt_batch_weakly, get_dataloaders
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
from skimage import io, transform
from sklearn.metrics import confusion_matrix
import cv2


class Trainer:
    def __init__(self, weakly_training, **kwargs):
        # Using cuda if possible
        self.dtype = get_dtype()

        self.weakly = weakly_training

        # Model
        net_params = kwargs['network_parameters']  # in_channels and depth
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
            if kwargs['loss_fn'] == 'ce':
                self.loss_fn = CrossEntropyLoss()
            else:
                self.loss_fn = FilteredJaccardLoss()

        # Accuracy function
        if self.weakly:
            self.acc_fn = weakly_acc
        else:
            self.acc_fn = acc_metric

        # Extras
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
                    outputs = self.softmax(outputs)
                    acc = self.acc_fn(outputs, y)
                    orig_acc = 0.0
                    if self.weakly:
                        orig_acc = acc_metric(outputs, y_orig)
                        running_orig_acc += orig_acc * dataloader.batch_size

                    running_acc += acc * dataloader.batch_size
                    running_loss += loss.item() * dataloader.batch_size

                    if epoch % 10 == 0 and flag:
                        with torch.no_grad():
                            outputs = self.model(x)
                            outputs = self.softmax(outputs)
                            if self.weakly:
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


def get_training_kwargs(weakly: bool, loss_fn: str, reg_weight: float, dataset: str):
    if dataset not in ['swinyseg', '95cloud-3d', '95cloud-4d']:
        raise Exception(f'Dataset {dataset} does not exist, dataset must be '
                        f'one of the following swinyseg, 95cloud-3d, 95cloud-4d')
    if loss_fn not in ['ce', 'jaccard']:
        raise Exception(f'loss function {loss_fn} does not exist, loss_fn must be one of the following ce, jaccard')

    if dataset == 'swinyseg':
        network_parameters = {'inception_depth': 6,
                              'input_channels': 3}
        dataset_parameters = {'name': 'swinyseg',
                              'csv_file': '../data/swinyseg/metadata_train.csv',
                              'root_dir': '../data/swinyseg/',
                              'transform': transforms.Compose([Rescale(192), ToTensor()]),
                              'batch_size': 16}
    else:
        assert not weakly, 'Weakly training is currently only for swinyseg dataset'
        network_parameters = {'inception_depth': 6,
                              'input_channels': 3 if '3d' in dataset else 4}
        dataset_parameters = {'name': 'cloud95',
                              'csv_file': '../data/95-cloud_train/training_patches_95-cloud_nonempty.csv',
                              'root_dir': '../data/95-cloud_train/',
                              'transform': transforms.Compose([Rescale(192), ToTensor()]),
                              'batch_size': 16}
    lr = 1e-3
    kwargs = {'lr': lr,
              'network_parameters': network_parameters,
              'dataset_parameters': dataset_parameters,
              'loss_fn': loss_fn}
    if weakly:
        dense_loss_weight = reg_weight
        kwargs['dense_loss_weight'] = dense_loss_weight

    return kwargs


def train_network(weakly, loss_fn, reg_weight, dataset, plot_name='', epochs=50):

    kwargs = get_training_kwargs(weakly, loss_fn, reg_weight, dataset)

    trainer = Trainer(weakly_training=weakly, **kwargs)

    train_loss, valid_loss, train_acc, valid_acc, train_orig_acc, valid_orig_acc = trainer.train(epochs=epochs)

    plot_graph(train_loss, valid_loss, 'loss', f'../plots/losses {dataset} {plot_name}.png')
    plot_graph(train_acc, valid_acc, 'accuracy', f'../plots/ accuracy {dataset} {plot_name}.png')
    plot_graph(train_orig_acc, valid_orig_acc, 'orig accuracy',
               f'../plots/orig accuracy {dataset} {plot_name}.png')

    if dataset == 'swinyseg':
        get_models_evaluation_swinyseg(trainer.model)


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


def evaluate_performance(model, test_dl):
    # Calculate TP TN FP FN for all images in test set
    tp, tn, fp, fn = 0.0, 0.0, 0.0, 0.0
    is_cuda = next(model.parameters()).is_cuda
    dtype = torch.cuda.FloatTensor if is_cuda else torch.FloatTensor
    softmax = nn.Softmax(dim=1)

    model.train(False)

    for sample_batched in test_dl:
        x = sample_batched['image'].type(dtype)
        y = sample_batched['gt'].type(dtype)

        with torch.no_grad():
            outputs = model(x)
            outputs = softmax(outputs)
            mask = torch.argmax(outputs, 1)
            tn_curr, fp_curr, fn_curr, tp_curr = confusion_matrix(y.cpu().numpy().flatten(),
                                                                  mask.cpu().numpy().flatten()).ravel()
            tp += tp_curr
            tn += tn_curr
            fp += fp_curr
            fn += fn_curr
    jaccard_index = tp / (tp + fp + fn)
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    specificity = tn / (tn + fp)
    accuracy = (tn + tp) / (tp + tn + fp + fn)
    print(f'jaccard_index = {jaccard_index}\nprecision = {precision} \nrecall = {recall} '
          f'\nspecificity = {specificity} \naccuracy = {accuracy}')


def get_models_evaluation_swinyseg(model=None):
    dataset = SwinysegDataset('../data/swinyseg/metadata_test.csv', '../data/swinyseg/',
                              transform=transforms.Compose([Rescale(192), ToTensor()]), train=False)
    dl = DataLoader(dataset, batch_size=1, shuffle=False)
    if model is None:
        model_saved_states = ['saved_state_swinyseg', 'saved_state_swinyseg_new',
                              'saved_state_swinyseg_full_training_celoss', 'saved_state_weakly_celoss',
                              'saved_state_weakly_denseloss_1e-9',
                              'saved_state_weakly_denseloss_3e-10', 'saved_state_weakly_denseloss_5e-10',
                              'saved_state_weakly_denseloss_1e-10', 'saved_state_weakly_denseloss_4e-10']
        for model_saved_state in model_saved_states:
            model = CloudNetPlus(3, 6, residual=True)
            saved_state = torch.load(model_saved_state, map_location='cpu')
            model.load_state_dict(saved_state['model_state'])
            model.type(torch.cuda.FloatTensor)
            print(model_saved_state)
            evaluate_performance(model, dl)
    else:
        evaluate_performance(model, dl)


def inference(model: nn.Module, images: torch.Tensor, saved_state=None, gt=None):
    if saved_state is not None:
        model.load_state_dict(saved_state['model_state'])

    if images.ndim == 3:
        images = images.unsqueeze(0)
    if gt is not None and gt.ndim == 3:
        gt = gt.cpu()

    model.eval()
    softmax = nn.Softmax(dim=1)
    with torch.no_grad():
        output = model(images)
        output = softmax(output)

    show_image_inference_batch(images.cpu(), output.cpu(), gt=gt)


def show_inference(saved_state_file, num_imgs=6, gt=False, print_patches=False):
    dtype = get_dtype()

    model = CloudNetPlus(3, 6, residual=True).type(dtype)
    saved_state = torch.load(saved_state_file, map_location='cpu')

    dataset = SwinysegDataset(csv_file='../data/swinyseg/metadata_test.csv',
                              root_dir='../data/swinyseg/',
                              transform=transforms.Compose([Rescale(192), ToTensor()]))
    dl = DataLoader(dataset, batch_size=num_imgs, shuffle=False, num_workers=4)

    batch = next(iter(dl))
    imgs = batch['image'].type(dtype)
    gt_images = None
    if gt:
        gt_images = batch['gt'].type(dtype)

    if print_patches:
        print(batch['patch_name'])

    inference(model, imgs, saved_state=saved_state, gt=gt_images)


if __name__ == "__main__":
    # show_inference(num_imgs=4, gt=True, print_patches=True)
    train_network(False, 'jaccard', 0, 'swinyseg', epochs=100)

