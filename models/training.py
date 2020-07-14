import time
from IPython.display import clear_output
import torch
import torch.nn as nn
from cloud_net_plus import CloudNetPlus
from losses import FilteredJaccardLoss
from dataset import Cloud95Dataset, ToTensor, Rescale
from torchvision import transforms
from torch.utils.data import DataLoader


def train(model, train_dl, valid_dl, loss_fn, optimizer, acc_fn, epochs=1):
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

    best_acc = 0.0

    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)  # Set trainind mode = true
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
                    loss = loss_fn(outputs, y)

                    # the backward pass frees the graph memory, so there is no
                    # need for torch.no_grad in this training pass
                    loss.backward()
                    optimizer.step()
                    # scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y.long())

                # stats - whatever is the phase
                acc = acc_fn(outputs, y, dtype)

                running_acc += acc * dataloader.batch_size
                running_loss += loss * dataloader.batch_size

                if step % 100 == 0:
                    # clear_output(wait=True)
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc,
                                                                                          torch.cuda.memory_allocated() / 1024 / 1024))

                    #print(torch.cuda.memory_summary())

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            clear_output(wait=True)
            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)

            train_loss.append(epoch_loss) if phase == 'train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    return train_loss, valid_loss


def acc_metric(pred, y, dtype):
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
    return (pred == y.type(dtype)).float().mean()


if __name__ == "__main__":
    # Little test
    cloud_net = CloudNetPlus(4, 6)
    dataset = Cloud95Dataset(csv_file='../data/95-cloud_train/training_patches_95-cloud_nonempty.csv',
                             root_dir='../data/95-cloud_train/',
                             transform=transforms.Compose([Rescale(192), ToTensor()]))
    length = len(dataset)
    train_size = int(0.85 * length)
    train_ds, valid_ds = torch.utils.data.random_split(dataset, (train_size, length - train_size))
    train_dl = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=6)
    valid_dl = DataLoader(valid_ds, batch_size=16, shuffle=True, num_workers=6)
    loss_fn = FilteredJaccardLoss()
    opt = torch.optim.Adam(cloud_net.parameters(), lr=0.01)
    train_loss, valid_loss = train(cloud_net, train_dl, valid_dl, loss_fn, opt, acc_metric, epochs=50)
