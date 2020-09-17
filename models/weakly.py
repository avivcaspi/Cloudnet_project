import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import os

from skimage import io, transform
from skimage.morphology import binary_erosion, square, disk, binary_opening, thin, binary_dilation
from torchvision.utils import save_image
from dataset import SwinysegDataset
from multiprocessing import Process
import multiprocessing
import torch.utils.data as data
from PIL import Image


# ---------------------- Start generation testing code ------------------------
def erode(img, selem):
    res = binary_erosion(img, selem).astype(float)
    return res


def open(img, selem):
    res = binary_opening(img, selem).astype(float)
    return res


def open_and_erode(img):
    img = np.pad(img, 1, 'constant', constant_values=0)
    selem_disk = disk(10)
    selem_square = square(20)
    plt.figure()
    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')

    res_img = open(img, selem_disk)
    plt.subplot(1, 3, 2)
    plt.imshow(res_img, cmap='gray')
    res_img = erode(res_img, selem_disk)
    plt.subplot(1, 3, 3)
    plt.imshow(res_img, cmap='gray')
    plt.show()


def BeamSearch(img: np.ndarray, start: tuple, finish: tuple, h: callable):
    k = 100
    open_list = []
    close_list = []
    if img[start] == 1:
        start_node = Node(start, h(img, start, finish), None)
        open_list.append(start_node)
    while len(open_list) > 0:
        next_node = open_list.pop(0)
        img[next_node.p] = 0.8
        close_list.append(next_node)
        if next_node.p == finish:
            return next_node
        children = get_next_states(img, next_node.p)
        for child in children:
            child_node = Node(child, h(img, child, finish), next_node)
            if child_node not in open_list and child_node not in close_list:
                insert_node_sorted_h(open_list, child_node)
                if len(open_list) > k:
                    open_list.pop()
    return None


def insert_node_sorted_h(list, node):
    for i, next in enumerate(list):
        if node.h < next.h:
            list.insert(i, node)
            return
    list.append(node)


def DFS_util(img, p):
    img[p] = 0.5
    flag = True
    best_len = 0
    best_p = None
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            child = (p[0] + i, p[1] + j)
            if 0 <= child[0] < img.shape[0] and 0 <= child[1] < img.shape[1] and img[child] == 1.0:
                flag = False
                length, end = DFS_util(img, child)
                if length > best_len:
                    best_p = end
                    best_len = length
    if flag:
        return 1, p
    return best_len + 1, best_p


def DFS(orig_img, p):
    img = orig_img.copy()
    length = 0
    end = p
    if img[p] == 1.0:
        length, end = DFS_util(img, p)

    return img, length, end
# ---------------------- End generation testing code ------------------------


# ---------------------- Start final scribbles generation code ------------------------
class Node:
    def __init__(self, p, h, father, g=0.0, w=0.5):
        self.p = p
        self.h = h
        self.father = father
        self.g = g
        self.f = w * h + (1 - w) * g

    def __eq__(self, other):
        if self.p == other.p:
            return True
        return False


def dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def heuristic(img, p, finish):
    distance = dist(p, finish)
    return distance


def get_next_states(img, p):
    children = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            child = (p[0] + i, p[1] + j)
            if 0 <= child[0] < img.shape[0] and 0 <= child[1] < img.shape[1]:
                if img[child] != 0:
                    children.append(child)
    return children


def p_in_node_list(p, list):
    for node in list:
        if node.p == p:
            return node
    return False


def BFS(orig_img, p):
    img = orig_img.copy()
    open_list = []
    close_list = []
    if img[p] == 1.0:
        open_list.append(p)

    while len(open_list) > 0:
        q = open_list.pop(0)
        img[q] = 0.5
        close_list.append(q)
        for i in [-1, 0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                child = (q[0] + i, q[1] + j)
                if 0 <= child[0] < img.shape[0] and 0 <= child[1] < img.shape[1] and\
                        child not in open_list and child not in close_list:
                    if img[child] == 1.0:
                        open_list.append(child)
    return img


def weighted_astar(img, start, finish, w=0.5, verbose=0):
    open_nodes = []
    close_nodes = []

    start_node = Node(start, heuristic(img, start, finish), None, w=w)
    open_nodes.append(start_node)

    end_node = None
    while len(open_nodes) > 0:
        current = open_nodes.pop(0)
        close_nodes.append(current)
        if current.p == finish:
            if verbose:
                print('found end')
            end_node = current
            break
        img[current.p] = 0.8
        children = get_next_states(img, current.p)
        for child in children:
            if p_in_node_list(child, close_nodes):
                continue

            child_node = Node(child, heuristic(img, child, finish), current, current.g + 1, w=w)
            child_old = p_in_node_list(child, open_nodes)
            if child_old:
                if child_old.g < child_node.g:
                    continue
                else:
                    open_nodes.remove(child_old)

            for i, node in enumerate(open_nodes):
                if child_node.f < node.f:
                    open_nodes.insert(i, child_node)
                    break
            if child_node not in open_nodes:
                open_nodes.append(child_node)

    '''while end_node.father is not None:
        p = end_node.p
        img[p] = 0.3
        end_node = end_node.father'''
    return end_node


def thin_and_connect(orig_img, verbose=0):
    routes = []
    img = orig_img.copy()
    thin_img = thin(img).astype(float)
    w, h = img.shape
    if sum(sum(thin_img)) == 0:
        return thin_img, orig_img
    while sum(sum(thin_img)) > 0:
        thin_len = 1
        route_len = 0
        j = 0
        q = 0.35
        while route_len < q * thin_len:
            j += 1
            if j % 5 == 0:
                q *= 0.8 if q > 0.08 else 0.08
                if verbose:
                    print(f'Size was decreased to {q}')
            i = 0
            while True:
                i += 1
                x1 = np.random.randint(0, h)
                y1 = np.random.randint(0, w)
                if thin_img[x1, y1] == 1:
                    break
            while True:
                i += 1
                x2 = np.random.randint(0, h)
                y2 = np.random.randint(0, w)
                if thin_img[x2, y2] == 1:
                    if verbose:
                        print(f'Start = ({x1}, {y1}),  Finish = ({x2}, {y2}). Found after {i} tries')
                    break
            start, finish = (x1, y1), (x2, y2)
            end_node = weighted_astar(thin_img.copy(), start, finish, w=1, verbose=verbose)
            res_img = np.zeros_like(img, dtype=float)
            if end_node is None:
                if verbose:
                    print('Route not found')
            while end_node is not None:
                p = end_node.p
                res_img[p] = 1.0
                end_node = end_node.father
            route_len = sum(sum(res_img))
            bfs_img = BFS(thin_img, start)
            bfs_img[bfs_img != 0.5] = 0
            bfs_img[bfs_img != 0] = 1
            thin_len = sum(sum(bfs_img))

            if verbose:
                print(f'Thin len = {thin_len}, Route len = {route_len}')
            final_img = img.copy() - res_img
            if thin_len < 5:
                continue
        if verbose:
            print(f'Good route after {j} tries')
        if verbose == 2:
            fig, ax = plt.subplots(1, 5, figsize=(20, 10))
            ax[0].imshow(img, cmap='gray')
            ax[0].set_title('Original', fontweight="bold", size=20)
            ax[1].imshow(thin_img, cmap='gray')
            ax[1].set_title('Thin', fontweight="bold", size=20)
            ax[2].imshow(res_img, cmap='gray')
            ax[2].set_title('One route', fontweight="bold", size=20)
            ax[3].imshow(bfs_img, cmap='gray')
            ax[3].set_title('Full part', fontweight="bold", size=20)
            ax[4].imshow(final_img, cmap='gray')
            ax[4].set_title('Final', fontweight="bold", size=20)
            plt.show()
        routes.append(res_img)
        thin_img -= bfs_img

    res_thin = sum(routes)
    final_res = binary_dilation(res_thin, square(3)).astype(float)
    fixed_res = final_res * img  # In case the dilation made the scribble go out of line
    if verbose:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(fixed_res, cmap='gray')
        ax[0].set_title('Final gt', fontweight="bold", size=20)
        ax[1].imshow(orig_img - 0.5 * fixed_res, cmap='gray')
        ax[1].set_title('Weakly', fontweight="bold", size=20)
        plt.show()

    return fixed_res, img - 0.5 * fixed_res


def convert_dataset_to_weakly(dataset):
    p_start_time = time.time()
    weakly_path = '../data/swinyseg/WeaklyGT/'
    full_res_path = '../data/swinyseg/WeaklyFull/'
    i = 0

    for sample in dataset:
        i += 1
        if i % 100 == 0:
            print(f'Done {i} images in process {multiprocessing.current_process()} in {time.time() - p_start_time} seconds')
        gt = sample['gt']
        inverted_gt = 1 - gt
        patch_name = sample['patch_name'].replace('.jpg', '.png')
        final_res, full_res = thin_and_connect(gt, 0)
        inverted_final_res, inverted_full_res = thin_and_connect(inverted_gt, 0)

        final_gt = np.ones_like(final_res) * 255.0
        final_gt[final_res == 1] = 1.0
        final_gt[inverted_final_res == 1] = 0.0
        final_full = full_res + inverted_final_res * 0.8

        final_im = Image.fromarray(final_gt).convert('L')
        final_im.save(weakly_path + patch_name)
        full_im = Image.fromarray(final_full * 255.0).convert('L')
        full_im.save(full_res_path + patch_name)

    print(f'{multiprocessing.current_process()} finished in {time.time() - p_start_time} seconds')


def generate_weakly_set():
    dataset = SwinysegDataset(
        csv_file='../data/swinyseg/metadata.csv',
        root_dir='../data/swinyseg/'
    )

    if not os.path.isdir('../data/swinyseg/WeaklyGT/'):
        os.mkdir('../data/swinyseg/WeaklyGT/')
    if not os.path.isdir('../data/swinyseg/WeaklyFull/'):
        os.mkdir('../data/swinyseg/WeaklyFull/')

    length = len(dataset)
    num_threads = 4
    process_len = round(length / num_threads)

    lens = [process_len] * (num_threads - 1)
    lens.append(length - ((num_threads - 1) * process_len))
    subsets = data.random_split(dataset, lens)

    threads = []
    for subset in subsets:
        threads.append(Process(target=convert_dataset_to_weakly, args=(subset,)))
        threads[-1].start()

    for thread in threads:
        thread.join()

# ---------------------- End final scribbles generation code ------------------------


if __name__ == '__main__':
    start_time = time.time()
    generate_weakly_set()
    print(f'Finished  all images in {time.time() - start_time} seconds')


    '''dataset = SwinysegDataset(
        csv_file='../data/swinyseg/metadata.csv',
        root_dir='../data/swinyseg/'
    )
    img = dataset[1]['gt']
    img = transform.resize(img, (256, 256))
    img[img > 0.5 ] = 1
    img[img <= 0.5] = 0
    fig, axes = plt.subplots(1, 3, figsize=(15, 7))
    axes = axes.reshape([1, 3])
    axes[0,0].set_title('Original', fontsize=15)
    axes[0,0].imshow(img, cmap='gray')

    start = (100,25)
    finish = (80,240)
    n_img = erode(img , disk(10))
    axes[0, 1].imshow(n_img, cmap='gray')
    axes[0, 1].set_title('Erode', fontsize=15)
    end_node = weighted_astar(n_img, start, finish, 1,1)
    while end_node is not None and end_node.father is not None:
        p = end_node.p
        img[p] = 0.3
        end_node = end_node.father

    axes[0,2].imshow(img, cmap='gray')
    axes[0, 2].set_title('Final',fontsize=15)

    plt.setp(axes, xticks=[], yticks=[])
    plt.show()'''
    '''dataset = SwinysegDataset(
        csv_file='../data/swinyseg/metadata.csv',
        root_dir='../data/swinyseg/'
    )


    img = np.zeros((100, 100), dtype=float)
    img[20:80, 30:70] = 1.0

    img = dataset[1]['gt']

    thin_img = thin(img).astype(float)

    p = (62, 166)
    plt.imshow(thin_img, cmap='gray')
    plt.show()
    _, length, q = DFS(thin_img, p)
    end_node = weighted_astar(thin_img, p, q)
    res_img = np.zeros_like(img)
    if end_node is None:
        print('Route not found')
    else:
        res_img[end_node.p] = 1.0
    while end_node is not None and end_node.father is not None:
        p = end_node.p
        res_img[p] = 1.0
        end_node = end_node.father
    plt.imshow(res_img, cmap='gray')
    plt.show()'''


