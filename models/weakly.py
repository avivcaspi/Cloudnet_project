import numpy as np
import torch
import matplotlib.pyplot as plt
import time
from skimage.morphology import binary_erosion, square, disk, binary_opening, thin

from dataset import SwinysegDataset


class Node:
    def __init__(self, p, h, father, g=0.0, w=0.5):
        self.p = p
        self.h = h
        self.father = father
        self.g = g
        self.f = w * h + (1-w) * g

    def __eq__(self, other):
        if self.p == other.p:
            return True
        return False


def BeamSearch(img: np.ndarray, start: tuple, finish: tuple, h: callable):
    k = 100
    open_list = []
    close_list = []
    if img[start] == 1:
        start_node = Node(start, h(img, start, finish), None)
        open_list.append(start_node)
    while len(open_list) > 0:
        next_node = open_list.pop(0)
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


def dist(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def heuristic(img, p, finish):
    distance = dist(p, finish)
    black = 10
    '''for i in range(1, 10, 2):
        if 0 <= p[0] + i < img.shape[0]:
            if img[p[0] + i, p[1]] == 0:
                black = i
                break
        if 0 <= p[0] - i < img.shape[0]:
            if img[p[0] - i, p[1]] == 0:
                black = i
                break
        if 0 <= p[1] + i < img.shape[1]:
            if img[p[0], p[1] + i] == 0:
                black = i
                break
        if 0 <= p[1] - i < img.shape[1]:
            if img[p[0], p[1] - 1] == 0:
                black = i
                break
    near_black = 30 - black*3'''
    near_black = 0
    return distance + near_black


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
        for i in [-1,0, 1]:
            for j in [-1, 0, 1]:
                if i == 0 and j == 0:
                    continue
                child = (q[0] + i, q[1] + j)
                if 0 <= child[0] < img.shape[0] and 0 <= child[1] < img.shape[1] and child not in open_list and child not in close_list:
                    if img[child] == 1.0:
                        open_list.append(child)
    return img


def weighted_astar(img, start, finish, w=0.5):
    open_nodes = []
    close_nodes = []

    start_node = Node(start, heuristic(img, start, finish), None, w=w)
    open_nodes.append(start_node)

    end_node = None
    while len(open_nodes) > 0:
        current = open_nodes.pop(0)
        close_nodes.append(current)
        if current.p == finish:
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


def thin_and_connect(img, verbose=False):
    routes = []
    thin_img = thin(img).astype(float)
    w, h = img.shape

    while sum(sum(thin_img)) > 0:
        thin_len = 1
        route_len = 0
        j = 0
        while route_len < 0.2 * thin_len:
            j += 1
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
                    print(f'Start = ({x1}, {y1}),  Finish = ({x2}, {y2}). Found after {i} tries')
                    break
            start, finish = (x1, y1), (x2, y2)
            end_node = weighted_astar(thin_img.copy(), start, finish, w=1)
            res_img = np.zeros_like(img, dtype=float)
            if end_node is None:
                print('Route not found')
            while end_node is not None and end_node.father is not None:
                p = end_node.p
                res_img[p] = 1.0
                end_node = end_node.father
            route_len = sum(sum(res_img))
            bfs_img = BFS(thin_img, start)
            bfs_img[bfs_img != 0.5] = 0
            bfs_img[bfs_img != 0] = 1
            thin_len = sum(sum(bfs_img))
            print(f'Thin len = {thin_len}, Route len = {route_len}')

            final_img = img.copy() - res_img

        print(f'Good route after {j} tries')
        if verbose:
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

    final_res = sum(routes)
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(final_res, cmap='gray')
    ax[0].set_title('Final gt', fontweight="bold", size=20)
    ax[1].imshow(img - final_res, cmap='gray')
    ax[1].set_title('Weakly', fontweight="bold", size=20)
    plt.show()
    return final_res, img - final_res


if __name__ == '__main__':
    start_time = time.time()

    dataset = SwinysegDataset(
        csv_file='../data/swinyseg/metadata.csv',
        root_dir='../data/swinyseg/'
    )
    i = 0
    res, full = [], []
    for sample in dataset:
        i += 1
        gt = sample['gt']

        final_res, weakly = thin_and_connect(gt, True)
        res.append(final_res)
        full.append(weakly)
        if i > 5:
            break

    fig, ax = plt.subplots(len(full), 2, figsize=(10, len(full) * 5))
    ax[0, 0].set_title('Final gt', fontweight="bold", size=20)
    ax[0, 1].set_title('Full image', fontweight="bold", size=20)
    for i, (final_gt, full_gt) in enumerate(zip(res, full)):
        ax[i, 0].imshow(final_gt, cmap='gray')
        ax[i, 1].imshow(full_gt, cmap='gray')

    plt.show()

    print(time.time() - start_time)


