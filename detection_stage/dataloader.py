import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import random
import math

from torch.utils.data import Dataset, DataLoader
from constants import *
from collections import namedtuple
from data_augmentation import Random



class TrajectoryTaggingDataset(Dataset):
    def __init__(self, data, args, maxlen, drop_num, drop_ratio, id2loc):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data = data
        self.maxlen = maxlen
        self.drop_num = drop_num
        self.drop_ratio = drop_ratio
        self.id2loc = id2loc

        self.drop_ratio_cl = [0.3, 0.4, 0.5, 0.6, 0.7]
        self.drop_num_cl = [1, 2, 3, 4]
        self.augmentation = Random()

    def __len__(self):
        return len(self.data)

    def sample_pair_contrastive(self, traj):
        augmented_pairs = []
        total_rate = random.sample(self.drop_ratio_cl, 1)[0]
        nums = random.choices(self.drop_num_cl, k=2)

        # augmented_pairs.append(traj)
        # augmented_pairs.append(self.augmentation(traj, nums[0], total_rate))
        for i in range(2):
            augmented_pairs.append(self.augmentation(traj, nums[i], total_rate))

        return augmented_pairs

    def traj_drop_gen(self, traj, minstart_idx=1):
        total_rate = random.sample(self.drop_ratio, 1)[0]
        num = random.sample(self.drop_num, 1)[0]
        #     print(total_rate, num)
        rate = total_rate / num

        traj = np.array(traj)
        length = len(traj)
        mingap = max(math.floor(len(traj) * rate), 1)
        slack = length - mingap * num - minstart_idx

        while slack <= 0:
            num -= 1
            slack = length - mingap * num - minstart_idx

        while slack <= num:  ## make sure that we drop fewer segments than slack number
            num -= 1

        increments = np.sort(np.random.choice(np.arange(slack), num))
        drop_idxs = minstart_idx + increments + mingap * np.arange(0, num)
        # print(length, mingap, drop_idxs)

        keep_index, binary_label, num_label = [], [], []
        start_idx = 0
        for idx in drop_idxs:
            keep_index.extend(list(range(start_idx, idx)))
            start_idx = idx + mingap
        keep_index.extend(list(range(start_idx, len(traj))))  ## and locations in the end

        for i in range(len(keep_index) - 1):
            if keep_index[i] + 1 != keep_index[i + 1]:  # has drop between two locations
                binary_label.append(1)
                num_label.append(keep_index[i+1] - keep_index[i] - 1)
            else:
                binary_label.append(0)
                num_label.append(0)

        if keep_index[-1] == length - 1:
            binary_label.append(0)
            num_label.append(0)
        else:
            binary_label.append(1)
            num_label.append(length - keep_index[-1] - 1)

        assert len(binary_label) == len(keep_index), "trip: {}, binary label: {}, keep index: {}".format(traj,
                                                                                                         binary_label,
                                                                                                         keep_index)
        return keep_index, np.array(binary_label), np.array(num_label)

    def collate_multi_class_label(self, num_label):
        res = []
        for num in num_label:
            if num == 0:
                res.append(0)
            elif 1 <= num <= 4:
                res.append(1)
            elif 5 <= num <= 9:
                res.append(2)
            elif 10 <= num <= 15:
                res.append(3)
            else:
                res.append(4)

        return np.array(res)

    def __getitem__(self, index):
        traj = self.data[index]
        traj = np.array(traj)
        keep_index, binary_label, num_label = self.traj_drop_gen(traj)
        num_label = self.collate_multi_class_label(num_label)
        traj_sparse = traj[keep_index]
        cl_pairs = self.sample_pair_contrastive(traj)

        return (traj_sparse, num_label, cl_pairs)


class TestingTaggingDataset(Dataset):
    def __init__(self, data_input, data_label, args, maxlen):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data_input = data_input
        self.data_label = data_label
        self.maxlen = maxlen


    def __len__(self):
        return len(self.data_input)


    def collate_multi_class_label(self, num_label):
        res = []
        for num in num_label:
            if num == 0:
                res.append(0)
            elif 1 <= num <= 4:
                res.append(1)
            elif 5 <= num <= 9:
                res.append(2)
            elif 10 <= num <= 15:
                res.append(3)
            else:
                res.append(4)

        return np.array(res)

    def __getitem__(self, index):
        traj = self.data_input[index]
        traj = np.array(traj)
        num_label = self.data_label[index]
        num_label = self.collate_multi_class_label(num_label)

        return (traj, num_label)





def dataloader_collate(batch):
    trips, labels, batch_cl_pairs = zip(*batch)
    batch_pairs = []
    for batch_cl in zip(*batch_cl_pairs):  # [batch_view1_list, batch_view2_list]
        batch_pairs.extend(batch_cl)

    lengths = list(map(len, trips))
    lengths_cl = list(map(len, batch_pairs))

    src = pad_arrays(trips)
    trip_locs = src[:, :, 0]
    trip_tms = src[:, :, 1:2]
    trip_coors = src[:, :, 2:]

    labels = pad_arrays(labels)
    src_cl_pairs = pad_arrays(batch_pairs)
    loc_cl_pairs = src_cl_pairs[:, :, 0]
    tm_cl_pairs = src_cl_pairs[:, :, 1:2]
    coors_cl_pairs = src_cl_pairs[:, :, 2:]


    res_tensors = (
        torch.tensor(trip_locs, dtype=torch.long),
        torch.tensor(trip_tms, dtype=torch.float),
        torch.tensor(trip_coors, dtype=torch.float),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )


    cl_tensors = (
        torch.tensor(loc_cl_pairs, dtype=torch.long),
        torch.tensor(tm_cl_pairs, dtype=torch.float),
        torch.tensor(coors_cl_pairs, dtype=torch.float),
        torch.tensor(lengths_cl, dtype=torch.long)
    )

    return res_tensors, cl_tensors


def dataloader_collate_test(batch):
    trips, labels = zip(*batch)

    lengths = list(map(len, trips))

    src = pad_arrays(trips)
    trip_locs = src[:, :, 0]
    trip_tms = src[:, :, 1:2]
    trip_coors = src[:, :, 2:]

    labels = pad_arrays(labels)

    res_tensors = (
        torch.tensor(trip_locs, dtype=torch.long),
        torch.tensor(trip_tms, dtype=torch.float),
        torch.tensor(trip_coors, dtype=torch.float),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(labels, dtype=torch.long)
    )

    return res_tensors



def invpermute(p):
    """
    inverse permutation
    """
    p = np.asarray(p)
    invp = np.empty_like(p)
    for i in range(p.size):
        invp[p[i]] = i
    return invp


def argsort(seq):
    """
    sort by length in reverse order
    ---
    seq (list[array[int32]])
    """
    return [x for x, y in sorted(enumerate(seq),
                                 key=lambda x: len(x[1]),
                                 reverse=True)]


def pad_array(a, max_length, pad_time=5000, pad=0, pad_lon=20447840.4, pad_lat=4419792.3):
    """
    a (array)
    """
    if len(a.shape) == 2 and a.shape[1] == 4: ## input seq (loc id, timestamp, lons, lats)
        arr_np = np.array([(pad, pad_time, pad_lon, pad_lat)] * (max_length - len(a)))
        if len(arr_np) != 0:
            res = np.concatenate((a, arr_np))
        else:
            res = a
    elif len(a.shape) == 2 and a.shape[1]==2: ## input seq (loc id, timestamp)
        arr_np = np.array([(pad, pad_time)] * (max_length - len(a)))
        if len(arr_np) != 0:
            res = np.concatenate((a, arr_np))
        else:
            res = a
    elif len(a.shape) == 1: ## label seq (0 or 1 for tagging)
        arr_np = np.array([pad] * (max_length - len(a)))
        res = np.concatenate((a, arr_np))
    # print(a.shape, arr_np.shape)
    # print(a, arr_np)

    return res


def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a)
    # print(a.shape, a)
    return a
