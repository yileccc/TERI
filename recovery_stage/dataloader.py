import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import math
import random

from torch.utils.data import Dataset
from constants import *
from collections import namedtuple
from data_augmentation import Random



class TrajectoryInfillingDataset(Dataset):
    def __init__(self, data, args, maxlen, drop_num, drop_ratio, id2loc):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data = data
        self.maxlen = maxlen
        self.drop_num = drop_num
        self.drop_ratio = drop_ratio
        self.drop_ratio_cl = [0.2, 0.3, 0.4, 0.5, 0.6]
        self.drop_num_cl = [1, 2, 3, 4]
        self.id2loc = id2loc
        self.augmentation = Random()

    def __len__(self):
        return len(self.data)


    def sample_pair_contrastive(self, traj):
        augmented_pairs = []
        total_rate = random.sample(self.drop_ratio_cl, 1)[0]
        nums = random.choices(self.drop_num_cl, k=2)

        for i in range(2):
            augmented_pairs.append(self.augmentation(traj))

        return augmented_pairs


    def sample_traj_recovery(self, traj):
        total_rate = random.sample(self.drop_ratio, 1)[0]
        num = random.sample(self.drop_num, 1)[0]
        rate = total_rate / num
        source, target, masked_pos, masked_tokens = self.traj_dropping(traj, num, rate)

        return source, target, masked_pos, masked_tokens



    def traj_dropping(self, traj, num, rate, minstart_idx=1, max_filling_length=25):
        length = len(traj)
        gap = max(math.floor(len(traj) * rate), 1)
        slack = length - gap * num - minstart_idx

        while slack <= 0:
            num -= 1
            slack = length - gap * num - minstart_idx
        while slack <= num:  ## make sure that we drop fewer segments than slack number
            num -= 1

        increments = np.sort(np.random.choice(np.arange(slack), num))
        drop_idxs = minstart_idx + increments + gap * np.arange(0, num)

        source, target, masked_pos, masked_tokens = [], [], [], []
        ## source/target include all the trajectory, masked_pos/tokens only inlcude the masked information
        start_idx, input_length = 0, 0

        keep_index = []
        for idx in drop_idxs:
            keep_index.extend(list(range(start_idx, idx)))
            start_idx = idx + gap
        keep_index.extend(list(range(start_idx, len(traj))))  ## and locations in the end


        for i in range(len(keep_index)-1):
            if keep_index[i] + 1 == keep_index[i+1]: ## no drop between two locations
                idx = keep_index[i]
                source.append(traj[idx])
                target.append(PAD_TOKEN)
                input_length += 1
            else:
                idx = keep_index[i]
                source.append(traj[idx])
                target.append(PAD_TOKEN)
                input_length += 1

                gap_length = keep_index[i+1] - keep_index[i] -1
                drop_index = list(range(keep_index[i]+1, keep_index[i+1]))
                drop_index = drop_index[:max_filling_length]
                cur_gap = len(drop_index)

                src_blk_tokens = self.collate_multi_class_label(gap_length)
                masked_index = list(range(input_length, input_length+len(src_blk_tokens)))
                input_length += len(src_blk_tokens)


                masked_pos.extend(masked_index)
                masked_tokens.extend(traj[drop_index, 0].tolist() + [NUL_TOKEN] * (len(src_blk_tokens) - cur_gap))
                source.extend(src_blk_tokens)
                target.extend(traj[drop_index, 0].tolist() + [NUL_TOKEN] * (len(src_blk_tokens) - cur_gap))

        source.extend(traj[list(range(keep_index[-1], len(traj)))])  ## and locations in the end
        target.extend([PAD_TOKEN] * (len(traj) - keep_index[-1]))


        assert len(source) == len(target), "length of source and target not equal"
        assert len(masked_pos) == len(masked_tokens), "length of masked pos and masked tokens not equal"
        return np.array(source), np.array(target), np.array(masked_pos), np.array(masked_tokens)


    def collate_multi_class_label(self, num_label):
        if self.args.num_cls == 8:
            if num_label < 5:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(num_label)]
            elif 5 <= num_label <= 6:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(6)]
            elif 7 <= num_label <= 10:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(10)]
            else:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
        elif self.args.num_cls == 5:
            if num_label == 0:
                res = []
            elif 1 <= num_label <= 4:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(4)]
            elif 5 <= num_label <= 9:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(9)]
            elif 10 <= num_label <= 15:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
            else:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(25)]
        elif self.args.num_cls == 2:
            if num_label == 0:
                res = []
            else:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
        else:
            raise ValueError("num of classes not specified")
        return res

    def __getitem__(self, index):
        traj = self.data[index]
        traj = np.array(traj)
        source, target, masked_pos, masked_tokens = self.sample_traj_recovery(traj)
        cl_pairs = self.sample_pair_contrastive(traj)

        return (source, target, masked_pos, masked_tokens, cl_pairs)


class TestingInfillingDataset(Dataset):
    def __init__(self, data_input, data_num_labels, data_label, args, maxlen):
        self.args = args
        self.batch_size = self.args.batch_size
        self.data_input = data_input
        self.data_num_labels = data_num_labels
        self.data_label = data_label
        self.maxlen = maxlen

    def __len__(self):
        return len(self.data_input)

    def collate_multi_class_label(self, num_label):
        if self.args.num_cls == 8:
            if num_label < 5:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(num_label)]
            elif 5 <= num_label <= 6:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(6)]
            elif 7 <= num_label <= 10:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(10)]
            else:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
        elif self.args.num_cls == 5:
            if num_label == 0:
                res = []
            elif 1 <= num_label <= 4:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(4)]
            elif 5 <= num_label <= 9:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(9)]
            elif 10 <= num_label <= 15:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
            else:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(25)]
        elif self.args.num_cls == 2:
            if num_label == 0:
                res = []
            else:
                res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
        else:
            raise ValueError("num of classes not specified")
        return res

    def get_input_data(self, traj, num_label):
        res, masked_pos = [], []
        assert len(traj) == len(num_label)
        for (record, num) in zip(traj, num_label):
            if num == 0:
                res.append(record)
            else:
                res.append(record)
                blk_tokens = self.collate_multi_class_label(num)
                masked_pos.extend(list(range(len(res), len(res)+len(blk_tokens))))
                res.extend(blk_tokens)


        return np.array(res), np.array(masked_pos)


    def __getitem__(self, index):
        traj, num_label = self.data_input[index], self.data_num_labels[index]
        traj = np.array(traj)
        traj_input, masked_pos = self.get_input_data(traj, num_label)
        truth = self.data_label[index]
        label = np.array([rec[0] for rec in truth])


        return (traj_input, masked_pos, label)


def dataloader_collate_test(batch):
    batch_inputs, batch_masked_pos, batch_labels = zip(*batch)
    lengths = list(map(len, batch_inputs))
    masked_pos_lengths = list(map(len, batch_masked_pos))

    src = pad_arrays(batch_inputs)
    trip_locs = src[:, :, 0]
    trip_tms = src[:, :, 1:2]
    trip_coors = src[:, :, 2:]


    batch_masked_pos = pad_arrays(batch_masked_pos)

    # print(src.shape, batch_masked_pos.shape)

    res_tensors = (
        torch.tensor(trip_locs, dtype=torch.long),
        torch.tensor(trip_tms, dtype=torch.float),
        torch.tensor(trip_coors, dtype=torch.float),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(batch_masked_pos, dtype=torch.long),
        torch.tensor(masked_pos_lengths, dtype=torch.long)
    )


    return res_tensors


def dataloader_collate(batch):
    batch_idxs, batch_idxs_output, batch_masked_pos, batch_masked_tokens, batch_cl_pairs = zip(*batch)

    batch_pairs = []
    for batch_cl in zip(*batch_cl_pairs): #[batch_view1_list, batch_view2_list]
        batch_pairs.extend(batch_cl)

    # print(batch_pairs)

    lengths = list(map(len, batch_idxs))
    lengths_cl = list(map(len, batch_pairs))

    src = pad_arrays(batch_idxs)
    trip_locs = src[:, :, 0]
    trip_tms = src[:, :, 1:2]
    trip_coors = src[:, :, 2:]

    src_cl_pairs = pad_arrays(batch_pairs)
    trip_cl_pairs = src_cl_pairs[:, :, 0]
    trip_cl_tms = src_cl_pairs[:, :, 1:2]
    trip_cl_coors = src_cl_pairs[:, : , 2:]

    batch_pred_inputs = [np.array([BLK_TOKEN] + pred_token[:-1].tolist()) for pred_token in batch_masked_tokens]
    batch_pred_targets = batch_masked_tokens
    batch_pred_inputs = pad_arrays(batch_pred_inputs)
    batch_pred_targets = pad_arrays(batch_pred_targets)

    batch_masked_pos = pad_arrays(batch_masked_pos)

    value_bool = (batch_pred_targets != NUL_TOKEN) & (batch_pred_targets != PAD_TOKEN)
    value_weight = 1
    batch_masked_weight = (batch_pred_targets != PAD_TOKEN).astype(float)
    batch_masked_weight[value_bool] = value_weight

    # print(src.shape, batch_pred_inputs.shape, batch_pred_targets.shape, batch_masked_pos.shape, batch_masked_weight.shape)

    rec_tensors = (
        torch.tensor(trip_locs, dtype=torch.long),
        torch.tensor(trip_tms, dtype=torch.float),
        torch.tensor(trip_coors, dtype=torch.float),
        torch.tensor(lengths, dtype=torch.long),
        torch.tensor(batch_masked_pos, dtype=torch.long),
        torch.tensor(batch_pred_inputs, dtype=torch.long),
        torch.tensor(batch_pred_targets, dtype=torch.long),
        torch.tensor(batch_masked_weight, dtype=torch.float)
    )

    cl_tensors = (
        torch.tensor(trip_cl_pairs, dtype=torch.long),
        torch.tensor(trip_cl_tms, dtype=torch.float),
        torch.tensor(trip_cl_coors, dtype=torch.float),
        torch.tensor(lengths_cl, dtype=torch.long)
    )


    return rec_tensors, cl_tensors



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
    a (array[int32])
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

    return res


def pad_arrays(a):
    max_length = max(map(len, a))
    a = [pad_array(a[i], max_length) for i in range(len(a))]
    a = np.stack(a)
    return a
