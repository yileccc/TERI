import random
import copy
import math
import numpy as np
from constants import *


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self):
        # self.data_augmentation_methods = [Crop2(), Mask2()]
        self.data_augmentation_methods = [Mask3()]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence)


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self):
        self.ratio = [0.7]

    def __call__(self, traj):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(traj)
        sample_ratio = random.sample(self.ratio, 1)[0]
        sub_seq_length = int(sample_ratio * len(copied_sequence))
        # randint generate int x in range: a <= x <= b
        start_index = random.randint(1, len(copied_sequence) - sub_seq_length - 1)
        if sub_seq_length < 1:
            return [copied_sequence[start_index]]
        else:
            cropped_seq = copied_sequence[start_index : start_index + sub_seq_length]
            return cropped_seq



class Crop2(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self):
        self.num = [1, 2, 3]
        self.ratio = [0.7]

    def __call__(self, traj, minstart_idx=1):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(traj)
        sample_ratio = random.sample(self.ratio, 1)[0]
        num = random.sample(self.num, 1)[0]
        ratio_each = sample_ratio/num

        length = len(copied_sequence)
        gap = max(math.floor(len(traj) * ratio_each), 1)
        slack = length - gap * num - minstart_idx

        while slack <= 0:
            num -= 1
            slack = length - gap * num - minstart_idx
        while slack <= num:  ## make sure that we drop fewer segments than slack number
            num -= 1

        increments = np.sort(np.random.choice(np.arange(slack), num))
        drop_idxs = minstart_idx + increments + gap * np.arange(0, num)
        # print(length, mingap, drop_idxs)

        start_idx, input_length = 0, 0

        keep_index = []
        for idx in drop_idxs:
            keep_index.extend(list(range(start_idx, idx)))
            start_idx = idx + gap
        keep_index.extend(list(range(start_idx, len(traj))))  ## and locations in the end
        cropped_traj = copied_sequence[keep_index]

        return cropped_traj



class Mask(object):
    """Randomly mask k items given a sequence"""

    def __init__(self):
        # self.ratio = [0.4, 0.5, 0.6, 0.7]
        self.ratio = [0.7]

    def __call__(self, sequence):
        sample_ratio = random.sample(self.ratio, 1)[0]
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(sequence)
        mask_nums = int(sample_ratio * len(copied_sequence))
        mask = [BLK_TOKEN for i in range(mask_nums)]
        mask_idx = random.sample([i for i in range(len(copied_sequence))], k=mask_nums)
        for idx, mask_value in zip(mask_idx, mask):
            copied_sequence[idx] = mask_value
        return copied_sequence



class Mask2(object):
    """Randomly mask k items given a sequence"""

    def __init__(self):
        self.num = [1,2,3]
        self.ratio = [0.7]


    def __call__(self, traj, minstart_idx=1):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(traj)
        sample_ratio = random.sample(self.ratio, 1)[0]
        num = random.sample(self.num, 1)[0]
        ratio_each = sample_ratio / num

        length = len(copied_sequence)
        gap = max(math.floor(len(traj) * ratio_each), 1)
        slack = length - gap * num - minstart_idx

        while slack <= 0:
            num -= 1
            slack = length - gap * num - minstart_idx
        while slack <= num:  ## make sure that we drop fewer segments than slack number
            num -= 1

        increments = np.sort(np.random.choice(np.arange(slack), num))
        drop_idxs = minstart_idx + increments + gap * np.arange(0, num)
        # print(length, mingap, drop_idxs)

        start_idx, input_length = 0, 0

        drop_index = []
        for idx in drop_idxs:
            drop_index.extend(list(range(idx, idx+gap)))

        for idx in drop_index:
            copied_sequence[idx] = BLK_TOKEN

        return copied_sequence



class Mask3(object):
    """Randomly mask k items given a sequence"""

    def __init__(self):
        self.num = [1,2,3]
        self.ratio = [0.3, 0.4, 0.5, 0.6, 0.7]


    def collate_multi_class_label(self, num_label):
        if num_label == 0:
            res = []
        elif 1 <= num_label <= 3:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(3)]
        elif 4 <= num_label <= 6:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(6)]
        elif 7 <= num_label <= 10:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(10)]
        else:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
        return res


    def __call__(self, traj, minstart_idx=1, max_filling_length=15):
        # make a deep copy to avoid original sequence be modified
        sample_ratio = random.sample(self.ratio, 1)[0]
        num = random.sample(self.num, 1)[0]
        ratio_each = sample_ratio / num

        length = len(traj)
        gap = max(math.floor(len(traj) * ratio_each), 1)
        slack = length - gap * num - minstart_idx

        while slack <= 0:
            num -= 1
            slack = length - gap * num - minstart_idx
        while slack <= num:  ## make sure that we drop fewer segments than slack number
            num -= 1

        increments = np.sort(np.random.choice(np.arange(slack), num))
        drop_idxs = minstart_idx + increments + gap * np.arange(0, num)
        # print(length, mingap, drop_idxs)

        start_idx, source = 0, []
        keep_index = []

        for idx in drop_idxs:
            keep_index.extend(list(range(start_idx, idx)))
            start_idx = idx + gap
        keep_index.extend(list(range(start_idx, len(traj))))  ## and locations in the end


        for i in range(len(keep_index) - 1):
            if keep_index[i] + 1 == keep_index[i + 1]:  ## no drop between two locations
                idx = keep_index[i]
                source.append(traj[idx])
            else:
                idx = keep_index[i]
                source.append(traj[idx])

                gap_length = keep_index[i + 1] - keep_index[i] - 1
                src_blk_tokens = self.collate_multi_class_label(gap_length)
                source.extend(src_blk_tokens)

        source.extend(traj[list(range(keep_index[-1], len(traj)))])  ## and locations in the end

        return np.array(source)



class Mask4(object):
    """Randomly mask k items given a sequence"""

    def __init__(self):
        self.num = [1,2,3,4]
        self.ratio = [0.3, 0.4, 0.5, 0.6, 0.7]


    def collate_multi_class_label(self, num_label):
        if num_label == 0:
            res = []
        elif 1 <= num_label <= 3:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(3)]
        elif 4 <= num_label <= 6:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(6)]
        elif 7 <= num_label <= 10:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(10)]
        else:
            res = [[BLK_TOKEN, PAD_TIME, PAD_LON, PAD_LAT] for _ in range(15)]
        return res


    def __call__(self, traj, minstart_idx=1, max_filling_length=15):
        # make a deep copy to avoid original sequence be modified
        sample_ratio = random.sample(self.ratio, 1)[0]
        num = random.sample(self.num, 1)[0]
        ratio_each = sample_ratio / num

        length = len(traj)
        gap = max(math.floor(len(traj) * ratio_each), 1)
        slack = length - gap * num - minstart_idx

        while slack <= 0:
            num -= 1
            slack = length - gap * num - minstart_idx
        while slack <= num:  ## make sure that we drop fewer segments than slack number
            num -= 1

        increments = np.sort(np.random.choice(np.arange(slack), num))
        drop_idxs = minstart_idx + increments + gap * np.arange(0, num)
        # print(length, mingap, drop_idxs)

        start_idx, source, source2 = 0, [], []
        keep_index = []

        for idx in drop_idxs:
            keep_index.extend(list(range(start_idx, idx)))
            start_idx = idx + gap
        keep_index.extend(list(range(start_idx, len(traj))))  ## and locations in the end


        for i in range(len(keep_index) - 1):
            if keep_index[i] + 1 == keep_index[i + 1]:  ## no drop between two locations
                idx = keep_index[i]
                source.append(traj[idx])
                source2.append(traj[idx])
            else:
                idx = keep_index[i]
                source.append(traj[idx])
                source2.append(traj[idx])

                gap_length = keep_index[i + 1] - keep_index[i] - 1
                src_blk_tokens = self.collate_multi_class_label(gap_length)
                src_blk_length = len(src_blk_tokens)
                source.extend(src_blk_tokens)
                source2.extend(traj[idx+1: idx+gap].tolist()+src_blk_tokens[:src_blk_length+1-gap_length])

        source.extend(traj[list(range(keep_index[-1], len(traj)))])  ## and locations in the end
        source2.extend(traj[list(range(keep_index[-1], len(traj)))])

        print("source {}".format(source))
        print("source2 {}".format(source2))
        print(len(source), len(source2), num)
        assert len(source) == len(source2)




        return np.array(source), np.array(source2)
