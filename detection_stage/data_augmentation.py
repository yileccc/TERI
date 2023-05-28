import random
import copy
import math
import numpy as np
from constants import *


class Random(object):
    """Randomly pick one data augmentation type every time call"""

    def __init__(self):
        # self.data_augmentation_methods = [Crop(), Mask()]
        self.data_augmentation_methods = [Crop2()]
        print("Total augmentation numbers: ", len(self.data_augmentation_methods))

    def __call__(self, sequence, num, ratio):
        # randint generate int x in range: a <= x <= b
        augment_method_idx = random.randint(0, len(self.data_augmentation_methods) - 1)
        augment_method = self.data_augmentation_methods[augment_method_idx]
        # print(augment_method.__class__.__name__) # debug usage
        return augment_method(sequence, num, ratio)


class Crop(object):
    """Randomly crop a subseq from the original sequence"""

    def __init__(self):
        self.ratio = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

    def __call__(self, traj, num, ratio):
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

    def __call__(self, traj, num, ratio, minstart_idx=1):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(traj)
        # sample_ratio = random.sample(self.ratio, 1)[0]
        # num = random.sample(self.num, 1)[0]
        ratio_each = ratio/num

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
        self.ratio = [0.2, 0.3, 0.4, 0.5, 0.6]

    def __call__(self, sequence, num, ratio):
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
        self.num = [1, 2, 3, 4]
        self.ratio = [0.2, 0.3, 0.4, 0.5, 0.6]


    def __call__(self, traj, num, ratio, minstart_idx=1):
        # make a deep copy to avoid original sequence be modified
        copied_sequence = copy.deepcopy(traj)
        # sample_ratio = random.sample(self.ratio, 1)[0]
        # num = random.sample(self.num, 1)[0]
        ratio_each = ratio/num

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

