import random
import pdb
import re
import numpy as np

import torch
from torch.utils.data.dataset import Dataset


class TISDataset(Dataset):
    def read_dna_file(self, file_location, class_id):
        current_file = open(file_location)
        dna_list = []

        for line in current_file:
            stripped_line = line.strip()
            # Ignore empty lines
            if stripped_line == '':
                pass
            else:
                if len(stripped_line) != 203:
                    print('len is different', len(stripped_line))
                dna_list.append((stripped_line, class_id))
        print(class_id, len(dna_list))
        return dna_list

    def __init__(self, data_loc_list, random_mask=0):
        # Initialization
        self.dna_list = []
        for data_loc in data_loc_list:
            if data_loc[-3:] == 'pos':
                self.dna_list.extend(self.read_dna_file(data_loc, 1))
            else:
                self.dna_list.extend(self.read_dna_file(data_loc, 0))
        self.data_len = len(self.dna_list)
        print('Dataset init with', self.data_len, 'samples')
        self.random_mask = random_mask

    def __getitem__(self, index):
        # Read data
        dna_data, label = self.dna_list[index]
        # One hot
        dna_data = dna_data.replace('A', '0')
        dna_data = dna_data.replace('G', '1')
        dna_data = dna_data.replace('C', '2')
        dna_data = dna_data.replace('T', '3')
        dna_data = dna_data.replace('N', '0')
        # A [1 0 0 0]
        # G [0 1 0 0]
        # C [0 0 1 0]
        # T [0 0 0 1]

        dna_data = np.asarray([int(digit) for digit in dna_data])
        one_hot_dna_data = np.zeros((len(dna_data), 4), dtype=int)
        one_hot_dna_data[np.arange(dna_data.size), dna_data] = 1

        # Random masking if not train
        if self.random_mask != 0:
            for mask_cnt in range(random.randint(0, self.random_mask)):
                one_hot_dna_data[random.randint(0, 202)] = [0, 0, 0, 0]

        # Numpy to tensor
        dna_data_as_ten = torch.from_numpy(one_hot_dna_data).float()
        dna_data_as_ten.unsqueeze_(dim=0)

        return dna_data_as_ten, label, index

    def __len__(self):
        return self.data_len


if __name__ == '__main__':
    tis_dataset = TISDataset(['../data/GRC-M.pos', '../data/GRC-M.neg'])
    data_as_ten, label, index = tis_dataset[0]
    print(data_as_ten.shape)
    print('label:', label, 'index:', index)
