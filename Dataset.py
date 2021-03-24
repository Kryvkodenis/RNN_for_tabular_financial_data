import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):

    def __init__(self, data, seq_id, cat_features, target, seq_len):
        super(CustomDataset, self).__init__()
        self.cat_features = cat_features
        self.seq_len = seq_len
        self.target = target
        self.data = data
        self.seq_id = seq_id
        self.uniq_vals = self.data[self.seq_id].unique()
        #preprocessing


    def __len__(self):
        return self.data[self.seq_id].nunique()

    def __getitem__(self, idx):
        data = self.data.groupby(by=self.seq_id).get_group(self.uniq_vals[idx]).drop(columns=self.target)
        data.drop(columns='id', inplace=True)
        return data.drop(columns=self.cat_features), data[[self.cat_features] + ['ddog']], self.data[self.target]

    def collect_fn(self, batch):

        #list of shape batch_size and data from getitem in tuple
        num_data, cat_data, y = zip(*batch)

        num_data_t = []
        for i in num_data:
            num_data_t.append(torch.tensor(i.sort_values(by='ddog').values))

        cat_data_t = []
        for i in num_data:
            cat_data_t.append(torch.tensor(i.sort_values(by='ddog').values))

        for i in range(len(num_data)):
            num_data_t[i] = pad_sequences(num_data_t[i].T, maxlen=self.seq_len, dtype='float32',
                                     padding='post', truncating='post').T
            cat_data_t[i] = pad_sequences(cat_data_t[i].T, maxlen=self.seq_len, dtype='float32',
                                          padding='post', truncating='post').T
        num_data_t = torch.tensor(num_data_t, dtype=torch.float32)
        cat_data_t = torch.tensor(cat_data_t, dtype=torch.float32) # torch.int8
        num_mask = num_data_t == 0
        cat_mask = num_data_t == 0

        y = torch.tensor(y, dtype=torch.float32) # torch.uint8


        return num_data_t, cat_data_t, y, (num_mask, cat_mask)



#data = pd.read_csv('data/data_lstm.csv').drop(columns='Unnamed: 0')
#dataset = CustomDataset(data, 'id', cat_features='1', target='2', seq_len=6)
#dataloader = DataLoader(dataset, collate_fn=dataset.collect_fn, shuffle=True, batch_size=5)
#for i in dataloader:
#    pass