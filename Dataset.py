import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import OrdinalEncoder
import pickle

class CustomDataset(Dataset):

    def __init__(self, data, date_col, seq_id, cat_features, target, seq_len, encoder=False):
        super(CustomDataset, self).__init__()
        self.date_col = date_col
        self.cat_features = cat_features
        self.seq_len = seq_len
        self.target = target
        self.data = data
        self.seq_id = seq_id
        self.uniq_vals = self.data[seq_id].unique()
        #preprocessing
        if not encoder:
            encoder = OrdinalEncoder()
            encoder.fit(self.data[self.cat_features])
            with open('encoder_dir/encoder.pkl', 'wb') as f:
                pickle.dump(encoder, f)
        else:
            with open('encoder_dir/encoder.pkl', 'rb') as f:
                encoder = pickle.load(f)
        self.data[self.cat_features] = encoder.transform(self.data[self.cat_features])


    def __len__(self):
        return self.data[self.seq_id].nunique()

    def __getitem__(self, idx):
        data = self.data.groupby(by=self.seq_id).get_group(self.uniq_vals[idx]).drop(columns=self.target)
        data.drop(columns=self.seq_id, inplace=True)
        return data.drop(columns=self.cat_features), data[self.cat_features + [self.date_col]], self.data[self.target]

    def collect_fn(self, batch):

        #list of shape batch_size and data from getitem in tuple
        num_data, cat_data, y = zip(*batch)

        num_data_t = []
        for i in num_data:
            num_data_t.append(torch.tensor(i.sort_values(by=self.date_col, ascending=True).drop(columns=self.date_col).values))

        cat_data_t = []
        for i in cat_data:
            cat_data_t.append(torch.tensor(i.sort_values(by=self.date_col, ascending=True).drop(columns=self.date_col).values))

        for i in range(len(num_data)):
            num_data_t[i] = pad_sequences(num_data_t[i].T, maxlen=self.seq_len, dtype='float32',
                                     padding='pre', truncating='pre').T
            cat_data_t[i] = pad_sequences(cat_data_t[i].T, maxlen=self.seq_len, dtype='float32',
                                          padding='pre', truncating='pre').T
        num_data_t = torch.tensor(num_data_t, dtype=torch.float32)
        cat_data_t = torch.tensor(cat_data_t, dtype=torch.float32) # torch.int8
        num_mask = num_data_t == 0
        cat_mask = num_data_t == 0

        y = torch.tensor(y, dtype=torch.float32) # torch.uint8


        return num_data_t, cat_data_t, y, (num_mask, cat_mask)



data = pd.read_csv('data/data_lstm.csv', parse_dates=['ddog']).drop(columns='Unnamed: 0')
num_cols = ['num_1', 'num_2', 'num_3']
cat_cols = ['cat_1', 'cat_2']
target_col = 'target'
dataset = CustomDataset(data, date_col='ddog', seq_id='id', cat_features=cat_cols, target=target_col, seq_len=7)
dataloader = DataLoader(dataset, collate_fn=dataset.collect_fn, shuffle=True, batch_size=2)
for i in dataloader:
    pass