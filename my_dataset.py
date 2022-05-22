import json, math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.data.sampler as samplers
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from text_tools import load_cmudict, symbol_to_id, text_to_id

class SortedSampler(samplers.Sampler):
    def __init__(self, data, sort_key):
        super().__init__(data)
        self.data = data
        self.sort_key = sort_key
        zip_ = [(i, self.sort_key(row)) for i, row in enumerate(self.data)]
        zip_ = sorted(zip_, key=lambda r: r[1], reverse=True)
        self.sorted_indexes = [item[0] for item in zip_]
    
    def __iter__(self):
        return iter(self.sorted_indexes)
    
    def __len__(self):
        return len(self.data)

class BucketBatchSampler(samplers.BatchSampler):
    def __init__(
        self,
        sampler,
        batch_size,
        drop_last,
        sort_key,
        bucket_size_multiplier,
    ):
        super().__init__(sampler, batch_size, drop_last)
        self.sort_key = sort_key
        self.bucket_sampler = samplers.BatchSampler(
            sampler, min(batch_size * bucket_size_multiplier, len(sampler)), False
        )
    
    def __iter__(self):
        for bucket in self.bucket_sampler:
            sorted_sampler = SortedSampler(bucket, self.sort_key)
            for batch in samplers.SubsetRandomSampler(
                list(
                    samplers.BatchSampler(
                        sorted_sampler, self.batch_size, self.drop_last
                    )
                )
            ):
                yield [bucket[i] for i in batch]
    
    def __len__(self):
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return math.ceil(len(self.sampler) / self.batch_size)

class TTSDataset(Dataset):
    def __init__(self, root, text_path):
        self.root = Path(root)

        with open(self.root / 'train.json') as file:
            metadata = json.load(file)
            self.metadata = [Path(path) for _, path in metadata]
        
        with open(self.root / 'lengths.json') as file:
            lengths = json.load(file)
            self.lengths = [lengths[path.stem] for path in self.metadata]
        
        self.index_longest_mel = np.argmax(self.lengths)

        self.cmudict = load_cmudict()

        train_set = {path.stem for path in self.metadata}
        with open(text_path) as file:
            text = (line.stript().split('|') for line in file)
            self.text = {
                key: transcript for key, _, transcript in text if key in train_set
            }
    
    def sort_key(self, index):
        return self.lengths[index]
    
    def __len__(self):
        return len(self.metadata)
    
    def __getitem__(self, index):
        path = self.root / self.metadata[index]
        mel = np.load(path.with_suffix('.mel.npy'))

        text = text_to_id(self.text[path.stem], self.cmudict)

        return (
            torch.Tensor(mel).transpose_(0, 1),
            torch.LongTensor(text),
            index == self.index_longest_mel,
        )

def pad_collate(batch, reduction_factor=2):
    mels, texts, attn_flag = zip(*batch)
    mels = list(mels)
    texts = list(texts)

    if len(mels[0]) % reduction_factor != 0:
        mels[0] = F.pad(mels[0], (0, 0, 0, reduction_factor - 1))
    
    mels = pad_sequence(mels, batch_first=True)
    texts = pad_sequence(texts, batch_first=True, padding_value=symbol_to_id['_'])

    attn_flag = [i for i, flag in enumerate(attn_flag) if flag]

    return mels.transpose_(1, 2), texts, mel_lengths, text_lengths, attn_flag
