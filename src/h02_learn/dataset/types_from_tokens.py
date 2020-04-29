import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import numpy as np

class TypesFromTokensDataset(Dataset):

    def __init__(self, token_dataloader, alphabet):
        self.types = {}
        self.type_indices = []
        self.type_weights = []
        self.type_ids = []
        for batch_x, batch_y, weights, ids in tqdm(token_dataloader, total=len(token_dataloader), \
                                desc='Building type dataset from tokens', mininterval=.2):
            for i, x in enumerate(batch_x):
                word = ''.join(alphabet.idx2word(x[1:]))
                if word in self.types:
                    continue
                self.types[word] = True
                # this is reconstructing the original word indices from the batch
                # include the last character, which is usually the end of word marker
                last_char = batch_y[i][np.argmax(x == 0) - 1]
                self.type_indices.append(torch.cat((x[x > 0], torch.unsqueeze(last_char, 0))))
                self.type_weights.append(torch.unsqueeze(weights[i], 0))
                self.type_ids.append(ids[i])

    def __len__(self):
        return len(self.type_indices)

    def __getitem__(self, index):
        return self.type_indices[index], self.type_weights[index], self.type_ids[index]
