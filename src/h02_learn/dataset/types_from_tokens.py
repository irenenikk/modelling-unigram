import torch
from torch.utils.data import Dataset
from tqdm import tqdm

class TypesFromTokensDataset(Dataset):

    def __init__(self, token_dataloader):
        self.types = {}
        self.type_indices = []
        self.type_weights = []
        self.type_ids = []
        self.type_words = []
        for batch_x, batch_y, weights, ids, tokens in \
                                tqdm(token_dataloader, total=len(token_dataloader), \
                                desc='Building type dataset from tokens', mininterval=.2):
            for i, x in enumerate(batch_x):
                word = tokens[i]
                if word in self.types:
                    continue
                self.types[word] = True
                # this is reconstructing the original word indices from the batch
                # include the last character, which is usually the end of word marker
                last_char_index = torch.nonzero(x == 0).flatten()[0].item() - 1 \
                                    if len(torch.nonzero(x == 0)) > 0 else -1
                last_char = batch_y[i][last_char_index]
                self.type_indices.append(torch.cat((x[x > 0], torch.unsqueeze(last_char, 0))))
                self.type_weights.append(torch.unsqueeze(weights[i], 0))
                self.type_ids.append(ids[i])
                self.type_words.append(word)

    def __len__(self):
        return len(self.type_indices)

    def __getitem__(self, index):
        return self.type_indices[index], self.type_weights[index], \
                self.type_ids[index], self.type_words[index]
