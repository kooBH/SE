import os, glob
import torch

class Dataset(torch.utils.data.Dataset):
    def __init__(self, root):
        self.root = hp.data.root

        # TODO : modify path
        self.list_data = glob.glob(os.path.join(root,"*.wav"))

    def __getitem__(self, index):
        data_item = self.list_data[index]

        # Process data_item if necessary.

        data = data_item

        return data

    def __len__(self):
        return len(self.list_data)


