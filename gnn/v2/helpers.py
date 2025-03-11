import os
import ast
import json

import torch
import gensim
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset


class CVFConfigDataset(Dataset):
    def __init__(
        self,
        program,
        num_nodes,
        dataset_file,
        A_file,
        emb_file,
        embedding_config,
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2", "datasets", program
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.A = torch.tensor(
            json.load(open(os.path.join(dataset_dir, A_file), "r")),
            dtype=torch.long,
        )
        self.num_nodes = num_nodes
        emb_file = os.path.join(dataset_dir, emb_file)
        self.train_embedding(emb_file, **embedding_config)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]

        result = (
            torch.tensor(
                [self.get_embedding_for_config(row["config"])], dtype=torch.float32
            ),
            torch.tensor([[row["M"]]]).float(),
        )

        return result

    def train_embedding(self, emb_file, window=5):
        data = []
        with open(emb_file, "r") as f:
            line = f.readline()
            while line:
                data.append(line.rstrip().split(","))
                line = f.readline()
        self.embedding_model = gensim.models.Word2Vec(
            data, min_count=1, vector_size=self.num_nodes, window=window
        )

    def get_embedding_for_config(self, config):
        return self.embedding_model.wv[config]
