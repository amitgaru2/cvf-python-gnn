import os
import sys
import ast
import random

import torch
import pandas as pd
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader

sys.path.append(os.path.join(os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2"))


class CVFConfigForGCNWSuccLSTMDataset(Dataset):
    def __init__(
        self,
        device,
        dataset_file,
        graph_encoding,
        program="coloring",
    ) -> None:
        dataset_dir = os.path.join(
            os.getenv("CVF_PROJECT_DIR", ""),
            "cvf-analysis",
            "v2",
            "datasets",
            program,
        )
        self.data = pd.read_csv(os.path.join(dataset_dir, dataset_file))
        self.device = device
        self.dataset_name = dataset_file.split("_config_rank_dataset.csv")[0]
        self.D = 7  # input dimension
        self.graph_encoding = graph_encoding
        self.total_succ_select = 6

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.loc[idx]
        config = [i for i in ast.literal_eval(row["config"])]
        succ = [i for i in ast.literal_eval(row["succ"])]
        if succ:
            if len(succ) >= self.total_succ_select:
                indxes = random.sample(range(len(succ)), self.total_succ_select)
                # indxes = list(range(self.total_succ_select))
            else:
                indxes = list(range(len(succ)))
            remaining_succ_count = self.total_succ_select - len(indxes)
            selected_succ = [succ[i] for i in indxes]
            for _ in range(remaining_succ_count):
                selected_succ.append([-1 for _ in range(self.D)])
            succ = torch.FloatTensor(selected_succ).to(self.device)
        else:
            # succ = torch.zeros(self.total_succ_select, len(config)).to(self.device)
            succ = torch.full((self.total_succ_select, self.D), -1).to(self.device)
        config = torch.FloatTensor([config]).to(self.device)
        labels = torch.FloatTensor([row["rank"]]).to(self.device)
        result = (
            torch.cat((self.graph_encoding, config, succ)),
            self.dataset_name,
        ), labels
        # result = config, labels
        return result

    def __repr__(self):
        return f"{self.__class__.__name__} {self.dataset_name}"


if __name__ == "__main__":
    device = "cuda"
    dataset = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "star_graph_n7_config_rank_dataset.csv",
        graph_encoding=F.one_hot(torch.tensor([0]), num_classes=7).to(device),
    )

    loader = DataLoader(dataset, batch_size=2, shuffle=False)

    for batch in loader:
        x = batch[0]
        y = batch[1]
        print(x[0].shape)
        # break
