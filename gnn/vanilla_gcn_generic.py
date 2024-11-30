import csv
import itertools

import torch

from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch.utils.data import DataLoader, random_split

from metrics import CustomR2Score
from helpers import CVFConfigForGCNDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)

color_mapping_categories = 15

# dataset_n1 = CVFConfigForGCNDataset(
#     device,
#     "graph_1_config_rank_dataset.csv",
#     "graph_1_edge_index.json",
#     color_mapping_categories,
# )

dataset_pl_n12 = CVFConfigForGCNDataset(
    device,
    "graph_powerlaw_cluster_graph_n12_config_rank_dataset.csv",
    "graph_powerlaw_cluster_graph_n12_edge_index.json",
    color_mapping_categories,
)

batch_size = 64

# dataset_coll = [dataset_pl_n5, dataset_pl_n6, dataset_pl_n7, dataset_pl_n8]
dataset_coll = [dataset_pl_n12]
train_dataloader_coll = []
test_dataloader_coll = []

for dataset in dataset_coll:
    train_size = int(0.75 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    train_dataloader_coll.append(train_loader)
    test_dataloader_coll.append(test_loader)

train_dataloader_coll_iter = [iter(i) for i in train_dataloader_coll]


def generate_batch():
    end_loop = [False for _ in range(len(train_dataloader_coll))]
    while not any(end_loop):
        for di, data_loader in enumerate(train_dataloader_coll_iter):
            if end_loop[di]:
                continue
            try:
                batch = next(data_loader)
            except StopIteration:
                end_loop[di] = True
                continue
            yield batch, di


print("Number of batches:", [len(i) for i in train_dataloader_coll])


csvfile = open("pl_n12__epoch_vs_acc.csv", "w")
writer = csv.DictWriter(csvfile, fieldnames=["epoch", "accuracy"])
writer.writeheader()


class VanillaGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_h)
        self.out = torch.nn.Linear(dim_h, dim_out)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = self.out(h)
        h = torch.relu(h)
        h = global_mean_pool(h, torch.zeros(h.size(1)).to(device).long())
        return h

    def fit(self, epochs):
        # criterion = torch.nn.CrossEntropyLoss()
        # criterion = torch.nn.MSELoss()
        criterion_sum = torch.nn.MSELoss(reduction="sum")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=5e-4)
        self.train()
        metric = CustomR2Score()
        # edge_index = dataset.edge_index.t().to(device)
        dataloaders = itertools.tee(generate_batch(), epochs)
        for epoch in range(1, epochs + 1):
            max_loss = -1
            total_loss = 0
            count = 0
            for batch, di in dataloaders[epoch - 1]:
                x = batch[0].to(device)
                y = batch[1].to(device)
                y = y.unsqueeze(0).reshape(-1, 1, 1).float()
                optimizer.zero_grad()
                out = self(x, dataset_coll[di].edge_index)
                # print("output", out, "y", y)
                loss = criterion_sum(out, y)
                metric.update(y, out)
                # loss_sum = cr(out, y)
                avg_loss = loss / len(batch[1])
                if avg_loss > max_loss:
                    max_loss = avg_loss
                total_loss += loss
                count += len(batch[1])
                loss.backward()
                optimizer.step()

            if count > 0:
                accuracy = round(metric.compute() * 100, 4)
                if accuracy < 0:
                    accuracy = 0
                writer.writerow({"epoch": epoch, "accuracy": accuracy})
                print(
                    "Epoch",
                    epoch,
                    "| Loss:",
                    round((total_loss / count).item(), 4),
                    "| Max Loss:",
                    round(max_loss.item(), 4),
                    "| Accuracy:",
                    round(metric.compute() * 100, 4),
                )


if __name__ == "__main__":
    gnn = VanillaGNN(color_mapping_categories, 64, 1).to(device)
    print(gnn)
    gnn.fit(epochs=50)
