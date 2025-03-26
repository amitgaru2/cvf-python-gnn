import csv
import datetime
import itertools

import torch

from skorch import NeuralNet
from torch_geometric.nn import GCNConv
from torch_geometric.nn.pool import global_mean_pool
from torch.utils.data import ConcatDataset, Sampler, DataLoader, random_split

from helpers import CVFConfigForGCNGridSearchDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device", device)


dataset_rr_n4 = CVFConfigForGCNGridSearchDataset(
    device,
    "graph_random_regular_graph_n4_d3_config_rank_dataset.csv",
    "graph_random_regular_graph_n4_d3_edge_index.json",
)

dataset_rr_n5 = CVFConfigForGCNGridSearchDataset(
    device,
    "graph_random_regular_graph_n5_d4_config_rank_dataset.csv",
    "graph_random_regular_graph_n5_d4_edge_index.json",
)

dataset_rr_n6 = CVFConfigForGCNGridSearchDataset(
    device,
    "graph_random_regular_graph_n6_d3_config_rank_dataset.csv",
    "graph_random_regular_graph_n6_d3_edge_index.json",
)

dataset_rr_n7 = CVFConfigForGCNGridSearchDataset(
    device,
    "graph_random_regular_graph_n7_d4_config_rank_dataset.csv",
    "graph_random_regular_graph_n7_d4_edge_index.json",
)

dataset_rr_n8 = CVFConfigForGCNGridSearchDataset(
    device,
    "graph_random_regular_graph_n8_d4_config_rank_dataset.csv",
    "graph_random_regular_graph_n8_d4_edge_index.json",
)

batch_size = 64

dataset_coll = [
    dataset_rr_n4,
    dataset_rr_n5,
    dataset_rr_n6,
    dataset_rr_n7,
    dataset_rr_n8,
]


train_sizes = [int(0.9 * len(ds)) for ds in dataset_coll]
test_sizes = [len(ds) - trs for ds, trs in zip(dataset_coll, train_sizes)]

train_test_datasets = [
    random_split(ds, [tr_s, ts])
    for ds, tr_s, ts in zip(dataset_coll, train_sizes, test_sizes)
]

train_datasets = [ds[0] for ds in train_test_datasets]
test_datasets = [ds[1] for ds in train_test_datasets]

datasets = ConcatDataset(train_datasets)


class CustomBatchSampler(Sampler):
    def __init__(self, datasets: ConcatDataset, batch_size: int):
        self.datasets = datasets
        self.batch_size = batch_size

    @property
    def module(self):
        return self._module

    @module.setter
    def module(self, val):
        self._module = val

    def __iter__(self):
        last_accessed = [0] + self.datasets.cumulative_sizes[:]
        end_loop = [False for _ in range(len(self.datasets.datasets))]

        while not all(end_loop):
            for turn in range(len(self.datasets.datasets)):
                if end_loop[turn]:
                    continue

                batch_size = self.batch_size
                if (
                    last_accessed[turn] + batch_size
                    >= self.datasets.cumulative_sizes[turn]
                ):
                    batch_size = (
                        self.datasets.cumulative_sizes[turn] - last_accessed[turn]
                    )
                    end_loop[turn] = True

                # currently explicitly setting edge index before yielding
                # TODO: find a better way to do it
                self.module.edge_index = self.datasets.datasets[turn].dataset.edge_index

                yield list(range(last_accessed[turn], last_accessed[turn] + batch_size))

                last_accessed[turn] += batch_size


batch_sampler = CustomBatchSampler(datasets, batch_size=batch_size)
dataloader = DataLoader(datasets, batch_sampler=batch_sampler)


class VanillaGNN(torch.nn.Module):
    def __init__(self, dim_in, dim_h, dim_out):
        super().__init__()
        self.gcn1 = GCNConv(dim_in, dim_h)
        self.gcn2 = GCNConv(dim_h, dim_h)
        self.out = torch.nn.Linear(dim_h, dim_out)

    @property
    def edge_index(self):
        return self._edge_index

    @edge_index.setter
    def edge_index(self, val):
        self._edge_index = val

    def forward(self, x):
        h = self.gcn1(x, self.edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, self.edge_index)
        h = torch.relu(h)
        h = self.out(h)
        h = torch.relu(h)
        h = global_mean_pool(h, torch.zeros(h.size(1)).to(device).long())
        return h

    def fit(self, epochs):
        dataloader.batch_sampler.module = self
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.001)
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0
            count = 0
            for batch in dataloader:
                x = batch[0]
                y = batch[1]
                optimizer.zero_grad()
                out = self(x)
                loss = criterion(out, y)
                total_loss += loss
                count += 1
                loss.backward()
                optimizer.step()

            print(
                "Training set | Epoch",
                epoch,
                "| Loss:",
                round((total_loss / count).item(), 4),
            )


test_concat_datasets = ConcatDataset(test_datasets)
batch_sampler = CustomBatchSampler(test_concat_datasets, batch_size=batch_size)
test_dataloader = DataLoader(test_concat_datasets, batch_sampler=batch_sampler)


def get_test_loss(model):
    torch.no_grad()

    criterion = torch.nn.MSELoss()
    # test_dataloader.batch_sampler.module = gnn

    count = 0
    total_loss = 0
    for batch in test_dataloader:
        x = batch[0]
        y = batch[1]
        out = torch.FloatTensor(model.predict(x)).to(device)
        loss = criterion(out, y)
        total_loss += loss
        count += 1

    loss = total_loss / count
    return loss


class CustomNeuralNet(NeuralNet):
    def fit_loop(self, X, y=None, epochs=None, **fit_params):
        # super().fit_loop()
        """The proper fit loop.

        Contains the logic of what actually happens during the fit
        loop.

        Parameters
        ----------
        X : input data, compatible with skorch.dataset.Dataset
          By default, you should be able to pass:

            * numpy arrays
            * torch tensors
            * pandas DataFrame or Series
            * scipy sparse CSR matrices
            * a dictionary of the former three
            * a list/tuple of the former three
            * a Dataset

          If this doesn't work with your data, you have to pass a
          ``Dataset`` that can deal with the data.

        y : target data, compatible with skorch.dataset.Dataset
          The same data types as for ``X`` are supported. If your X is
          a Dataset that contains the target, ``y`` may be set to
          None.

        epochs : int or None (default=None)
          If int, train for this number of epochs; if None, use
          ``self.max_epochs``.

        **fit_params : dict
          Additional parameters passed to the ``forward`` method of
          the module and to the ``self.train_split`` call.

        """
        self.check_data(X, y)
        self.check_training_readiness()
        epochs = epochs if epochs is not None else self.max_epochs

        on_epoch_kwargs = {
            "dataset_train": datasets,
            "dataset_valid": test_concat_datasets,
        }
        dataloader.batch_sampler.module = self.module_
        iterator_train = dataloader
        test_dataloader.batch_sampler.module = self.module_
        iterator_valid = test_dataloader

        for _ in range(epochs):
            self.notify("on_epoch_begin", **on_epoch_kwargs)

            self.run_single_epoch(
                iterator_train,
                training=True,
                prefix="train",
                step_fn=self.train_step,
                **fit_params,
            )

            self.run_single_epoch(
                iterator_valid,
                training=False,
                prefix="valid",
                step_fn=self.validation_step,
                **fit_params,
            )

            self.notify("on_epoch_end", **on_epoch_kwargs)

        return self


params = {
    "lr": [0.01, 0.001, 0.05],
    "batch_size": [32, 128, 512],
    "max_epochs": [25, 50, 100],
    "optimizer": [torch.optim.SGD, torch.optim.Adam],
    "module__dim_h": [16, 32, 64, 128],
    "optimizer__weight_decay": [0.01, 0.001, 0.05],
}


param_combinations = list(itertools.product(*params.values()))

param_combinations_dict = [
    dict(zip(params.keys(), combination)) for combination in param_combinations
]

avg_for = 3

result_filename_suffix = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
f = open(f"grid_search_results_{result_filename_suffix}.csv", "w", newline="")
csv_writer = csv.writer(f)
csv_writer.writerow(["params", f"avg loss ({avg_for})"])

for params in param_combinations_dict:
    avg_loss = 0.0
    for i in range(avg_for):
        net = CustomNeuralNet(
            VanillaGNN,
            train_split=None,
            device=device,
            criterion=torch.nn.MSELoss,
            # optimizer=torch.optim.Adam,
            # optimizer__weight_decay=0.01,
            module__dim_in=1,
            # module__dim_h=32,
            module__dim_out=1,
            **params,
        )
        net.fit(datasets, y=None)
        avg_loss += get_test_loss(net)

    csv_writer.writerow([params, (avg_loss / avg_for).detach().item()])
    # print(params, avg_loss / avg_for)
