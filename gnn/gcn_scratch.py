import time
import random
import datetime
import argparse

import torch
import torch.nn as nn

from zeus.monitor import ZeusMonitor
from torch_geometric.nn.conv import SAGEConv
from torch_geometric.nn.pool import global_mean_pool
from torch.utils.data import ConcatDataset, DataLoader, random_split, Sampler, Subset


from custom_logger import logger

from helpers import (
    CVFConfigForGCNWSuccWEIDataset,
    CVFConfigForGCNWSuccWEIDatasetForMM,
    profile_peak_gpu_memory,
)

monitor = ZeusMonitor(gpu_indices=[0])

device = "cuda"  # force cuda or exit

subset_size = 100_000_000


def get_subset_sampled_loader(train_datasets, batch_size):
    indices = [
        (
            random.sample(range(len(ds)), subset_size)
            if subset_size <= len(ds)
            else range(len(ds))
        )
        for ds in train_datasets
    ]
    subsets = [Subset(ds, ind) for (ds, ind) in zip(train_datasets, indices)]
    datasets = ConcatDataset(subsets)
    batch_sampler = CustomBatchSampler(datasets, batch_size=batch_size)
    dataloader = DataLoader(datasets, batch_sampler=batch_sampler)
    return dataloader


class SimpleGCN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.gcn1 = SAGEConv(input_size, hidden_size, bias=False)
        self.gcn2 = SAGEConv(hidden_size, hidden_size, bias=False)
        self.gcn3 = SAGEConv(hidden_size, hidden_size, bias=False)
        # self.gcn4 = SAGEConv(hidden_size, hidden_size, bias=False)
        # self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=0.3)
        self.out = torch.nn.Linear(hidden_size, output_size)

    def forward(self, x, edge_index):
        h = self.gcn1(x, edge_index)
        h = torch.relu(h)
        h = self.gcn2(h, edge_index)
        h = torch.relu(h)
        h = self.gcn3(h, edge_index)
        # h = torch.relu(h)
        # h = self.gcn4(h, edge_index)
        h = self.dropout(h)
        h = torch.relu(h)
        h = self.out(h)
        h = torch.relu(h)
        h = global_mean_pool(h, torch.zeros(h.size(1)).to(device).long())
        return h

    def validation_model(self, valid_datasets):
        [total_loss, count], [total_matched, dataset_size], accuracy = evaluate(
            self, valid_datasets
        )

        logger.info(
            f"Validation set | MSE loss: {round((total_loss / count).item(), 4)} | Total matched: {total_matched:,} out of {dataset_size:,} (Accuracy: {accuracy:,}%)",
        )

    @profile_peak_gpu_memory
    def fit(self, epochs, train_datasets, valid_datasets, batch_size):
        monitor.begin_window("training")
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005, weight_decay=0.0001)
        dataloader = get_subset_sampled_loader(train_datasets, batch_size)
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.train()
            total_loss = 0
            count = 0
            for batch in dataloader:
                x = batch[0]
                y = batch[1]
                y = y.unsqueeze(-1)
                out = self(x[0], x[1][0])
                optimizer.zero_grad()
                loss = criterion(out, y)
                total_loss += loss
                count += 1
                loss.backward()
                optimizer.step()

            logger.info(
                "Training set | Epoch: %s/%s | MSE Loss: %s | Time taken: %ss",
                epoch,
                epochs,
                round((total_loss / count), 4),
                round(time.time() - start_time, 4),
            )

            self.validation_model(valid_datasets)

            logger.info("\n")

        measurement = monitor.end_window("training")
        logger.info(
            f"Energy usage - Entire training: {measurement.time} s, {measurement.total_energy} J"
        )


class CustomBatchSampler(Sampler):
    def __init__(self, datasets: ConcatDataset, batch_size: int):
        self.datasets = datasets
        self.batch_size = batch_size

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

                yield list(range(last_accessed[turn], last_accessed[turn] + batch_size))

                last_accessed[turn] += batch_size


def get_dataset_coll(program, *graph_names):
    dataset_coll = []

    DatasetKlass = (
        CVFConfigForGCNWSuccWEIDatasetForMM
        if program == "maximal_matching"
        else CVFConfigForGCNWSuccWEIDataset
    )

    for graph_name in graph_names:
        dataset_coll.append(
            DatasetKlass(
                device,
                f"{graph_name}_config_rank_dataset.csv",
                f"{graph_name}_edge_index.json",
                program=program,
            )
        )

    return dataset_coll


def evaluate(model, datasets):
    logger.debug("Evaluating model...")

    model.eval()

    with torch.no_grad():
        criterion = torch.nn.MSELoss()
        batch_sampler = CustomBatchSampler(datasets, batch_size=1024)
        dataloader = DataLoader(datasets, batch_sampler=batch_sampler)

        total_loss = 0
        total_matched = 0
        count = 0
        for batch in dataloader:
            x = batch[0]
            y = batch[1]
            y = y.unsqueeze(-1)
            out = model(x[0], x[1][0])
            loss = criterion(out, y)
            total_loss += loss
            out = torch.round(out)
            matched = (out == y).sum().item()
            total_matched += matched
            count += 1

    return (
        [total_loss, count],
        [total_matched, len(datasets)],
        round(total_matched / len(datasets) * 100, 2),
    )


def test_model(model, test_concat_datasets):

    [total_loss, count], [total_matched, dataset_size], accuracy = evaluate(
        model, test_concat_datasets
    )

    logger.info(
        f"Test set | MSE loss: {round((total_loss / count).item(), 4)} | Total matched: {total_matched:,} out of {dataset_size:,} (Accuracy: {accuracy:,}%)",
    )


def main(program, graph_names, H, batch_size, epochs):
    logger.info(
        "Timestamp: %s | Program: %s | Training with Graphs: %s | Batch size: %s | Epochs: %s | Hidden size: %s.",
        datetime.datetime.now().timestamp(),
        program,
        ", ".join(graph_names),
        batch_size,
        epochs,
        H,
    )
    logger.info("\n")
    dataset_coll = get_dataset_coll(program, *graph_names)
    D = dataset_coll[0].D

    train_valid_test_split = [0.8, 0.1]
    # train_valid_test_split = [0.9, 0.05]
    train_valid_test_split.append(1.0 - sum(train_valid_test_split))

    logger.info("Train, Validation, Test set split: %s", train_valid_test_split)

    train_sizes = [int(train_valid_test_split[0] * len(ds)) for ds in dataset_coll]
    valid_sizes = [int(train_valid_test_split[1] * len(ds)) for ds in dataset_coll]
    test_sizes = [
        len(ds) - trs - vs
        for ds, trs, vs in zip(dataset_coll, train_sizes, valid_sizes)
    ]

    train_test_datasets = [
        random_split(ds, [tr_s, vs, ts])
        for ds, tr_s, vs, ts in zip(dataset_coll, train_sizes, valid_sizes, test_sizes)
    ]

    train_datasets = [ds[0] for ds in train_test_datasets]
    valid_datasets = [ds[1] for ds in train_test_datasets]
    test_datasets = [ds[2] for ds in train_test_datasets]

    datasets = ConcatDataset(train_datasets)

    valid_datasets = ConcatDataset(valid_datasets)

    test_concat_datasets = ConcatDataset(test_datasets)

    logger.info(
        f"Train dataset size: {len(datasets):,}, Subset size: {subset_size:,} | Validation dataset size: {len(valid_datasets):,} | Test dataset size: {len(test_concat_datasets):,}"
    )
    logger.info("\n")

    model = SimpleGCN(D, H, 1).to(device)
    logger.info("Model %s", model)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("\n")
    start_time = time.time()
    model.fit(
        epochs=epochs,
        train_datasets=train_datasets,
        valid_datasets=valid_datasets,
        batch_size=batch_size,
    )
    # model.fit(epochs=epochs, dataloader=dataloader)
    logger.info("\n")
    logger.info(
        "End Training | Total training time taken %ss",
        round(time.time() - start_time, 4),
    )
    logger.info("\n")
    model_name = f"trained_models/gcn_trained_at_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt"
    logger.info("Saving model %s", model_name)
    torch.save(model, model_name)
    logger.info("\n")
    logger.info("Testing model...")
    test_model(model, test_concat_datasets)


def wrap_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument(
        "--graph-names",
        type=str,
        nargs="+",
        help="list of graph names in the 'graphs_dir' or list of number of nodes for implict graphs (if implicit program)",
        required=True,
    )
    parser.add_argument(
        "--logging",
        choices=[
            "INFO",
            "DEBUG",
        ],
        required=False,
    )
    args = parser.parse_args()
    main(
        program=args.program,
        epochs=args.epochs,
        batch_size=args.batch_size,
        H=args.hidden_size,
        graph_names=args.graph_names,
    )


def test_model_for_new_graphs(model_name, program, graph_names):
    def _get_model(model_name):
        model = torch.load(f"trained_models/{model_name}.pt", weights_only=False).to(
            device
        )
        model.eval()
        return model

    logger.info(
        "Testing model: %s | Program: %s | Graphs: %s", model_name, program, graph_names
    )
    model = _get_model(model_name)
    dataset_coll = get_dataset_coll(program, *graph_names)
    concat_datasets = ConcatDataset(dataset_coll)
    test_model(model, concat_datasets)


if __name__ == "__main__":
    wrap_main()
    # test_model_for_new_graphs(
    #     "lstm_trained_at_2025_08_29_11_02",
    #     "graph_coloring",
    #     [
    #         # "star_graph_n7",
    #         # "star_graph_n8",
    #         # "star_graph_n9",
    #         # "star_graph_n10",
    #         # "star_graph_n11",
    #         # "star_graph_n12",
    #         # "graph_powerlaw_cluster_graph_n7",
    #         # "graph_powerlaw_cluster_graph_n6",
    #         # "graph_random_regular_graph_n7_d4",
    #         # "graph_random_regular_graph_n90_d4",
    #         # "graph_powerlaw_cluster_graph_n8"
    #         # "star_graph_n3",
    #         # "star_graph_n14"
    #     ],
    # )
