import csv
import random
import time
import datetime
import argparse

import torch
import torch.nn as nn

from torch_geometric.nn.pool import global_mean_pool
from torch.utils.data import ConcatDataset, DataLoader, random_split, Sampler, Subset

from custom_logger import logger
from helpers import (
    CVFConfigForGCNWSuccLSTMDataset,
    CVFConfigForGCNWSuccLSTMDatasetForMM,
)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda"  # force cuda or exit

subset_size = 200_000


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super().__init__()
        embed_size = 64
        self.embedding = nn.Linear(input_size, embed_size)
        self.lstm = nn.GRU(
            embed_size, hidden_size, num_layers=num_layers, batch_first=True
        )
        self.norm = nn.LayerNorm(hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)
        output = self.norm(lstm_out)
        output = self.h2o(output)
        output = torch.relu(output)
        output = global_mean_pool(output, torch.zeros(output.size(1)).to(device).long())
        return output

    def fit(self, epochs, train_datasets, batch_size):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        for epoch in range(1, epochs + 1):
            start_time = time.time()
            self.train()
            total_loss = 0
            count = 0
            dataloader = get_subset_sampled_loader(train_datasets, batch_size)
            for batch in dataloader:
                x = batch[0]
                y = batch[1]
                y = y.unsqueeze(-1)
                out = self(x[0])
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
                round((total_loss / count).item(), 4),
                round(time.time() - start_time, 4),
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
        CVFConfigForGCNWSuccLSTMDatasetForMM
        if program == "maximal_matching"
        else CVFConfigForGCNWSuccLSTMDataset
    )
    for graph_name in graph_names:
        dataset_coll.append(
            DatasetKlass(
                device, f"{graph_name}_config_rank_dataset.csv", program=program
            )
        )

    return dataset_coll


def test_model(model, test_concat_datasets, save_result=False):
    if save_result:
        f = open(
            f"test_results/test_result_w_succ_diff_nodes_lstm_script_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.csv",
            "w",
            newline="",
        )
        csv_writer = csv.writer(f)
        csv_writer.writerow(["Dataset", "Actual", "Predicted"])

    criterion = torch.nn.MSELoss()

    model.eval()

    with torch.no_grad():
        # test_concat_datasets = ConcatDataset(test_datasets)
        test_batch_sampler = CustomBatchSampler(test_concat_datasets, batch_size=1)
        test_dataloader = DataLoader(
            test_concat_datasets, batch_sampler=test_batch_sampler
        )

        total_loss = 0
        total_matched = 0
        count = 0
        for batch in test_dataloader:
            x = batch[0]
            y = batch[1]
            y = y.unsqueeze(-1)
            out = model(x[0])
            # print(y, out)
            if save_result:
                csv_writer.writerows(
                    (i, j.item(), k.item())
                    for (i, j, k) in zip(
                        x[1], y.detach().cpu().numpy(), out.detach().cpu().numpy()
                    )
                )
            loss = criterion(out, y)
            total_loss += loss
            out = torch.round(out)
            matched = (out == y).sum().item()
            total_matched += matched
            count += 1

        logger.info(
            f"Test set | MSE loss: {round((total_loss / count).item(), 4)} | Total matched: {total_matched:,} out of {len(test_concat_datasets):,} (Accuracy: {round(total_matched / len(test_concat_datasets) * 100, 2):,}%)",
        )

    if save_result:
        f.close()


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


def main(program, graph_names, H, batch_size, epochs, num_layers):
    logger.info(
        "Timestamp: %s | Program: %s | Training with Graphs: %s | Batch size: %s | Epochs: %s | Hidden size: %s | Num layers: %s.",
        datetime.datetime.now().timestamp(),
        program,
        ", ".join(graph_names),
        batch_size,
        epochs,
        H,
        num_layers,
    )
    logger.info("\n")
    dataset_coll = get_dataset_coll(program, *graph_names)
    D = dataset_coll[0].D
    train_sizes = [int(0.98 * len(ds)) for ds in dataset_coll]
    test_sizes = [len(ds) - trs for ds, trs in zip(dataset_coll, train_sizes)]

    train_test_datasets = [
        random_split(ds, [tr_s, ts])
        for ds, tr_s, ts in zip(dataset_coll, train_sizes, test_sizes)
    ]

    train_datasets = [ds[0] for ds in train_test_datasets]
    test_datasets = [ds[1] for ds in train_test_datasets]

    datasets = ConcatDataset(train_datasets)

    test_concat_datasets = ConcatDataset(test_datasets)

    logger.info(
        f"Train dataset size: {len(datasets):,}, Subset size: {subset_size:,} | Test dataset size: {len(test_concat_datasets):,}"
    )
    logger.info("\n")

    model = SimpleLSTM(D, H, 1, num_layers=num_layers).to(device)
    logger.info("Model %s", model)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("\n")
    start_time = time.time()
    model.fit(epochs=epochs, train_datasets=train_datasets, batch_size=batch_size)
    logger.info("\n")
    logger.info(
        "End Training | Total training time taken %ss",
        round(time.time() - start_time, 4),
    )
    logger.info("\n")
    model_name = f"trained_models/lstm_trained_at_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt"
    logger.info("Saving model %s", model_name)
    torch.save(model, model_name)

    logger.info("\n")
    logger.info("Testing model...")
    test_model(model, test_concat_datasets, save_result=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program", type=str, default="coloring")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=1)
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
        num_layers=args.num_layers,
        graph_names=args.graph_names,
    )
