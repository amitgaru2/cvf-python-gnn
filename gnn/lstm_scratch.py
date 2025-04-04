import csv
import datetime

import torch
import torch.nn as nn

from torch_geometric.nn.pool import global_mean_pool
from torch.utils.data import ConcatDataset, DataLoader, random_split, Sampler

from custom_logger import logger
from helpers import CVFConfigForGCNWSuccLSTMDataset


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = "cuda"  # force cuda or exit


class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # self.gcn = GCNConvByHand(input_size, input_size, bias=False, device=device)
        self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.h2o = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # h = self.gcn(x, A)
        # h = torch.relu(h)
        rnn_out, _ = self.rnn(x)
        output = self.h2o(rnn_out)
        output = torch.relu(output)
        output = global_mean_pool(output, torch.zeros(output.size(1)).to(device).long())
        return output

    def fit(self, epochs, dataloader):
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01, weight_decay=0.0001)
        for epoch in range(1, epochs + 1):
            self.train()
            total_loss = 0
            count = 0
            for batch in dataloader:
                x = batch[0]
                y = batch[1]
                y = y.unsqueeze(-1)
                out = self(x[0])
                # print(out.shape, y.shape)
                optimizer.zero_grad()
                loss = criterion(out, y)
                total_loss += loss
                count += 1
                loss.backward()
                optimizer.step()

            logger.info(
                "Training set | Epoch %s | MSE Loss: %s",
                epoch,
                round((total_loss / count).item(), 4),
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


def get_dataset_coll():
    dataset_s_n7 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "star_graph_n7_config_rank_dataset.csv",
        "star_graph_n7_edge_index.json",
    )

    dataset_s_n13 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "star_graph_n13_config_rank_dataset.csv",
        "star_graph_n13_edge_index.json",
    )

    dataset_s_n15 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "star_graph_n15_config_rank_dataset.csv",
        "star_graph_n15_edge_index.json",
    )

    dataset_rr_n7 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "graph_random_regular_graph_n7_d4_config_rank_dataset.csv",
        "graph_random_regular_graph_n7_d4_edge_index.json",
    )

    dataset_rr_n8 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "graph_random_regular_graph_n8_d4_config_rank_dataset.csv",
        "graph_random_regular_graph_n8_d4_edge_index.json",
    )

    dataset_plc_n7 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "graph_powerlaw_cluster_graph_n7_config_rank_dataset.csv",
        "graph_powerlaw_cluster_graph_n7_edge_index.json",
    )

    dataset_plc_n9 = CVFConfigForGCNWSuccLSTMDataset(
        device,
        "graph_powerlaw_cluster_graph_n9_config_rank_dataset.csv",
        "graph_powerlaw_cluster_graph_n9_edge_index.json",
    )

    dataset_coll = [
        dataset_s_n7,
        # dataset_s_n13,
        # dataset_s_n15,
        # dataset_rr_n7,
        # dataset_rr_n8,
        # dataset_plc_n7,
        # dataset_plc_n9,
    ]

    return dataset_coll


def test_model(model, test_datasets):
    f = open(
        f"test_results/test_result_w_succ_diff_nodes_lstm_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}.csv",
        "w",
        newline="",
    )
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Dataset", "Actual", "Predicted"])

    criterion = torch.nn.MSELoss()

    model.eval()

    with torch.no_grad():
        test_concat_datasets = ConcatDataset(test_datasets)
        test_batch_sampler = CustomBatchSampler(test_concat_datasets, batch_size=10240)
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

    f.close()


def main(H=32, batch_size=64, epochs=10):
    dataset_coll = get_dataset_coll()
    D = dataset_coll[0].D
    train_sizes = [int(0.95 * len(ds)) for ds in dataset_coll]
    test_sizes = [len(ds) - trs for ds, trs in zip(dataset_coll, train_sizes)]

    train_test_datasets = [
        random_split(ds, [tr_s, ts])
        for ds, tr_s, ts in zip(dataset_coll, train_sizes, test_sizes)
    ]

    train_datasets = [ds[0] for ds in train_test_datasets]
    test_datasets = [ds[1] for ds in train_test_datasets]

    datasets = ConcatDataset(train_datasets)
    logger.info(f"Train dataset size: {len(datasets):,}")

    batch_sampler = CustomBatchSampler(datasets, batch_size=batch_size)
    dataloader = DataLoader(datasets, batch_sampler=batch_sampler)

    model = SimpleLSTM(D, H, 1).to(device)
    logger.info("Model %s", model)
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    model.fit(epochs=epochs, dataloader=dataloader)

    logger.info("Saving model.")
    torch.save(
        model,
        f"trained_models/lstm_trained_at_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt",
    )

    logger.info("Testing model.")
    test_model(model, test_datasets)


if __name__ == "__main__":
    main()
