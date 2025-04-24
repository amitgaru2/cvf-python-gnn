import csv
import time
import datetime

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import ConcatDataset, DataLoader, random_split


from dataset import (
    logger,
    CVFConfigForTransformerDataset,
    CVFConfigForTransformerTestDatasetWName,
)


device = "cuda"


def generate_local_mask(seq_len):
    mask = torch.full((seq_len, seq_len), float("-inf"))
    for i in range(1, seq_len):
        mask[i, i - 1] = 0  # Only allow attending to the previous token
    mask[0, 0] = 0  # Optional: allow first token to attend to itself
    return mask


def get_dataset_coll(batch_size):
    dataset_s_n7 = CVFConfigForTransformerDataset(
        device,
        "star_graph_n7",
        "star_graph_n7_pt_adj_list.txt",
        "star_graph_n7_config_rank_dataset.csv",
        D=7,
    )

    dataset_rr_n7 = CVFConfigForTransformerDataset(
        device,
        "graph_random_regular_graph_n7_d4",
        "graph_random_regular_graph_n7_d4_pt_adj_list.txt",
        "graph_random_regular_graph_n7_d4_config_rank_dataset.csv",
        D=7,
    )

    dataset_plc_n7 = CVFConfigForTransformerDataset(
        device,
        "graph_powerlaw_cluster_graph_n7",
        "graph_powerlaw_cluster_graph_n7_pt_adj_list.txt",
        "graph_powerlaw_cluster_graph_n7_config_rank_dataset.csv",
        D=7,
    )

    dataset_coll = [
        dataset_s_n7,
        # dataset_rr_n7,
        # dataset_plc_n7,
    ]

    logger.info(f"Datasets: {dataset_coll}")

    train_sizes = [int(0.95 * len(ds)) for ds in dataset_coll]
    test_sizes = [len(ds) - trs for ds, trs in zip(dataset_coll, train_sizes)]

    train_test_datasets = [
        random_split(ds, [tr_s, ts])
        for ds, tr_s, ts in zip(dataset_coll, train_sizes, test_sizes)
    ]

    train_datasets = [ds[0] for ds in train_test_datasets]
    # test_datasets = [ds[1] for ds in train_test_datasets]

    datasets = ConcatDataset(train_datasets)
    logger.info(f"Train Dataset size: {len(datasets):,}")

    loader = DataLoader(datasets, batch_size=batch_size)

    sequence_length = max(d.sequence_length for d in dataset_coll)
    logger.info(f"Max sequence length: {sequence_length:,}")

    N = dataset_coll[0].D

    return loader, sequence_length, N


class EmbeddingProjectionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(EmbeddingProjectionModel, self).__init__()
        self.projection = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.projection(x)


class CausalTransformer(nn.Module):
    def __init__(self, vocab_size, hidden_dim, num_layers, seq_length):
        super().__init__()
        self.embedding = EmbeddingProjectionModel(vocab_size, hidden_dim)
        decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_dim, nhead=4)
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.output_head = nn.Linear(hidden_dim, 1)
        self.sequence_length = seq_length

    def forward(self, x, padding_mask):
        x = self.embedding(x).transpose(0, 1)
        mask = generate_local_mask(self.sequence_length).to(x.device)
        out = self.transformer(
            x,
            memory=torch.zeros(1, x.size(1), x.size(2)).to(x.device),
            tgt_mask=mask,
            tgt_key_padding_mask=padding_mask,
        )
        out = self.output_head(out.transpose(0, 1)).squeeze(-1)
        return out

    def fit(self, num_epochs, dataloader):
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=1e-3)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            count = 0
            for _, batch in enumerate(dataloader):
                x = batch[0][0]
                padding_mask = (~batch[0][1]).float()
                y = batch[1]
                out = self(x, padding_mask)
                optimizer.zero_grad()
                loss = criterion(out, y)
                total_loss += loss
                count += 1
                loss.backward()
                optimizer.step()

            logger.info(
                "Training set | Epoch %s | MSE Loss: %s"
                % (
                    epoch + 1,
                    round((total_loss / count).item(), 4),
                )
            )


def test_model(model, sequence_length, vocab_size):
    model.eval()

    criterion = torch.nn.MSELoss()

    test_result_fn = f"test_results/test_result_transformer_same_node_seql_{datetime.datetime.now().strftime("%Y_%m_%d_%H_%M")}.csv"

    logger.info("Saving test results to %s.", test_result_fn)

    f = open(
        test_result_fn,
        "w",
        newline="",
    )
    csv_writer = csv.writer(f)
    csv_writer.writerow(["Dataset", "Actual", "Predicted", "Correct"])

    dataset_s_n7_test = CVFConfigForTransformerTestDatasetWName(
        device,
        "star_graph_n7",
        "star_graph_n7_config_rank_dataset.csv",
        D=7,
    )

    dataset_rr_n7_test = CVFConfigForTransformerTestDatasetWName(
        device,
        "graph_random_regular_graph_n7_d4",
        "graph_random_regular_graph_n7_d4_config_rank_dataset.csv",
        D=7,
    )

    dataset_plc_n7_test = CVFConfigForTransformerTestDatasetWName(
        device,
        "graph_powerlaw_cluster_graph_n7",
        "graph_powerlaw_cluster_graph_n7_config_rank_dataset.csv",
        D=7,
    )

    test_datasets = ConcatDataset(
        [dataset_s_n7_test, dataset_rr_n7_test, dataset_plc_n7_test]
    )

    with torch.no_grad():
        test_dataloader = DataLoader(test_datasets, batch_size=10240)

        total_loss = 0
        total_matched = 0
        count = 0
        total_seq_count = 0
        for batch in test_dataloader:
            x = batch[0][:, 0, :]
            padd = torch.full((sequence_length - 1, vocab_size), -1).to(device)
            padded_batches = [torch.cat([batch.unsqueeze(0), padd]) for batch in x]
            x = torch.stack(padded_batches)
            padding_mask = torch.full(
                (x.shape[0], sequence_length), 1, dtype=torch.bool
            ).to(device)
            padding_mask[:, 0] = False
            padding_mask = (~padding_mask).float()
            y = batch[1]
            out = model(x, padding_mask)
            out = out[:, 0].unsqueeze(-1)
            matched = torch.round(out) == y
            csv_writer.writerows(
                (n, j.item(), k.item(), z.item())
                for (n, j, k, z) in zip(
                    batch[2],
                    y.detach().cpu().numpy(),
                    out.detach().cpu().numpy(),
                    matched,
                )
            )
            loss = criterion(out, y)
            total_loss += loss
            out = torch.round(out)
            matched = matched.sum().item()
            total_seq_count += out.numel()
            total_matched += matched
            count += 1

        logger.info(
            f"Test set | MSE loss: {round((total_loss / count).item(), 4)} | Total matched: {total_matched:,} out of {total_seq_count:,} (Accuracy: {round(total_matched / total_seq_count * 100, 2):,}%)",
        )

    f.close()


def main(num_epochs, batch_size):
    loader, sequence_length, N = get_dataset_coll(batch_size)
    vocab_size = N
    hidden_dim = 16
    num_layers = 2

    model = CausalTransformer(vocab_size, hidden_dim, num_layers, sequence_length).to(
        device
    )
    logger.info(f"{model}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    start_time = time.time()
    model.fit(num_epochs, loader)
    logger.info("\n")
    logger.info(
        "End Training | Total training time taken %ss",
        round(time.time() - start_time, 4),
    )
    logger.info("\n")
    model_name = f"trained_models/transformer_trained_at_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt"
    logger.info("Saving model %s", model_name)
    torch.save(model, model_name)

    logger.info("Testing model.")
    test_model(model, sequence_length, vocab_size)


if __name__ == "__main__":
    main(num_epochs=50, batch_size=1024)
    logger.info("Done!")
