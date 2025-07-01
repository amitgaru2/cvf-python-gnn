import torch
import datetime
import torch.nn as nn

from torch.utils.data import DataLoader, random_split
from transformers import BertConfig, BertModel

from dataset import CVFConfigForBertDataset, logger


device = "cuda"

batch_size = 64

epochs = 2000


# dataset = CVFConfigForBertDataset(
#     device,
#     "graph_random_regular_graph_n6_d3",
#     "graph_random_regular_graph_n6_d3_pt_adj_list.txt",
#     D=6,
# )

dataset = CVFConfigForBertDataset(
    device,
    "implicit_graph_n5",
    "implicit_graph_n5_pt_adj_list.txt",
    D=5,
    program="dijkstra",
)

logger.info(f"Dataset: {dataset.dataset_name} | Size: {len(dataset):,}")

train_size = int(0.95 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


class TokenVectorBERT(nn.Module):

    def __init__(self, input_dim, vocab_dim=64, bert_hidden=64, max_seq_len=128):
        super().__init__()
        # Learnable MASK token (for masking positions)
        self.mask_vector = nn.Parameter(torch.zeros(input_dim), requires_grad=True)
        self.token_proj = nn.Linear(
            input_dim, vocab_dim
        )  # turn [0, 0, 2] into an embedding
        self.config = BertConfig(
            vocab_size=1,  # dummy, unused
            hidden_size=bert_hidden,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=bert_hidden * 2,
            max_position_embeddings=max_seq_len,
            pad_token_id=0,
        )
        self.bert = BertModel(self.config)
        self.mlm_head = nn.Linear(bert_hidden, vocab_dim)
        self.decoder_proj = nn.Linear(vocab_dim, input_dim)

    def forward(self, input_vecs, attention_mask=None):
        # input_vecs: (batch_size, seq_len, input_dim) like (2, 4, 3)
        x = self.token_proj(input_vecs)  # (batch_size, seq_len, vocab_dim)
        outputs = self.bert(inputs_embeds=x, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        logits = self.mlm_head(sequence_output)
        pred_token = self.decoder_proj(logits)
        return pred_token


# ----- Masking Function -----
def mask_input_tokens(inputs, mask_before, mask_vector, mask_prob):
    labels = inputs.clone()
    masked_inputs = inputs.clone()
    b_mask = torch.rand(masked_inputs[:, :, 0].shape) < mask_prob  # shape: (B, T)

    for i, mb in enumerate(mask_before):
        b_mask[i, mb:] = False  # do not mask the padded vectors

    for i in range(masked_inputs.size(0)):
        for j in range(masked_inputs.size(1)):
            if b_mask[i, j]:
                masked_inputs[i, j] = mask_vector

    return masked_inputs, labels, b_mask


def masked_mse_loss(pred, target, mask):
    valid_tokens = mask.sum()
    if valid_tokens == 0:
        valid_tokens = 1e-8
    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # (B, T)
    loss = loss * mask.float()

    return loss.sum() / valid_tokens


def main():
    model = TokenVectorBERT(input_dim=dataset.D, vocab_dim=64, bert_hidden=64)

    logger.info(
        "Total parameters: {:,}".format(sum(p.numel() for p in model.parameters()))
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for batch in loader:
            x = batch[0]
            attention_mask = batch[1]
            mask_before = batch[2]
            masked_inputs, target_labels, loss_mask = mask_input_tokens(
                x, mask_before, model.mask_vector, mask_prob=0.15
            )

            logits = model(masked_inputs, attention_mask)

            # Compute loss only on masked positions
            optimizer.zero_grad()
            loss = masked_mse_loss(logits, target_labels, loss_mask)
            loss.backward()
            optimizer.step()
            total_loss += loss

        logger.info(
            f"Epoch {epoch+1}/{epochs} | Loss: {total_loss.item()/ len(loader):.4f}"
        )

    model_name = f"trained_models/bert_trained_at_{datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')}.pt"
    logger.info("Saving model %s", model_name)
    torch.save(model, model_name)

    model.eval()

    test_loader = DataLoader(test_dataset, batch_size=1024, shuffle=False)

    total_loss = 0.0
    for batch in test_loader:
        x = batch[0]
        attention_mask = batch[1]
        mask_before = batch[2]
        masked_inputs, target_labels, loss_mask = mask_input_tokens(
            x, mask_before, model.mask_vector, mask_prob=0.15
        )
        logits = model(masked_inputs, attention_mask)
        loss = masked_mse_loss(logits, target_labels, loss_mask)
        total_loss += loss

    logger.info(f"Test dataset | Loss: {total_loss.item()/ len(test_loader):.4f}")


if __name__ == "__main__":
    main()
