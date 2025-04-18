import sys

import pandas as pd
import numpy as np
import torch

from torch.utils.data import DataLoader


from lstm_scratch import SimpleLSTM
from helpers import CVFConfigForAnalysisDataset


model_name = "lstm_trained_at_2025_04_10_00_11"

# graph_name = "star_graph_n15"
# graph_name = "star_graph_n7"
# graph_name = "graph_powerlaw_cluster_graph_n7"
# graph_name = "graph_random_regular_graph_n8_d4"
graph_name = sys.argv[1]


device = "cuda"

# Model class must be defined somewhere
model = torch.load(f"trained_models/{model_name}.pt", weights_only=False)
model.eval()


dataset = CVFConfigForAnalysisDataset(device, graph_name)

data = []
# result_df = pd.DataFrame([], columns=['node', 'rank_effect'])

result_df = pd.DataFrame(
    {"node": pd.Series(dtype="int"), "rank effect": pd.Series(dtype="float")}
)

with torch.no_grad():
    test_dataloader = DataLoader(dataset, batch_size=1)

    count = 0
    for batch in test_dataloader:
        for i in range(len(batch[0])):
            frm_idx = batch[1][i].item()
            frm_rank = model(batch[0][i].unsqueeze(0))
            for (
                position,
                to_indx,
            ) in dataset.cvf_analysis.possible_perturbed_state_frm(frm_idx):
                to = dataset[to_indx]
                to_rank = model(to[0].unsqueeze(0))
                rank_effect = (frm_rank - to_rank).item()  # to round off at 0.5
                data.append({"node": position, "rank effect": rank_effect})

        temp_df = pd.DataFrame(data, columns=["node", "rank effect"])
        data = []
        result_df = pd.concat([result_df, temp_df], ignore_index=True)


result_df["rank effect"] = np.floor(result_df["rank effect"] + 0.5)
# result_df.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf.csv")

print("Done!")


ml_grp_by_node_re = (
    result_df.groupby(["node", "rank effect"]).size().reset_index(name="ml_count")
)
ml_grp_by_node_re.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf_by_node.csv")

ml_grp_by_re = result_df.groupby(["rank effect"]).size().reset_index(name="ml_count")
ml_grp_by_re.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf.csv")

import os

results_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", ""), "cvf-analysis", "v2", "results", "coloring"
)

results_file = f"rank_effects_avg__{graph_name}.csv"
file_path = os.path.join(results_dir, results_file)
f_grp_by_re = pd.read_csv(file_path)
f_grp_by_re = f_grp_by_re.drop(f_grp_by_re.columns[0], axis=1)
f_grp_by_re.rename(columns={"count": "fa_count"}, inplace=True)

f_grp_by_re

df_grp_by_re = pd.merge(
    f_grp_by_re, ml_grp_by_re, on="rank effect", how="outer"
).fillna(0)
df_grp_by_re

df_grp_by_re.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf.csv")


results_file = f"rank_effects_by_node_avg__{graph_name}.csv"

file_path = os.path.join(results_dir, results_file)
f_grp_by_node_re = pd.read_csv(file_path)

f_grp_by_node_re = f_grp_by_node_re.melt(
    id_vars="node",
    value_vars=set(f_grp_by_node_re.columns) - {"node"},
    var_name="rank effect",
    value_name="fa_count",
)
f_grp_by_node_re["rank effect"] = f_grp_by_node_re["rank effect"].astype(float)
f_grp_by_node_re

df_grp_by_node_re = pd.merge(
    f_grp_by_node_re, ml_grp_by_node_re, on=["node", "rank effect"], how="outer"
).fillna(0)
df_grp_by_node_re

df_grp_by_node_re.to_csv(f"ml_predictions/{model_name}__{graph_name}__cvf_by_node.csv")

print(f"Complete for {graph_name}")
