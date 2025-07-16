import os
import ast
import sys
import csv

import torch

program = "coloring"
graph = sys.argv[1]
filename = f"{graph}_config_rank_dataset.csv"
w_filename = f"{graph}_config_rank_dataset_v2.csv"
dataset_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", ""),
    "cvf-analysis",
    "datasets",
    program,
)

file_path = os.path.join(dataset_dir, filename)
write_file_path = os.path.join(dataset_dir, w_filename)


csv_reader = csv.DictReader(open(file_path, "r"), fieldnames=["config", "rank", "succ"])
csv_writer = csv.DictWriter(
    open(write_file_path, "w"), fieldnames=["config", "rank", "succ1", "succ2"]
)

csv_writer.writeheader()

next(csv_reader)

data = []
for row in csv_reader:
    config = torch.FloatTensor([ast.literal_eval(row["config"])])
    succ = ast.literal_eval(row["succ"])
    if succ:
        succ = torch.FloatTensor(succ)
        succ1 = torch.mean(succ, dim=0).unsqueeze(0)  # column wise
        succ2 = torch.mean(succ, dim=1)  # row wise
        succ2 = torch.sum(succ2).repeat(succ1.shape)
    else:
        succ1 = torch.zeros(1, config.shape[1])
        succ2 = succ1.clone()

    data.append(
        {
            "config": config.tolist(),
            "rank": row["rank"],
            "succ1": succ1.tolist(),
            "succ2": succ2.tolist(),
        }
    )
    if len(data) >= 1000:
        csv_writer.writerows(data)
        data = []

if data:
    csv_writer.writerows(data)
    data = []
