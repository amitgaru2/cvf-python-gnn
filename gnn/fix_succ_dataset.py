import os
import csv

program = "coloring"
graph = "graph_powerlaw_cluster_graph_n7"
filename = f"{graph}_config_rank_dataset.csv"
w_filename = f"{graph}_config_rank_dataset_v2.csv"
dataset_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", ""),
    "cvf-analysis",
    "v2",
    "datasets",
    program,
)

file_path = os.path.join(dataset_dir, filename)
write_file_path = os.path.join(dataset_dir, w_filename)


csv_reader = csv.DictReader(open(file_path, "r"), fieldnames={"config", "rank", "succ"})
csv_writer = csv.DictWriter(
    open(write_file_path, "w"), fieldnames={"config", "rank", "succ"}
)

csv_writer.writeheader()

next(csv_reader)

for row in csv_reader:
    print(row)
    break
