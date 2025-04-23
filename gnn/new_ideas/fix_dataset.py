import os
import dataset

program = "dijkstra"
filename = "implicit_graph_n7_pt_adj_list.txt"
dataset_dir = os.path.join(
    os.getenv("CVF_PROJECT_DIR", ""),
    "cvf-analysis",
    "v2",
    "datasets",
    program,
)

file_path = os.path.join(dataset_dir, filename)

max_cols = 0
with open(file_path, "r") as f:
    row = f.readline()
    while row:
        row_len = len(row.split(","))
        if row_len > max_cols:
            max_cols = row_len
        row = f.readline()


print("max_cols", max_cols)

with open(file_path, "r+") as f:
    content = f.read()
    f.seek(0)
    f.write("".join(["," for _ in range(max_cols)]) + "\n")
