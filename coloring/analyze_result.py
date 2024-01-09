import pandas as pd
import seaborn as sns

from matplotlib import pyplot as plt


folder = "results/"
graph = "graph_powerlaw_cluster_graph_n9"

pt_df = pd.read_csv(f"{folder}program_transitions_{graph}.csv")
cvf_df = pd.read_csv(f"{folder}cvf_{graph}.csv")

pt_avg_counts = pt_df["Ar"].value_counts()
pt_max_counts = pt_df["M"].value_counts()

cvf_avg_counts = cvf_df["Ar"].value_counts()
cvf_max_counts = cvf_df["M"].value_counts()

print()
print("Average Count:")
print("--------------")
print(
    "{0: >25}{1: >25}{2: >25}".format("Rank Effect", "Program Trans Count", "CVF Count")
)
for i in sorted(set(cvf_avg_counts.index) | set(pt_avg_counts.index)):
    print("{0: >25}{1: >25}{2: >25}".format(i, pt_avg_counts.get(i, '0'), cvf_avg_counts.get(i, '0')))

print()
print()

print("Max Count:")
print("----------")
print(
    "{0: >25}{1: >25}{2: >25}".format("Rank Effect", "Program Trans Count", "CVF Count")
)
for i in sorted(set(cvf_max_counts.index) | set(pt_max_counts.index)):
    print("{0: >25}{1: >25}{2: >25}".format(i, pt_max_counts.get(i, '0'), cvf_max_counts.get(i, '0')))
