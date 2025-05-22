from SubjectLogAnalyzer import (
    compare_all_cluster_models,
    generate_summary_text,
    add_cluster_purity_to_df,
    generate_purity_stats_summary  # ← NEW function
)

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# === Step 1: Run comparison between baseline and cluster models ===
df = compare_all_cluster_models(
    baseline_dir=r"C:\Users\wells\OneDrive\Documents\GitHub\BCI_Log_Analysis\Data\BaselineLog",
    cluster_root=r"C:\Users\wells\OneDrive\Documents\GitHub\BCI_Log_Analysis\Data\ClusterModels"
)

# === Step 2: Add cluster purity per subject ===
df = add_cluster_purity_to_df(df, r"C:\Users\wells\OneDrive\Documents\GitHub\BCI_Log_Analysis\Data\ClusterModels")

# === Step 3: Save full comparison data ===
df.to_csv("subject_level_comparison.csv", index=False)
print("✅ Saved subject-level comparison → subject_level_comparison.csv")

# === Step 4: Generate summary report per cluster model ===
summary = generate_summary_text(df)
print(summary)
with open("subject_summary_scorecard.txt", "w", encoding="utf-8") as f:
    f.write(summary)
print("✅ Saved textual scorecard → subject_summary_scorecard.txt")

# === Step 5: Generate purity correlation summary ===
purity_summary = generate_purity_stats_summary(df)
print(purity_summary)
with open("subject_purity_summary.txt", "w", encoding="utf-8") as f:
    f.write(purity_summary)
print("✅ Saved purity vs accuracy stats → subject_purity_summary.txt")