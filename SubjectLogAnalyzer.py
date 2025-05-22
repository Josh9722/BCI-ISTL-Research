import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List


def extract_log_metrics(log_text: str) -> Dict[str, float]:
    val_accs = []
    train_accs = []

    for line in log_text.splitlines():
        m_acc = re.search(r"Acc:\s+([0-9.]+).*Val Acc:\s+([0-9.]+)", line)
        if m_acc:
            train_accs.append(float(m_acc.group(1)))
            val_accs.append(float(m_acc.group(2)))

    final_val_acc = val_accs[-1] if val_accs else float('nan')
    best_val_acc = max(val_accs) if val_accs else float('nan')
    epoch_to_best = val_accs.index(best_val_acc) + 1 if val_accs else float('nan')
    val_acc_stability = np.std(val_accs) if val_accs else float('nan')

    final_train_val_gap = (
        train_accs[-1] - val_accs[-1] if train_accs and val_accs else float('nan')
    )
    max_train_val_gap = (
        max(np.array(train_accs) - np.array(val_accs)) if train_accs and val_accs else float('nan')
    )
    converged = abs(final_val_acc - best_val_acc) <= 0.02 if not np.isnan(final_val_acc) else False

    recall = {'T0': float('nan'), 'T1': float('nan'), 'T2': float('nan')}
    in_class_report = False
    for line in log_text.splitlines():
        if "Classification Report" in line:
            in_class_report = True
        if in_class_report:
            m = re.match(r"\s*T(\d).*?([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", line)
            if m:
                recall[f"T{m.group(1)}"] = float(m.group(3))

    return {
        "final_val_acc": final_val_acc,
        "best_val_acc": best_val_acc,
        "epoch_to_best_val_acc": epoch_to_best,
        "val_acc_stability": val_acc_stability,
        "final_train_val_gap": final_train_val_gap,
        "max_train_val_gap": max_train_val_gap,
        "converged": converged,
        "recall_T0": recall['T0'],
        "recall_T1": recall['T1'],
        "recall_T2": recall['T2']
    }


def extract_subject_id(filename: str) -> str:
    match = re.search(r"_S(\d+)", filename)
    return f"S{match.group(1)}" if match else None


def compare_all_cluster_models(baseline_dir: str, cluster_root: str) -> pd.DataFrame:
    baseline_files = {
        extract_subject_id(fn): os.path.join(baseline_dir, fn)
        for fn in os.listdir(baseline_dir)
        if fn.endswith(".txt") and extract_subject_id(fn) is not None
    }

    rows = []
    for cluster_model_name in os.listdir(cluster_root):
        cluster_model_path = os.path.join(cluster_root, cluster_model_name)
        if not os.path.isdir(cluster_model_path):
            continue

        for file in os.listdir(cluster_model_path):
            if not file.endswith(".txt"):
                continue
            subject_id = extract_subject_id(file)
            if subject_id is None:
                continue

            baseline_path = baseline_files.get(subject_id)
            cluster_path = os.path.join(cluster_model_path, file)

            if not baseline_path or not os.path.exists(cluster_path):
                print(f"Missing file for subject {subject_id}")
                continue

            with open(baseline_path, 'r', encoding='utf-8') as f:
                baseline_log = f.read()
            with open(cluster_path, 'r', encoding='utf-8') as f:
                cluster_log = f.read()

            base_metrics = extract_log_metrics(baseline_log)
            clus_metrics = extract_log_metrics(cluster_log)

            delta_metrics = {
                f"{k}_delta": clus_metrics.get(k, float('nan')) - base_metrics.get(k, float('nan'))
                for k in base_metrics if isinstance(base_metrics[k], (int, float, np.float64))
            }

            # Extract true group name from filename, e.g., ClusterModel1_Group2_S7...
            match = re.match(r"(ClusterModel\d+_Group\d+)", file)
            group_name = match.group(1) if match else cluster_model_name

            rows.append({
                "cluster_model": group_name,  # ← includes both cluster + group ID
                "subject": subject_id,
                **base_metrics,
                **clus_metrics,
                **delta_metrics
            })

    return pd.DataFrame(rows)

def generate_summary_text(df: pd.DataFrame) -> str:
    lines = []

    if df.empty:
        return "No valid subject comparisons found."

    def mean_delta(group, field):
        return group[f"{field}_delta"].mean() if f"{field}_delta" in group.columns else float('nan')

    for model in df['cluster_model'].unique():
        group = df[df['cluster_model'] == model]
        subject_count = group['subject'].nunique()  # ✅ CORRECT: count unique subject IDs

        improved = (group["final_val_acc_delta"] > 0).sum()
        worsened = (group["final_val_acc_delta"] < 0).sum()
        unchanged = (group["final_val_acc_delta"] == 0).sum()

        lines.append(f"\n======== SUMMARY: {model} ========")
        lines.append(f"Total subjects .......... {subject_count}")
        lines.append(f"↑ Improved .............. {improved}")
        lines.append(f"↓ Worsened .............. {worsened}")
        lines.append(f"= Unchanged ............. {unchanged}")
        lines.append(f"Δ Stability ............. {mean_delta(group, 'val_acc_stability'):+.3f}")
        if "cluster_purity" in group.columns:
            lines.append(f"Avg Cluster Purity ...... {group['cluster_purity'].mean():.3f}")

    return "\n".join(lines)

def generate_purity_stats_summary(df: pd.DataFrame) -> str:
    lines = ["\n======== CLUSTER PURITY vs ACCURACY ========"]

    if 'cluster_purity' not in df.columns:
        return "cluster_purity column not found in DataFrame."

    for model in df['cluster_model'].unique():
        group = df[df['cluster_model'] == model]

        lines.append(f"\n--- {model} ---")
        n = len(group)
        valid = group[['cluster_purity', 'final_val_acc_delta']].dropna()

        if len(valid) < 3:
            lines.append("Insufficient data for correlation.")
            continue

        corr = valid['cluster_purity'].corr(valid['final_val_acc_delta'])
        lines.append(f"Subjects analyzed ............ {len(valid)}/{n}")
        lines.append(f"Pearson r (purity vs Δ acc) .. {corr:.3f}")
        lines.append(f"Mean Purity .................. {valid['cluster_purity'].mean():.3f}")
        lines.append(f"Mean Δ Accuracy .............. {valid['final_val_acc_delta'].mean():+.3f}")
        lines.append(f"Median Purity ................ {valid['cluster_purity'].median():.3f}")
        lines.append(f"Median Δ Accuracy ............ {valid['final_val_acc_delta'].median():+.3f}")

        # Optional: bucket breakdown
        buckets = [0.5, 0.7, 0.8, 0.9, 1.01]
        labels = ["<0.7", "0.7–0.8", "0.8–0.9", "≥0.9"]
        valid['bucket'] = pd.cut(valid['cluster_purity'], bins=buckets, labels=labels, include_lowest=True)
        bucket_means = valid.groupby('bucket')['final_val_acc_delta'].mean()

        lines.append("Δ Accuracy by Purity Range:")
        for label in labels:
            delta = bucket_means.get(label, np.nan)
            if not np.isnan(delta):
                lines.append(f"  {label:<7} .............. {delta:+.3f}")

    return "\n".join(lines)

def add_cluster_purity_to_df(df: pd.DataFrame, cluster_root: str) -> pd.DataFrame:
    purity_scores = []
    print("\n🔍 Starting cluster purity extraction...\n")

    for full_model_name in df['cluster_model'].unique():
        # Extract base cluster model (e.g., ClusterModel1 from ClusterModel1_Group2)
        base_match = re.match(r"(ClusterModel\d+)_Group\d+", full_model_name)
        if not base_match:
            print(f"❌ Could not extract base model name from {full_model_name}")
            continue

        base_model_name = base_match.group(1)
        model_num_match = re.search(r"Model(\d+)", base_model_name)
        model_num = model_num_match.group(1) if model_num_match else base_model_name[-1]

        dist_file = f"Cluster_Distribution_Model_{model_num}.txt"
        dist_path = os.path.join(cluster_root, base_model_name, dist_file)

        if not os.path.exists(dist_path):
            print(f"⚠️ Missing cluster distribution file: {dist_path}")
            continue

        try:
            dist_df = pd.read_csv(dist_path, sep=r'\s+', header=0, index_col=0, engine='python')
            dist_df.index = dist_df.index.map(lambda x: f"S{int(x)}" if str(x).isdigit() else x)
        except Exception as e:
            print(f"❌ Failed to read {dist_path}: {e}")
            continue

        print(f"\n✅ Processing {full_model_name} (→ using {base_model_name}'s distribution)")
        print("Example cluster distribution entries:")
        print(dist_df.head(3))

        model_subjects = df[df['cluster_model'] == full_model_name]['subject'].unique()
        matched = 0

        for subject_id in model_subjects:
            if subject_id in dist_df.index:
                row = dist_df.loc[subject_id]
                total = row.sum()
                purity = row.max() / total if total > 0 else np.nan
                purity_scores.append((full_model_name, subject_id, purity))
                matched += 1
            else:
                print(f"⚠️ Subject {subject_id} not found in distribution for {base_model_name}")

        print(f"✅ Matched {matched}/{len(model_subjects)} subjects in {full_model_name}\n")

    purity_df = pd.DataFrame(purity_scores, columns=['cluster_model', 'subject', 'cluster_purity'])
    print("📊 Sample of purity scores:")
    print(purity_df.head(5))

    df_merged = df.merge(purity_df, on=['cluster_model', 'subject'], how='left')
    print(f"\n🔁 Final merge complete. Subjects with non-null purity: {df_merged['cluster_purity'].notna().sum()} / {len(df_merged)}")

    return df_merged