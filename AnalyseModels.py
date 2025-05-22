import os
import re
import numpy as np
import pandas as pd
from io import StringIO
from scipy.stats import wilcoxon
from typing import List


def _load_table_block(txt: str) -> str:
    """
    Extracts the last TSV-style accuracy table block from mixed logs or returns the whole text if it's just a TSV.
    """
    if txt.lstrip().startswith("Subject"):
        return txt

    blocks = re.findall(r"Subject\t.*?(?:\n\d+\t.*)+", txt, re.DOTALL)
    if not blocks:
        raise ValueError("No subject accuracy table found.")
    return blocks[-1]

def _extract_recall(txt: str) -> dict:
    """
    Extracts recall values for classes T0, T1, T2 from a Classification Report in the log.
    """
    recall = {}
    for ln in txt.splitlines():
        m = re.match(r"^(T\d).*?\s([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", ln.strip())
        if m:
            recall[m.group(1)] = float(m.group(2))
    return recall


class ModelLogAnalyzer:
    def __init__(self, baseline_log: str, cluster_logs: List[str], default_out_dir: str = "./logs"):
        self.baseline_log = baseline_log
        self.cluster_logs = cluster_logs
        self.default_out_dir = default_out_dir
        os.makedirs(self.default_out_dir, exist_ok=True)

    def _load_log(self, path: str):
        txt = open(path, encoding="utf-8").read()
        block = _load_table_block(txt)
        df = pd.read_csv(StringIO(block), sep="\t")
        df.columns = df.columns.str.strip().str.lower()
        df = df.rename(columns={"overallacc": "accuracy"})
        recall = _extract_recall(txt)
        return df, recall

    def _baseline_summary(self) -> str:
        df_b, rec_b = self._load_log(self.baseline_log)
        m_b = df_b["accuracy"].mean()
        s_b = df_b["accuracy"].std(ddof=1)
        floor_b, top_b = df_b["accuracy"].min(), df_b["accuracy"].max()
        iqr_b = df_b["accuracy"].quantile(.75) - df_b["accuracy"].quantile(.25)
        pct50_b = (df_b["accuracy"] >= .5).mean()
        classes = ["T0", "T1", "T2"]
        recall_str = ", ".join(
            f"{cl}:{rec_b.get(cl, 0.0)-rec_b.get(cl, 0.0):+.2f}" for cl in classes
        )
        return (
            "======== BASELINE SCORECARD ========\n"
            f"  1. mean acc ........ {m_b:.3f}\n"
            f"  2. stdev ........... {s_b:.3f}\n"
            f"  3. floor/ceil ...... {floor_b:.2f}\u2013{top_b:.2f}\n"
            f"  4. ≥0.50 ............ {pct50_b:.0%}\n"
            f"  5. IQR .............. {iqr_b:.2f}\n"
            f"  6. Wilcoxon Δacc .... n/a\n"
            f"  7. Δ recall per cls . {recall_str}\n"
        )

    def analyze_cluster(self, cluster_log: str) -> dict:
        df_b, rec_b = self._load_log(self.baseline_log)
        df_c, rec_c = self._load_log(cluster_log)
        m_b, s_b = df_b["accuracy"].mean(), df_b["accuracy"].std(ddof=1)
        m_c, s_c = df_c["accuracy"].mean(), df_c["accuracy"].std(ddof=1)
        floor_b, top_b = df_b["accuracy"].min(), df_b["accuracy"].max()
        floor_c, top_c = df_c["accuracy"].min(), df_c["accuracy"].max()
        iqr_b = df_b["accuracy"].quantile(.75) - df_b["accuracy"].quantile(.25)
        iqr_c = df_c["accuracy"].quantile(.75) - df_c["accuracy"].quantile(.25)
        pct50_b = (df_b["accuracy"] >= .5).mean()
        pct50_c = (df_c["accuracy"] >= .5).mean()
        shared = df_b.merge(df_c, on="subject", suffixes=("_b", "_c"))
        pval = wilcoxon(shared["accuracy_c"], shared["accuracy_b"]).pvalue if len(shared) >= 3 else np.nan
        classes = ["T0", "T1", "T2"]
        d_recall = {cl: rec_c.get(cl, np.nan) - rec_b.get(cl, np.nan)
                    for cl in classes if cl in rec_b and cl in rec_c}
        return {
            "cluster_log": os.path.basename(cluster_log),
            "mean_accuracy_baseline": m_b,
            "mean_accuracy_cluster": m_c,
            "mean_accuracy_delta": m_c - m_b,
            "std_accuracy_baseline": s_b,
            "std_accuracy_cluster": s_c,
            "floor_ceiling_baseline": (floor_b, top_b),
            "floor_ceiling_cluster": (floor_c, top_c),
            "pct_above_50_baseline": pct50_b,
            "pct_above_50_cluster": pct50_c,
            "iqr_baseline": iqr_b,
            "iqr_cluster": iqr_c,
            "wilcoxon_p": pval,
            "recall_deltas": d_recall
        }

    def produce_report(self, out_dir: str = None, out_name: str = None):
        lines = [
            self._baseline_summary(),
            "\n======== CLUSTER SCORECARDS vs BASELINE ========",
            f"Baseline file: {self.baseline_log}\n"
        ]
        all_means, all_stds = [], []
        for log in self.cluster_logs:
            try:
                R = self.analyze_cluster(log)
                recall_part = ", ".join(f"{k}:{v:+.2f}" for k, v in R['recall_deltas'].items()) or 'n/a'
                lines.append(
                    f"[{R['cluster_log']}]\n"
                    f"  1. mean acc ........ {R['mean_accuracy_baseline']:.3f} → {R['mean_accuracy_cluster']:.3f} ({R['mean_accuracy_delta']:+.1%})\n"
                    f"  2. stdev ........... {R['std_accuracy_baseline']:.3f} → {R['std_accuracy_cluster']:.3f} ({R['std_accuracy_cluster']-R['std_accuracy_baseline']:+.0%})\n"
                    f"  3. floor/ceil ...... {R['floor_ceiling_baseline'][0]:.2f}\u2013{R['floor_ceiling_baseline'][1]:.2f} vs {R['floor_ceiling_cluster'][0]:.2f}\u2013{R['floor_ceiling_cluster'][1]:.2f}\n"
                    f"  4. ≥0.50 ............ {R['pct_above_50_baseline']:.0%} → {R['pct_above_50_cluster']:.0%}\n"
                    f"  5. IQR .............. {R['iqr_baseline']:.2f} → {R['iqr_cluster']:.2f}\n"
                    f"  6. Wilcoxon Δacc .... {('p='+format(R['wilcoxon_p'],'.4f')) if not np.isnan(R['wilcoxon_p']) else 'n<3 shared'}\n"
                    f"  7. Δ recall per cls . {recall_part}\n"
                )
                all_means.append(R['mean_accuracy_cluster'])
                all_stds.append(R['std_accuracy_cluster'])
            except Exception as e:
                lines.append(f"[{os.path.basename(log)}] — could not parse ({e})\n")
        if all_means:
            g_mean = np.mean(all_means)
            g_std  = np.mean(all_stds)
            lines.insert(2, f"GLOBAL (avg over {len(all_means)} clusters)  • mean acc {g_mean:.3f}  ·  stdev {g_std:.3f}\n")
        report = "\n".join(lines)
        print(report)
        # Ensure output path is relative to current working directory
        base_dir = os.getcwd()
        target_dir = os.path.join(base_dir, out_dir or self.default_out_dir)
        target_name = out_name or f"scorecards_{len(self.cluster_logs)}clusters.txt"
        os.makedirs(target_dir, exist_ok=True)
        out_path = os.path.join(target_dir, target_name)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved condensed report → {out_path}\n")

def produceEEGNetReport():
    # Cluster Model 1
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel1\ClusterModel1_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel1\ClusterModel1_Group2_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir = "logs/Unique_EEGNet", out_name="Cluster Model 1 Scorecard.txt")

    # Cluster Model 2
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel2\ClusterModel2_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel2\ClusterModel2_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel2\ClusterModel2_Group3_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir = "logs/Unique_EEGNet", out_name="Cluster Model 2 Scorecard.txt")

    # Cluster Model 3
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel3\ClusterModel3_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel3\ClusterModel3_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel3\ClusterModel3_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel3\ClusterModel3_Group4_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir = "logs/Unique_EEGNet", out_name="Cluster Model 3 Scorecard.txt")

    # Cluster Model 4
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel4\ClusterModel4_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel4\ClusterModel4_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel4\ClusterModel4_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel4\ClusterModel4_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel4\ClusterModel4_Group5_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir = "logs/Unique_EEGNet", out_name="Cluster Model 4 Scorecard.txt")

    # Cluster Model 5
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel5\ClusterModel5_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel5\ClusterModel5_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel5\ClusterModel5_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel5\ClusterModel5_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel5\ClusterModel5_Group5_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering model only on T0\clustering_models\ClusterModel5\ClusterModel5_Group6_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir = "logs/Unique_EEGNet", out_name="Cluster Model 5 Scorecard.txt")

def produceAllClassEEGNetReport():
    # UNIQUE LSO EEGNET REPORTS
    # Cluster Model 1
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel1\ClusterModel1_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel1\ClusterModel1_Group2_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_AllClass_EEGNet", out_name="Cluster Model 1 Scorecard.txt")
    # Cluster Model 2
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel2\ClusterModel2_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel2\ClusterModel2_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel2\ClusterModel2_Group3_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_AllClass_EEGNet", out_name="Cluster Model 2 Scorecard.txt")
    # Cluster Model 3
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel3\ClusterModel3_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel3\ClusterModel3_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel3\ClusterModel3_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel3\ClusterModel3_Group4_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_AllClass_EEGNet", out_name="Cluster Model 3 Scorecard.txt")
    # Cluster Model 4
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel4\ClusterModel4_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel4\ClusterModel4_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel4\ClusterModel4_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel4\ClusterModel4_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel4\ClusterModel4_Group5_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_AllClass_EEGNet", out_name="Cluster Model 4 Scorecard.txt")
    # Cluster Model 5
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel5\ClusterModel5_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel5\ClusterModel5_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel5\ClusterModel5_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel5\ClusterModel5_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel5\ClusterModel5_Group5_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\EEGNET\Unique - LOSO - Clustering Models Include all classes\clustering_models\ClusterModel5\ClusterModel5_Group6_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_AllClass_EEGNet", out_name="Cluster Model 5 Scorecard.txt")


def produceShallowNetReport():
    
    # UNIQUE LOSO SHALLOWNET REPORTS
    # Cluster Model 1
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel1\ClusterModel1_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel1\ClusterModel1_Group2_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_ShallowNet", out_name="Cluster Model 1 Scorecard.txt")

    # Cluster Model 2
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel2\ClusterModel2_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel2\ClusterModel2_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel2\ClusterModel2_Group3_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_ShallowNet", out_name="Cluster Model 2 Scorecard.txt")

    # Cluster Model 3
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel3\ClusterModel3_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel3\ClusterModel3_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel3\ClusterModel3_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel3\ClusterModel3_Group4_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_ShallowNet", out_name="Cluster Model 3 Scorecard.txt")
    
    # Cluster Model 4
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel4\ClusterModel4_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel4\ClusterModel4_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel4\ClusterModel4_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel4\ClusterModel4_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel4\ClusterModel4_Group5_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_ShallowNet", out_name="Cluster Model 4 Scorecard.txt")
    
    # Cluster Model 5
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel5\ClusterModel5_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel5\ClusterModel5_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel5\ClusterModel5_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel5\ClusterModel5_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel5\ClusterModel5_Group5_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Unique\clustering_models\ClusterModel5\ClusterModel5_Group6_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Unique_ShallowNet", out_name="Cluster Model 5 Scorecard.txt")

    # MAJORITY LOSO SHALLOWNET REPORTS
    # Cluster Model 1
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel1\ClusterModel1_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel1\ClusterModel1_Group2_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Majority_ShallowNet", out_name="Cluster Model 1 Scorecard.txt")
    
    # Cluster Model 2
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel2\ClusterModel2_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel2\ClusterModel2_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel2\ClusterModel2_Group3_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Majority_ShallowNet", out_name="Cluster Model 2 Scorecard.txt")
    
    # Cluster Model 3
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel3\ClusterModel3_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel3\ClusterModel3_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel3\ClusterModel3_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel3\ClusterModel3_Group4_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Majority_ShallowNet", out_name="Cluster Model 3 Scorecard.txt")

    # Cluster Model 4
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel4\ClusterModel4_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel4\ClusterModel4_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel4\ClusterModel4_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel4\ClusterModel4_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel4\ClusterModel4_Group5_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Majority_ShallowNet", out_name="Cluster Model 4 Scorecard.txt")

    # Cluster Model 5
    baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
    cluster_logs = [
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel5\ClusterModel5_Group1_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel5\ClusterModel5_Group2_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel5\ClusterModel5_Group3_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel5\ClusterModel5_Group4_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel5\ClusterModel5_Group5_50epochs_5chans.keras.txt",
        r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\ShallowNet\LOSO - Majority\clustering_models\ClusterModel5\ClusterModel5_Group6_50epochs_5chans.keras.txt"
    ]
    analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
    analyzer.produce_report(out_dir="logs/Majority_ShallowNet", out_name="Cluster Model 5 Scorecard.txt")

def produceDeepConvReport():
        # Cluster Model 1
        baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
        cluster_logs = [
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\\clustering_models\ClusterModel1\ClusterModel1_Group1_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\\clustering_models\ClusterModel1\ClusterModel1_Group2_50epochs_5chans.keras.txt"
        ]
        analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
        analyzer.produce_report(out_dir="logs/Unique_DeepConvNet", out_name="Cluster Model 1 Scorecard.txt")

        # Cluster Model 2
        baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
        cluster_logs = [
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel2\ClusterModel2_Group1_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel2\ClusterModel2_Group2_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel2\ClusterModel2_Group3_50epochs_5chans.keras.txt"
        ]
        analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
        analyzer.produce_report(out_dir="logs/Unique_DeepConvNet", out_name="Cluster Model 2 Scorecard.txt")
        
        # Cluster Model 3
        baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
        cluster_logs = [
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel3\ClusterModel3_Group1_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel3\ClusterModel3_Group2_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel3\ClusterModel3_Group3_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel3\ClusterModel3_Group4_50epochs_5chans.keras.txt"
        ]
        analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
        analyzer.produce_report(out_dir="logs/Unique_DeepConvNet", out_name="Cluster Model 3 Scorecard.txt")

        # Cluster Model 4
        baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
        cluster_logs = [
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel4\ClusterModel4_Group1_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel4\ClusterModel4_Group2_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel4\ClusterModel4_Group3_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel4\ClusterModel4_Group4_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel4\ClusterModel4_Group5_50epochs_5chans.keras.txt"
        ]
        analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
        analyzer.produce_report(out_dir="logs/Unique_DeepConvNet", out_name="Cluster Model 4 Scorecard.txt")
        # Cluster Model 5
        baseline_log = r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\baseline_model\EEGNet_Baseline_50epochs_5chans.keras.txt"
        cluster_logs = [
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel5\ClusterModel5_Group1_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel5\ClusterModel5_Group2_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel5\ClusterModel5_Group3_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel5\ClusterModel5_Group4_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\clustering_models\ClusterModel5\ClusterModel5_Group5_50epochs_5chans.keras.txt",
            r"C:\Users\wells\OneDrive\Documents\BCI-ISTL-Research\Saves\Deep Conv\Unique - T0\\clustering_models\\ClusterModel5\\ClusterModel5_Group6_50epochs_5chans.keras.txt"
        ]
        analyzer = ModelLogAnalyzer(baseline_log, cluster_logs)
        analyzer.produce_report(out_dir="logs/Unique_DeepConvNet", out_name="Cluster Model 5 Scorecard.txt")



if __name__ == "__main__":
    # ------------------------------------------------
    # EEGNET UNIQUE LOSO 
    # produceEEGNetReport() 

    # EEGNET UNIQUE LOSO on all classes 
    # produceAllClassEEGNetReport()
    # ------------------------------------------------

    # ------------------------------------------------
    # SHALLOWNET UNIQUE & MAJORITY LOSO
    # produceShallowNetReport()
    # ------------------------------------------------
    
    # ------------------------------------------------
    # SHALLOWNET UNIQUE LOSO
    # produceDeepConvReport()
    # ------------------------------------------------

