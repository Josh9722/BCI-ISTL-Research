# AnalyseModels.py – condensed 7-feature evaluation
import os, re, numpy as np, pandas as pd
from io import StringIO
from scipy.stats import wilcoxon

# ─── helpers ────────────────────────────────────────────────────────────
def _per_sub_df(block: str) -> pd.DataFrame:
    rows = [ln for ln in block.splitlines()
            if ln.strip() and ln.lstrip()[0].isdigit()]
    return pd.read_csv(StringIO("\n".join(rows)),
                       sep=r"\s+",
                       names=["subject", "n_epochs", "accuracy", "f1"])

def _class_recall(txt: str) -> dict:
    """
    Extract recall for T0, T1, T2 anywhere in the log.
    Works even if the Classification Report header format varies.
    """
    rec = {}
    for ln in txt.splitlines():
        ln = ln.lstrip()
        # match 'T0 (Rest) ... precision recall f1 support'
        m = re.match(r"^(T\d).*?\s([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)", ln)
        if m:
            rec[m.group(1)] = float(m.group(3))   # recall column
    return rec

def _grab_test_table(txt: str) -> str:
    """
    Return text under 'Per-subject metrics (test set):' heading.
    Stops at the first blank line after the table.
    """
    head_pat = r"Per-subject metrics \(test set\):"
    m = re.search(head_pat, txt)
    if not m:
        return ""
    start = m.end()
    rest  = txt[start:].lstrip("\n")
    # table ends at first blank line
    end   = rest.find("\n\n")
    return rest[:end if end != -1 else None]


# ─── single comparison card ─────────────────────────────────────────────
class _ScoreCard:
    """Compare one cluster log to baseline; output 7 key metrics."""

    def __init__(self, base_log: str, clus_log: str):
        self.base_log, self.clus_log = base_log, clus_log
        self.df_b, self.rec_b = self._load(base_log)
        self.df_c, self.rec_c = self._load(clus_log)

    def _load(self, path):
        txt = open(path, encoding="utf-8").read()
        per_sub = _per_sub_df(_grab_test_table(txt))
        return per_sub, _class_recall(txt)

    def headline(self) -> str:
        # 1-5 global stats ------------------------------------------------
        m_b, s_b = self.df_b.accuracy.mean(), self.df_b.accuracy.std(ddof=1)
        m_c, s_c = self.df_c.accuracy.mean(), self.df_c.accuracy.std(ddof=1)
        floor_b, top_b = self.df_b.accuracy.min(), self.df_b.accuracy.max()
        floor_c, top_c = self.df_c.accuracy.min(), self.df_c.accuracy.max()
        iqr_b = self.df_b.accuracy.quantile(.75) - self.df_b.accuracy.quantile(.25)
        iqr_c = self.df_c.accuracy.quantile(.75) - self.df_c.accuracy.quantile(.25)
        pct50_b = (self.df_b.accuracy >= .5).mean()
        pct50_c = (self.df_c.accuracy >= .5).mean()

        # 6 Wilcoxon on shared subjects ----------------------------------
        shared = self.df_b.merge(self.df_c, on="subject", suffixes=("_b", "_c"))
        pval = wilcoxon(shared.accuracy_c, shared.accuracy_b).pvalue \
               if len(shared) >= 3 else np.nan

        # 7 per-class recall Δ -------------------------------------------
        classes = ["T0", "T1", "T2"]
        d_recall = ", ".join(
            f"{cl}:{self.rec_c.get(cl, np.nan)-self.rec_b.get(cl, np.nan):+.2f}"
            if cl in self.rec_b and cl in self.rec_c else f"{cl}:n/a"
            for cl in classes
        )

        return (
            f"[{os.path.basename(self.clus_log)}]\n"
            f"  1. mean acc ........ {m_b:.3f} → {m_c:.3f} ({m_c-m_b:+.1%})\n"
            f"  2. stdev ........... {s_b:.3f} → {s_c:.3f} ({s_c-s_b:+.0%})\n"
            f"  3. floor/ceil ...... {floor_b:.2f}–{top_b:.2f} vs {floor_c:.2f}–{top_c:.2f}\n"
            f"  4. ≥0.50 ............ {pct50_b:.0%} → {pct50_c:.0%}\n"
            f"  5. IQR .............. {iqr_b:.2f} → {iqr_c:.2f}\n"
            f"  6. Wilcoxon Δacc .... "
            f"{'p='+format(pval,'.4f') if not np.isnan(pval) else 'n<3 shared'}\n"
            f"  7. Δ recall per cls . {d_recall}\n"
        )


# ─── public API (call-pattern unchanged) ────────────────────────────────
class AnalyseModels:
    def __init__(self, baseline_log: str, clustered_logs: list, out_dir="./logs"):
        self.base = baseline_log
        self.clustered = clustered_logs
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    def produceReport(self):
        lines = ["======== CLUSTER SCORECARDS vs BASELINE ========",
                 f"Baseline file: {self.base}\n"]

        for log in self.clustered:
            try:
                lines.append(_ScoreCard(self.base, log).headline())
            except Exception as e:
                lines.append(f"[{os.path.basename(log)}]  — could not parse ({e})\n")

        # ---- global average across clusters -------------------------
        all_means, all_stds = [], []
        for log in self.clustered:
            try:
                df, _ = _ScoreCard(self.base, log)._load(log)
                all_means.append(df.accuracy.mean())
                all_stds.append(df.accuracy.std(ddof=1))
            except Exception:
                pass

        if all_means and len(lines) >= 2:
            g_mean = np.mean(all_means)
            g_std  = np.mean(all_stds)
            lines.insert(2,
                f"GLOBAL (avg over {len(all_means)} clusters)"
                f"  • mean acc {g_mean:.3f}  ·  stdev {g_std:.3f}\n")

        # ---- print & save ------------------------------------------
        report = "\n".join(lines)
        print(report)

        fname = os.path.join(self.out_dir,
                             f"scorecards_{len(self.clustered)}clusters.txt")
        with open(fname, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved condensed report → {fname}\n")
