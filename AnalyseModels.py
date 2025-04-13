import os
import re
import re
import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd

class AnalyseModels:
    def __init__(self, baseline_log, clustered_logs):
        """
        :param baseline_log: String path to the baseline log file.
        :param clustered_logs: List of string paths for each clustered model log file.
        """
        self.baseline_log = baseline_log
        self.clustered_logs = clustered_logs
        self.baseline_data = self.parse_log_file(baseline_log)
        self.clustered_data = [self.parse_log_file(log) for log in clustered_logs]

    def parse_log_file(self, file_path):
        """
        Parses a training log file and returns a dictionary with two keys:
          - 'epochs': a list of dictionaries for each epoch with keys:
              'epoch', 'total_epochs', 'loss', 'acc', 'val_loss', 'val_acc'
          - 'report': a dictionary with classification report details parsed into two subkeys:
              'classes': a dict mapping each class ("T0 (Rest)", "T1 (Left-Hand)", "T2 (Right-Hand)")
                         to its precision, recall, f1-score, support.
              'overall': other overall metrics like 'accuracy', 'macro avg', 'weighted avg'.
        """
        data = {
            'epochs': [],
            'report': {}
        }
        with open(file_path, 'r', encoding='utf-8') as f:
            contents = f.read()
        
        # Extract epoch information using regex.
        epoch_pattern = r"Epoch\s+(\d+)/(\d+)\s*\|\s*Loss:\s*([0-9.]+)\s*\|\s*Acc:\s*([0-9.]+)\s*\|\s*Val Loss:\s*([0-9.]+)\s*\|\s*Val Acc:\s*([0-9.]+)"
        epochs = re.findall(epoch_pattern, contents)
        for ep in epochs:
            epoch_dict = {
                'epoch': int(ep[0]),
                'total_epochs': int(ep[1]),
                'loss': float(ep[2]),
                'acc': float(ep[3]),
                'val_loss': float(ep[4]),
                'val_acc': float(ep[5])
            }
            data['epochs'].append(epoch_dict)
        
        # Extract classification report.
        parts = contents.split("Classification Report:")
        if len(parts) > 1:
            report_str = parts[1].strip()
            lines = report_str.splitlines()
            # Remove empty lines
            lines = [line.strip() for line in lines if line.strip()]
            class_report = {}
            overall_report = {}
            # Parse the lines for the three classes of interest and overall metrics.
            for line in lines:
                m = re.match(
                    r"^(T0 \(Rest\)|T1 \(Left-Hand\)|T2 \(Right-Hand\))\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", 
                    line)
                if m:
                    cls_name = m.group(1)
                    class_report[cls_name] = {
                        'precision': float(m.group(2)),
                        'recall': float(m.group(3)),
                        'f1-score': float(m.group(4)),
                        'support': int(m.group(5))
                    }
                else:
                    m_acc = re.match(r"^accuracy\s+([0-9.]+)\s+(\d+)$", line)
                    if m_acc:
                        overall_report['accuracy'] = float(m_acc.group(1))
                        overall_report['support'] = int(m_acc.group(2))
                    m_macro = re.match(r"^macro avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", line)
                    if m_macro:
                        overall_report['macro avg'] = {
                            'precision': float(m_macro.group(1)),
                            'recall': float(m_macro.group(2)),
                            'f1-score': float(m_macro.group(3)),
                            'support': int(m_macro.group(4))
                        }
                    m_weighted = re.match(r"^weighted avg\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+(\d+)$", line)
                    if m_weighted:
                        overall_report['weighted avg'] = {
                            'precision': float(m_weighted.group(1)),
                            'recall': float(m_weighted.group(2)),
                            'f1-score': float(m_weighted.group(3)),
                            'support': int(m_weighted.group(4))
                        }
            data['report']['classes'] = class_report
            data['report']['overall'] = overall_report

        return data
    
    def compute_slope(self, epochs_list, metric_key, smoothing_window=5):
        """
        Computes the slope (improvement per epoch) for a given accuracy curve.
        Optionally smooths the data using a moving average.
        
        :param epochs_list: List of epoch dictionaries containing an 'epoch' key.
        :param metric_key: The metric to compute the slope for ('acc' or 'val_acc').
        :param smoothing_window: Size of the window for moving average smoothing.
        :return: Slope (average improvement per epoch).
        """
        # Extract epoch numbers and corresponding metric values.
        epochs = [ep['epoch'] for ep in epochs_list]
        accuracies = [ep[metric_key] for ep in epochs_list]
        # Smooth the curve if needed.
        if smoothing_window > 1 and len(accuracies) >= smoothing_window:
            series = pd.Series(accuracies)
            smoothed = series.rolling(window=smoothing_window, min_periods=1, center=True).mean()
            accuracies = smoothed.values
        X = np.array(epochs).reshape(-1, 1)
        y = np.array(accuracies)
        reg = LinearRegression().fit(X, y)
        return reg.coef_[0]


    def produceReport(self):
        """
        Produces a comparative report that shows differences in:
          - Final epoch metrics (loss, accuracy, validation loss, and validation accuracy)
          - Classification report per class (T0, T1, T2) differences (precision, recall, F1)
          - Slope (i.e. average improvement per epoch) differences for training and validation accuracy
        The reported differences are computed as: (cluster model metric) - (baseline model metric).
        """
        baseline_epochs = self.baseline_data.get('epochs', [])
        baseline_final = baseline_epochs[-1] if baseline_epochs else None
        baseline_report = self.baseline_data.get('report', {})

        if not baseline_final or not baseline_report:
            print("Error: Baseline log did not contain required metrics.")
            return

        base_classes = baseline_report.get('classes', {})

        report_lines = []

        report_lines.append("========= Model Report Analyis =========")
        # Compute baseline slopes for training and validation accuracy.

        report_lines.append(" - Baseline Model")
        baseline_train_slope = self.compute_slope(baseline_epochs, 'acc')
        baseline_val_slope = self.compute_slope(baseline_epochs, 'val_acc')

        report_lines.append(" - Baseline Learning Rate (Average Improvement per Epoch) ===")
        report_lines.append(f"Baseline Training Acc Slope: {baseline_train_slope:.4f} per epoch")
        report_lines.append(f"Baseline Validation Acc Slope: {baseline_val_slope:.4f} per epoch\n")
        report_lines.append(" - End Baseline Model\n")

        report_lines.append("Beginning Cluster Models")
        # For each clustered model, compare with baseline.
        for idx, cluster_data in enumerate(self.clustered_data):
            cluster_epochs = cluster_data.get('epochs', [])
            cluster_final = cluster_epochs[-1] if cluster_epochs else None
            cluster_report = cluster_data.get('report', {})

            if not cluster_final or not cluster_report:
                report_lines.append(f"Cluster Model #{idx+1}: Missing metrics, skipping comparison.\n")
                continue

            report_lines.append(f"=== Comparison: Cluster Model #{idx+1} vs Baseline ===")
            # Compute differences in final epoch metrics.
            diff_loss = cluster_final['loss'] - baseline_final['loss']
            diff_acc = cluster_final['acc'] - baseline_final['acc']
            diff_val_loss = cluster_final['val_loss'] - baseline_final['val_loss']
            diff_val_acc = cluster_final['val_acc'] - baseline_final['val_acc']

            report_lines.append("Differences in Final Epoch Metrics (Cluster - Baseline):")
            report_lines.append(f"  Loss diff: {diff_loss:.4f}")
            report_lines.append(f"  Acc diff: {diff_acc:.4f}")
            report_lines.append(f"  Val Loss diff: {diff_val_loss:.4f}")
            report_lines.append(f"  Val Acc diff: {diff_val_acc:.4f}")

            # Compute slopes for training and validation accuracy.
            cluster_train_slope = self.compute_slope(cluster_epochs, 'acc')
            cluster_val_slope = self.compute_slope(cluster_epochs, 'val_acc')
            diff_train_slope = cluster_train_slope - baseline_train_slope
            diff_val_slope = cluster_val_slope - baseline_val_slope

            report_lines.append("\nDifferences in Learning Rate (Slope) (Cluster - Baseline):")
            report_lines.append(f"  Training Acc Slope diff: {diff_train_slope:+.4f} per epoch")
            report_lines.append(f"  Validation Acc Slope diff: {diff_val_slope:+.4f} per epoch")

            # Compare classification report differences for each class.
            cluster_classes = cluster_report.get('classes', {})
            report_lines.append("\nDifferences in Classification Report per Class (Cluster - Baseline):")
            for cls in ['T0 (Rest)', 'T1 (Left-Hand)', 'T2 (Right-Hand)']:
                if cls in base_classes and cls in cluster_classes:
                    diff_prec = cluster_classes[cls]['precision'] - base_classes[cls]['precision']
                    diff_recall = cluster_classes[cls]['recall'] - base_classes[cls]['recall']
                    diff_f1 = cluster_classes[cls]['f1-score'] - base_classes[cls]['f1-score']
                    report_lines.append(
                        f"{cls} - Precision diff: {diff_prec:.4f}, "
                        f"Recall diff: {diff_recall:.4f}, F1 diff: {diff_f1:.4f}"
                    )
            report_lines.append("\n" + "=" * 50 + "\n")

        full_report = "\n".join(report_lines)
        print(full_report)

         # Naming convention: baseline_{baseline_epochs}_cluster_{cluster_epochs}_{n_groups}groups.txt
        baseline_epoch_count = len(baseline_epochs)
        # Assuming that all cluster models are trained for the same number of epochs, take the first one.
        if self.clustered_data and self.clustered_data[0].get('epochs', []):
            cluster_epoch_count = len(self.clustered_data[0]['epochs'])
        else:
            cluster_epoch_count = "unknown"
        n_groups = len(self.clustered_data)
        file_name = f"baseline_{baseline_epoch_count}_cluster_{cluster_epoch_count}_{n_groups}groups.txt"
        report_dir = "./logs"
        os.makedirs(report_dir, exist_ok=True)
        report_file = os.path.join(report_dir, file_name)
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(full_report)
        print(f"Report saved to {report_file}")


    
    
