# File: ModelTester.py
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score


class ModelTester:
    def __init__(self, trainer, f1_average: str = "macro"):
        # ------- shared initialisation -----------------------------------
        self.model      = trainer.model
        self.train_epo  = getattr(trainer, "train_epochs", None)  # NEW
        self.val_epo    = getattr(trainer, "val_epochs",  None)
        self.test_epo   = getattr(trainer, "test_epochs", None)
        self.log_file   = trainer.log_file
        self.f1_average = f1_average

        if self.model is None or self.test_epo is None:
            raise ValueError("Trainer must have model + test_epochs.")

        # ---------- cache predictions ------------------------------------
        if self.train_epo is not None:
            (self._train_pred,
             self._train_true,
             self._train_subj) = self._predict(self.train_epo)

        (self._test_pred,
         self._test_true,
         self._test_subj) = self._predict(self.test_epo)

        if self.val_epo is not None:
            (self._val_pred,
             self._val_true,
             self._val_subj) = self._predict(self.val_epo)

    # ---------------------------------------------------------------------
    def _predict(self, epo):
        X = epo.get_data()[..., np.newaxis]
        y_pred = np.argmax(self.model.predict(X, verbose=0), axis=1)
        y_true = epo.events[:, 2] - 1
        subjects = epo.metadata["subject"].values
        return y_pred, y_true, subjects

    # ---------------------------------------------------------------------
    def overall_report(self):
        labels = ["T0 (Rest)", "T1 (Left-Hand)", "T2 (Right-Hand)"]
        rep = classification_report(self._test_true,
                                    self._test_pred,
                                    target_names=labels)
        print("\nClassification Report (test set):\n")
        print(rep)
        with open(self.log_file, "a") as f:
            f.write("\nClassification Report (test set):\n")
            f.write(rep + "\n")

    # ---------------------------------------------------------------------
    def per_subject_metrics(self,
                            include_training: bool = True,
                            include_validation: bool = True,
                            zero_division: int = 0):
        """
        Print per-subject metrics for train, test, and (optionally) val sets.
        Returns the test-set DataFrame.
        """
        # ----- helper ----------------------------------------------------
        def table(y_true, y_pred, subjects):
            rows = []
            for s in np.unique(subjects):
                m = subjects == s
                rows.append({
                    "subject":  int(s),
                    "n_epochs": int(m.sum()),
                    "accuracy": accuracy_score(y_true[m], y_pred[m]),
                    f"f1_{self.f1_average}":
                        f1_score(y_true[m], y_pred[m],
                                 average=self.f1_average,
                                 zero_division=zero_division)
                })
            return pd.DataFrame(rows).sort_values("subject")

        # ---------- training set -----------------------------------------
        if include_training and self.train_epo is not None:
            df_train = table(self._train_true, self._train_pred, self._train_subj)
            print("\nPer-subject metrics (training set):")
            print(df_train.to_string(index=False, formatters={
                "accuracy": "{:.3f}".format,
                f"f1_{self.f1_average}": "{:.3f}".format}))
            with open(self.log_file, "a") as f:
                f.write("\nPer-subject metrics (training set):\n")
                f.write(df_train.to_string(index=False) + "\n")

        # ---------- test set ---------------------------------------------
        df_test = table(self._test_true, self._test_pred, self._test_subj)
        print("\nPer-subject metrics (test set):")
        print(df_test.to_string(index=False, formatters={
            "accuracy": "{:.3f}".format,
            f"f1_{self.f1_average}": "{:.3f}".format}))
        with open(self.log_file, "a") as f:
            f.write("\nPer-subject metrics (test set):\n")
            f.write(df_test.to_string(index=False) + "\n")

        # ---------- validation set ---------------------------------------
        if include_validation and self.val_epo is not None:
            df_val = table(self._val_true, self._val_pred, self._val_subj)
            print("\nPer-subject metrics (validation set):")
            print(df_val.to_string(index=False, formatters={
                "accuracy": "{:.3f}".format,
                f"f1_{self.f1_average}": "{:.3f}".format}))
            with open(self.log_file, "a") as f:
                f.write("\nPer-subject metrics (validation set):\n")
                f.write(df_val.to_string(index=False) + "\n")

        return df_test  
