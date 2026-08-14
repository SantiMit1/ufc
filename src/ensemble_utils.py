import copy
import numpy as np
from sklearn.base import clone
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import TimeSeriesSplit


def _logit(p):
    p = np.clip(np.asarray(p, dtype=float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class PlattCalibrator:
    """Platt scaling: logistic regression on the log-odds of the raw probability."""

    def __init__(self, C=1.0, max_iter=3000):
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        from sklearn.linear_model import LogisticRegression

        X = np.asarray(X)
        logit = _logit(X[:, 0])
        self.lr_ = LogisticRegression(C=self.C, max_iter=self.max_iter, solver="lbfgs")
        self.lr_.fit(logit.reshape(-1, 1), np.asarray(y))
        return self

    def predict(self, X):
        X = np.asarray(X)
        logit = _logit(X[:, 0])
        return self.lr_.predict_proba(logit.reshape(-1, 1))[:, 1]


class ChronologicalStackingEnsemble:
    def __init__(self, estimators, final_estimator, cv, calibrators=None):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.calibrators = calibrators
        self.calibrator_name = None

    @staticmethod
    def _as_frame(X):
        return X if hasattr(X, "iloc") else np.asarray(X)

    def _select_rows(self, X, idx):
        return X.iloc[idx] if hasattr(X, "iloc") else X[idx]

    def fit(self, X, y):
        X_frame = self._as_frame(X)
        y_array = np.asarray(y)

        oof_parts = []
        target_parts = []

        for train_idx, val_idx in self.cv.split(X_frame):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            X_train_fold = self._select_rows(X_frame, train_idx)
            X_val_fold = self._select_rows(X_frame, val_idx)
            y_train_fold = y_array[train_idx]
            fold_predictions = []

            for _, estimator in self.estimators:
                fitted = clone(estimator)
                fitted.fit(X_train_fold, y_train_fold)
                fold_proba = fitted.predict_proba(X_val_fold)
                fold_predictions.append(fold_proba[:, 1] if fold_proba.ndim > 1 else fold_proba)

            oof_parts.append(np.column_stack(fold_predictions))
            target_parts.append(y_array[val_idx])

        if not oof_parts:
            raise ValueError("Chronological stacking needs at least one non-empty validation fold")

        self.oof_features_ = np.vstack(oof_parts)
        self.oof_targets_ = np.concatenate(target_parts)

        self.named_estimators_ = {}
        for name, estimator in self.estimators:
            fitted = clone(estimator)
            fitted.fit(X_frame, y_array)
            self.named_estimators_[name] = fitted

        final_estimator = clone(self.final_estimator)
        final_estimator.fit(self.oof_features_, self.oof_targets_)
        self.final_estimator_ = final_estimator

        if self.calibrators:
            self._fit_calibrators(X_frame, y_array)
        return self

    def _fit_calibrators(self, X, y):
        outer = TimeSeriesSplit(n_splits=5)
        p_oof = []
        y_oof = []
        for train_idx, val_idx in outer.split(X):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue
            X_tr = self._select_rows(X, train_idx)
            X_va = self._select_rows(X, val_idx)
            fold = ChronologicalStackingEnsemble(
                self.estimators, self.final_estimator, self.cv, calibrators=None)
            fold.fit(X_tr, y[train_idx])
            p_oof.append(fold.predict_proba(X_va)[:, 1])
            y_oof.append(y[val_idx])

        self.oof_calib_probs_ = np.concatenate(p_oof)
        self.oof_calib_targets_ = np.concatenate(y_oof)

        # Score calibrators on a chronological holdout of the OOF points so the
        # choice is made with no lookahead and never touches the test set.
        # OOF probs are concatenated fold-by-fold (each val block is later than
        # the previous one), so the last portion is the most recent training-era
        # fights.
        n_oof = len(self.oof_calib_probs_)
        cal_split = int(n_oof * 0.8)
        self.calibrator_scores_ = {}
        if n_oof >= 40 and cal_split >= 10 and n_oof - cal_split >= 10:
            cal_fit_p = self.oof_calib_probs_[:cal_split]
            cal_fit_y = self.oof_calib_targets_[:cal_split]
            cal_eval_p = self.oof_calib_probs_[cal_split:]
            cal_eval_y = self.oof_calib_targets_[cal_split:]
            for name, template in self.calibrators.items():
                calib = copy.deepcopy(template)
                calib.fit(cal_fit_p.reshape(-1, 1), cal_fit_y)
                p_pred = np.clip(
                    np.asarray(calib.predict(cal_eval_p.reshape(-1, 1)), dtype=float).ravel(),
                    1e-6, 1 - 1e-6,
                )
                self.calibrator_scores_[name] = {
                    "oof_log_loss": float(log_loss(cal_eval_y, p_pred)),
                    "oof_brier": float(brier_score_loss(cal_eval_y, p_pred)),
                }

        self.calibrators_ = {}
        for name, template in self.calibrators.items():
            calib = copy.deepcopy(template)
            calib.fit(self.oof_calib_probs_.reshape(-1, 1), self.oof_calib_targets_)
            self.calibrators_[name] = calib

    def _stack_features(self, X):
        X_frame = self._as_frame(X)
        base_predictions = []
        for _, estimator in self.named_estimators_.items():
            proba = estimator.predict_proba(X_frame)
            base_predictions.append(proba[:, 1] if proba.ndim > 1 else proba)
        return np.column_stack(base_predictions)

    def predict_proba(self, X):
        meta_features = self._stack_features(X)
        full = self.final_estimator_.predict_proba(meta_features)
        name = getattr(self, "calibrator_name", None)
        if name and hasattr(self, "calibrators_") and name in self.calibrators_:
            raw = full[:, 1]
            calib = self.calibrators_[name].predict(raw.reshape(-1, 1))
            calib = np.clip(np.asarray(calib, dtype=float).ravel(), 1e-6, 1 - 1e-6)
            return np.column_stack([1.0 - calib, calib])
        return full

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)
