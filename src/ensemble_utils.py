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


class CappedCalibrator:
    """Wrap a base calibrator and cap how much it can inflate underdog probabilities.

    For raw probabilities below 0.5 the calibrated output cannot exceed
    ``raw + cap_increase`` (a monotone cap, so the mapping stays non-decreasing).
    This prevents overconfidence in underdogs when the base model's low tail is
    already too generous. ``cap_increase=0`` forbids any upward correction below
    the 0.5 line.
    """

    def __init__(self, base, cap_increase=0.0):
        self.base = base
        self.cap_increase = cap_increase

    def fit(self, X, y):
        self.base.fit(X, y)
        return self

    def predict(self, X):
        X = np.asarray(X)
        raw = X[:, 0]
        p = np.asarray(self.base.predict(X), dtype=float)
        cap = np.where(raw < 0.5, raw + self.cap_increase, 1.0)
        return np.minimum(p, cap)


class ChronologicalStackingEnsemble:
    def __init__(self, estimators, final_estimator, cv, calibrators=None,
                 calib_recent_fraction=0.5):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv
        self.calibrators = calibrators
        self.calibrator_name = None
        self.calib_recent_fraction = calib_recent_fraction

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

        # Calibration should match the era the model will be used on (the most
        # recent fights). OOF probs span the whole training era; fitting
        # calibrators on all of it bakes in stale relationships (e.g. underdogs
        # were far less extreme pre-2020), which inflates underdog probabilities
        # on recent fights. So calibrators are fitted AND selected on only the
        # most recent ``calib_recent_fraction`` of the OOF points. OOF probs are
        # concatenated fold-by-fold, so later indices are later fights.
        n_oof = len(self.oof_calib_probs_)
        win = int(n_oof * self.calib_recent_fraction)
        win_start = n_oof - win
        self.calibrator_scores_ = {}
        if win >= 40:
            # Selection holdout: fit on the first 80% of the window, score on the
            # last 20% (most recent) — no lookahead, never the test set.
            sel_end = win_start + int(0.8 * win)
            if sel_end - win_start >= 10 and n_oof - sel_end >= 10:
                cal_fit_p = self.oof_calib_probs_[win_start:sel_end]
                cal_fit_y = self.oof_calib_targets_[win_start:sel_end]
                cal_eval_p = self.oof_calib_probs_[sel_end:]
                cal_eval_y = self.oof_calib_targets_[sel_end:]
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
            calib.fit(self.oof_calib_probs_[win_start:].reshape(-1, 1),
                      self.oof_calib_targets_[win_start:])
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
