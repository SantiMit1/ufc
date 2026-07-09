import numpy as np
from sklearn.base import clone


class ChronologicalStackingEnsemble:
    def __init__(self, estimators, final_estimator, cv):
        self.estimators = estimators
        self.final_estimator = final_estimator
        self.cv = cv

    @staticmethod
    def _as_frame(X):
        return X if hasattr(X, "iloc") else np.asarray(X)

    def fit(self, X, y):
        X_frame = self._as_frame(X)
        y_array = np.asarray(y)

        oof_parts = []
        target_parts = []

        for train_idx, val_idx in self.cv.split(X_frame):
            if len(train_idx) == 0 or len(val_idx) == 0:
                continue

            if hasattr(X_frame, "iloc"):
                X_train_fold = X_frame.iloc[train_idx]
                X_val_fold = X_frame.iloc[val_idx]
            else:
                X_train_fold = X_frame[train_idx]
                X_val_fold = X_frame[val_idx]
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
        return self

    def _stack_features(self, X):
        X_frame = self._as_frame(X)
        base_predictions = []
        for _, estimator in self.named_estimators_.items():
            proba = estimator.predict_proba(X_frame)
            base_predictions.append(proba[:, 1] if proba.ndim > 1 else proba)
        return np.column_stack(base_predictions)

    def predict_proba(self, X):
        meta_features = self._stack_features(X)
        return self.final_estimator_.predict_proba(meta_features)

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)