#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: yl254115
"""

from sklearn.linear_model import RidgeCV
from sklearn.model_selection import GridSearchCV, KFold
from mne.decoding import TimeDelayingRidge, ReceptiveField
import numpy as np


class TimeDelayingRidgeCV(TimeDelayingRidge, RidgeCV):
    def __init__(self, tmin, tmax, sfreq, reg_type='ridge',
                 alphas=(0.1, 1.0, 10.0), *,
                 fit_intercept=True, normalize=False, scoring='r2',
                 cv=5, store_cv_values=False,
                 alpha_per_target=False, n_jobs=1,
                 edge_correction=True):
        self.alphas = np.asarray(alphas)
        self.fit_intercept = fit_intercept
        self.normalize = normalize
        self.scoring = scoring
        self.cv = cv
        self.store_cv_values = store_cv_values
        self.alpha_per_target = alpha_per_target
        self.tmin = tmin
        self.tmax = tmax
        self.sfreq = sfreq
        self.reg_type = reg_type
        self.n_jobs = n_jobs
        self.edge_correction = edge_correction

    def fit(self, X, y, sample_weight=None):
        """Fit Ridge regression model with cv.
        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Training data. If using GCV, will be cast to float64
            if necessary.
        y : ndarray of shape (n_samples,) or (n_samples, n_targets)
            Target values. Will be cast to X's dtype if necessary.
        sample_weight : float or ndarray of shape (n_samples,), default=None
            Individual weights for each sample. If given a float, every sample
            will have the same weight.
        Returns
        -------
        self : object
        Notes
        -----
        When sample_weight is provided, the selected hyperparameter may depend
        on whether we use leave-one-out cross-validation (cv=None or cv='auto')
        or another form of cross-validation, because only leave-one-out
        cross-validation takes the sample weights into account when computing
        the validation score.
        """
        cv = self.cv
        if cv is None:
            raise "Implementation error"
        else:
            if self.store_cv_values:
                raise ValueError("cv!=None and store_cv_values=True"
                                 " are incompatible")
            estimator = TimeDelayingRidge(tmin=self.tmin,
                                          tmax=self.tmax,
                                          sfreq=self.sfreq,
                                          reg_type=self.reg_type,
                                          n_jobs=self.n_jobs,
                                          fit_intercept=self.fit_intercept,
                                          edge_correction=self.edge_correction)

            rf = ReceptiveField(self.tmin, self.tmax, self.sfreq,
                                estimator=estimator,
                                scoring='corrcoef',
                                n_jobs=-1)

            n_outputs = y.shape[-1]
            n_features = X.shape[-1]
            n_delays = len(np.arange(int(np.round(self.tmin * self.sfreq)),
                                     int(np.round(self.tmax * self.sfreq) +
                                         1)))
            self.coef_ = np.empty((n_outputs, n_features, n_delays)) * np.nan
            self.intercept_ = np.nan
            self.best_score_ = np.asarray([float('-inf')]*n_outputs)
            self.alpha_ = np.zeros(n_outputs)

            if self.alpha_per_target:
                inner_cv = KFold(n_splits=cv, shuffle=True, random_state=0)
                for i_alpha, alpha in enumerate(self.alphas):
                    scores = np.empty((cv, n_outputs)) * np.nan
                    for i_split, (train, test) in enumerate(inner_cv.split(
                            X.transpose([1, 2, 0]),
                            y.transpose([1, 2, 0]))):
                        rf.fit(X[:, train, :], y[:, train, :])
                        scores[i_split] = rf.score(X[:, test, :],
                                                   y[:, test, :])

                    IXs_output_update = \
                        np.mean(scores, axis=0) > self.best_score_
                    if IXs_output_update.any():
                        self.coef_[IXs_output_update, :, :] = \
                            rf.coef_[IXs_output_update, :, :]
                        self.best_score_[IXs_output_update] = \
                            np.mean(scores, axis=0)[IXs_output_update]
                        self.alpha_[IXs_output_update] = alpha

                        self.intercept_ = 0.

            else:  # Separately optimizes alpha for each output
                # gs = GridSearchCV(rf, parameters...
                raise "Implementation error"

        return self
