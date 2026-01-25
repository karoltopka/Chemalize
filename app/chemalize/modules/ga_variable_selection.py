"""
Genetic Algorithm for Variable Selection in Multiple Linear Regression
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score, train_test_split, LeaveOneOut, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.feature_selection import VarianceThreshold
from scipy import stats
import random
import matplotlib.pyplot as plt
import seaborn as sns
import os
from app.utils.watermark import add_watermark_matplotlib_after_plot


class GeneticAlgorithmSelector:
    """
    Genetic Algorithm for selecting optimal variable subset for MLR
    """

    def __init__(self,
                 n_variables=None,
                 correlation_threshold=0.95,
                 mutation_rate=0.1,
                 random_models_ratio=0.1,
                 population_size=50,
                 n_iterations=100,
                 max_retries=3,
                 cv_folds=5,
                 cv_folds_validation=3,
                 early_stop_rounds=20,
                 early_stop_min_delta=1e-4,
                 metrics_interval=5,
                 shuffle_cv=True,
                 cv_n_jobs=-1,
                 use_validation=False,
                 test_normality=True,
                 normality_alpha=0.05,
                 n_best_models=5,
                 check_ad=False,
                 ad_threshold=100.0,
                 split_method=None,
                 split_params=None,
                 split_train_idx=None,
                 split_test_idx=None,
                 metrics_X=None,
                 metrics_y=None,
                 min_split_r2=0.68,
                 min_split_q2=0.68,
                 progress_callback=None,
                 internal_cv_type='kfold',
                 sorted_step=5,
                 random_state=42):
        """
        Initialize GA selector

        Parameters:
        -----------
        n_variables : int or None
            Target number of variables to select. If None, will be optimized
        correlation_threshold : float
            Maximum allowed correlation between features (default: 0.95)
        mutation_rate : float
            Probability of mutation (default: 0.1)
        random_models_ratio : float
            Ratio of random models in population (default: 0.1)
        population_size : int
            Size of the population (default: 50)
        n_iterations : int
            Number of generations (default: 100)
        max_retries : int
            Maximum number of algorithm retries (default: 3)
        cv_folds : int
            Number of cross-validation folds (default: 5)
        cv_folds_validation : int
            Number of CV folds for validation set (default: 3)
        early_stop_rounds : int
            Stop if no improvement for this many generations (default: 20)
        early_stop_min_delta : float
            Minimum improvement to reset early stop counter (default: 1e-4)
        metrics_interval : int
            Compute basic metrics every N generations (default: 5)
        shuffle_cv : bool
            Whether to shuffle CV splits (default: True)
        cv_n_jobs : int
            Parallel jobs for CV scoring (default: -1)
        use_validation : bool
            Whether to use validation set (default: False)
        test_normality : bool
            Whether to test residuals normality (default: True)
        normality_alpha : float
            Alpha for normality test (default: 0.05)
        n_best_models : int
            Number of best models to keep (default: 5)
        check_ad : bool
            Whether to check Applicability Domain for training set (default: False)
        ad_threshold : float
            Minimum required AD coverage for training set in percent (default: 100.0)
            Models with AD_train < ad_threshold will be rejected
        split_method : str or None
            Split method for basic R2/Q2 metrics (uses user-defined split when available)
        split_params : dict or None
            Parameters for split_method
        split_train_idx : list[int] or None
            Train indices for user-defined split (relative to metrics data)
        split_test_idx : list[int] or None
            Test indices for user-defined split (relative to metrics data)
        metrics_X : array-like or DataFrame or None
            Full feature matrix to compute R2/Q2 on user split (defaults to GA training data)
        metrics_y : array-like or Series or None
            Full target array to compute R2/Q2 on user split (defaults to GA training data)
        min_split_r2 : float
            Minimum split R2 required to continue detailed evaluation (default: 0.68)
        min_split_q2 : float
            Minimum split Q2 required to continue detailed evaluation (default: 0.68)
        internal_cv_type : str
            Type of internal CV for fitness calculation: 'kfold' or 'sorted' (default: 'kfold')
        sorted_step : int
            For sorted CV: take every nth element for test set (default: 5)
            Number of iterations uses cv_folds parameter.
        random_state : int
            Random state for reproducibility
        """
        self.n_variables = n_variables
        self.correlation_threshold = correlation_threshold
        self.mutation_rate = mutation_rate
        self.random_models_ratio = random_models_ratio
        self.population_size = population_size
        self.n_iterations = n_iterations
        self.max_retries = max_retries
        self.cv_folds = cv_folds
        self.cv_folds_validation = cv_folds_validation
        self.early_stop_rounds = early_stop_rounds
        self.early_stop_min_delta = early_stop_min_delta
        self.metrics_interval = metrics_interval
        self.shuffle_cv = shuffle_cv
        self.cv_n_jobs = cv_n_jobs
        self.use_validation = use_validation
        self.test_normality = test_normality
        self.normality_alpha = normality_alpha
        self.n_best_models = n_best_models
        self.check_ad = check_ad
        self.ad_threshold = ad_threshold
        self.split_method = split_method
        self.split_params = split_params or {}
        self._split_train_idx = split_train_idx
        self._split_test_idx = split_test_idx
        self._metrics_X = metrics_X
        self._metrics_y = metrics_y
        self.min_split_r2 = min_split_r2
        self.min_split_q2 = min_split_q2
        self.progress_callback = progress_callback
        self.internal_cv_type = internal_cv_type
        self.sorted_step = sorted_step
        self.random_state = random_state

        self.best_features_ = None
        self.best_score_ = -np.inf
        self.fitness_history_ = []
        self.feature_names_ = None
        self.best_models_ = []  # List to store top N models
        self._corr_matrix = None
        self._cv = None
        self._cv_validation = None
        self._validation_X = None
        self._validation_y = None
        self.no_models_reason_ = None

        np.random.seed(random_state)
        random.seed(random_state)

    def _make_cv(self, n_samples, n_splits):
        """Create a CV splitter safe for small sample sizes."""
        if n_samples is None:
            return None
        n_samples = int(n_samples)
        n_splits = int(n_splits)
        if n_samples < 2:
            return None
        n_splits = min(n_splits, n_samples)
        if n_splits < 2:
            return None
        if self.shuffle_cv:
            return KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        return KFold(n_splits=n_splits, shuffle=False)

    def _create_sorted_cv_splits(self, y, step=None, n_iterations=None):
        """
        Create CV splits for sorted stratified cross-validation.

        The data is sorted by Y values, then every nth element goes to the test set.
        Multiple iterations are performed with different starting offsets.

        Args:
            y: Target values (used for sorting)
            step: Take every nth element for test set (default: self.sorted_step)
            n_iterations: Number of iterations with different offsets (default: self.cv_folds)

        Returns:
            list of tuples (train_indices, test_indices)
        """
        if step is None:
            step = self.sorted_step
        if n_iterations is None:
            n_iterations = self.cv_folds  # Use cv_folds as number of iterations

        y_array = np.array(y).ravel()
        sorted_indices = np.argsort(y_array)
        splits = []

        # Limit iterations to step (can't have more offsets than step size)
        actual_iterations = min(n_iterations, step)

        for offset in range(actual_iterations):
            # Test indices: every step-th element starting from offset
            test_indices = sorted_indices[offset::step]
            # Train indices: all others
            test_set = set(test_indices)
            train_indices = np.array([i for i in sorted_indices if i not in test_set])
            splits.append((train_indices, test_indices))

        return splits

    def _calculate_r2cv(self, X, y, feature_mask):
        """
        Calculate cross-validated R² (R²cv) using internal CV method (k-fold or sorted).

        This is the main fitness metric used during GA optimization.

        Args:
            X: Feature matrix
            y: Target values
            feature_mask: Boolean mask for selected features

        Returns:
            float: R²cv value, or -np.inf if calculation fails
        """
        selected_indices = np.where(feature_mask)[0]
        if len(selected_indices) == 0:
            return -np.inf

        X_selected = X[:, selected_indices]

        try:
            model = LinearRegression()

            if self.internal_cv_type == 'sorted':
                # Sorted stratified CV
                splits = self._create_sorted_cv_splits(y)
                if not splits:
                    return -np.inf

                cv_scores = []
                for train_idx, test_idx in splits:
                    if len(train_idx) == 0 or len(test_idx) == 0:
                        continue
                    X_train = X_selected[train_idx]
                    y_train = y[train_idx]
                    X_test = X_selected[test_idx]
                    y_test = y[test_idx]

                    model.fit(X_train, y_train)
                    score = model.score(X_test, y_test)
                    cv_scores.append(score)

                if not cv_scores:
                    return -np.inf
                return float(np.mean(cv_scores))

            else:
                # Default: k-fold CV
                cv = self._cv or self._make_cv(len(y), self.cv_folds)
                if cv is None:
                    # Fallback: train on all, return R² (not ideal but avoids crash)
                    model.fit(X_selected, y)
                    return float(model.score(X_selected, y))

                cv_scores = cross_val_score(
                    model, X_selected, y,
                    cv=cv,
                    scoring='r2',
                    n_jobs=self.cv_n_jobs
                )
                return float(np.mean(cv_scores))

        except Exception:
            return -np.inf

    def _normalize_split_indices(self, n_samples):
        """Normalize and clamp split indices to the metrics dataset size."""
        def _normalize(indices):
            if indices is None:
                return None
            try:
                normalized = [int(i) for i in indices]
            except TypeError:
                return None
            return [i for i in normalized if 0 <= i < n_samples]

        self._split_train_idx = _normalize(self._split_train_idx)
        self._split_test_idx = _normalize(self._split_test_idx)

        if self._split_train_idx is not None and len(self._split_train_idx) == 0:
            self._split_train_idx = None
        if self._split_test_idx is None:
            self._split_test_idx = None

    def _check_correlation(self, X, feature_mask):
        """Check if selected features meet correlation threshold"""
        if self.correlation_threshold >= 1.0:
            return True

        selected_indices = np.where(feature_mask)[0]
        if len(selected_indices) < 2:
            return True

        if self._corr_matrix is not None:
            corr_matrix = self._corr_matrix[np.ix_(selected_indices, selected_indices)]
            upper = corr_matrix[np.triu_indices_from(corr_matrix, k=1)]
            return not np.any(upper > self.correlation_threshold)

        X_selected = X[:, selected_indices]
        corr_matrix = np.corrcoef(X_selected.T)
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)

        # Check upper triangle of correlation matrix
        for i in range(len(selected_indices)):
            for j in range(i + 1, len(selected_indices)):
                if abs(corr_matrix[i, j]) > self.correlation_threshold:
                    return False
        return True

    def _calculate_ad_coverage(self, X, y, feature_mask):
        """
        Calculate Applicability Domain coverage for training set using Williams Plot method.

        The AD is defined by:
        - Leverage threshold: h* = 3(p+1)/n where p is number of variables, n is number of samples
        - Residual threshold: |standardized residual| < 3

        A sample is within AD if: leverage < h* AND |std_residual| < 3

        Parameters:
        -----------
        X : array
            Feature matrix (training data)
        y : array
            Target values (training data)
        feature_mask : array
            Boolean mask for selected features

        Returns:
        --------
        float
            AD coverage percentage (0-100). Returns 0 if calculation fails.
        """
        selected_indices = np.where(feature_mask)[0]
        if len(selected_indices) == 0:
            return 0.0

        X_selected = X[:, selected_indices]
        n_samples = X_selected.shape[0]
        n_features = X_selected.shape[1]

        # Need at least n_features + 2 samples for meaningful calculation
        if n_samples <= n_features + 1:
            return 0.0

        try:
            # Fit model using statsmodels OLS (same as coefficient calculation)
            X_with_const = sm.add_constant(X_selected)
            sm_model = sm.OLS(y, X_with_const).fit()
            y_pred = sm_model.predict(X_with_const)

            # Calculate hat matrix for leverage FIRST (needed for studentized residuals)
            # H = X(X'X)^(-1)X' - use X without constant for leverage calculation
            # Add small regularization for numerical stability
            XtX = X_selected.T @ X_selected
            reg_factor = 1e-8 * np.trace(XtX) / n_features if n_features > 0 else 1e-8
            XtX_reg = XtX + reg_factor * np.eye(n_features)

            try:
                XtX_inv = np.linalg.inv(XtX_reg)
            except np.linalg.LinAlgError:
                XtX_inv = np.linalg.pinv(XtX_reg)

            # Calculate leverage (diagonal of hat matrix)
            leverage = np.sum((X_selected @ XtX_inv) * X_selected, axis=1)

            # Calculate residuals
            residuals = y - y_pred

            # Calculate internally studentized residuals: r_i = e_i / (s * sqrt(1 - h_i))
            # s = sqrt(MSE) = sqrt(SSE / (n - p - 1))
            SSE = np.sum(residuals ** 2)
            df = n_samples - n_features - 1  # degrees of freedom (n - p - 1 for intercept)
            if df <= 0:
                df = 1  # Avoid division by zero
            MSE = SSE / df
            s = np.sqrt(MSE)

            # Studentized residuals
            # Avoid division by zero when leverage is close to 1
            leverage_factor = np.sqrt(np.maximum(1 - leverage, 1e-10))
            std_residuals = residuals / (s * leverage_factor)

            # Calculate h* threshold: h* = 3(p+1)/n where p is number of predictors
            h_star = 3 * (n_features + 1) / n_samples

            # Count samples within AD
            # Within AD if: leverage < h* AND |std_residual| < 3
            within_ad = (leverage < h_star) & (np.abs(std_residuals) < 3)
            ad_coverage = 100.0 * np.sum(within_ad) / n_samples

            return float(ad_coverage)

        except Exception as e:
            # If calculation fails, return 0 (model will be rejected if check_ad is True)
            return 0.0

    def _calculate_basic_metrics(self, feature_mask):
        """
        Calculate split-based metrics for a feature subset using user-defined split:
        R2 (train) and Q2 (test).
        """
        selected_indices = np.where(feature_mask)[0]

        if len(selected_indices) == 0 or self._metrics_X is None or self._metrics_y is None:
            return {
                'r2': 0.0,
                'q2': None,
                'valid': False
            }

        X_metrics = self._metrics_X
        y_metrics = self._metrics_y
        X_selected = X_metrics[:, selected_indices]

        try:
            model = LinearRegression()

            if self._split_train_idx is not None:
                train_idx = self._split_train_idx
                test_idx = self._split_test_idx or []

                if len(train_idx) == 0:
                    return {
                        'r2': 0.0,
                        'q2': None,
                        'valid': False
                    }

                X_train = X_selected[train_idx]
                y_train = y_metrics[train_idx]

                model.fit(X_train, y_train)
                r2 = float(model.score(X_train, y_train)) if len(y_train) > 1 else 0.0

                q2 = None
                if len(test_idx) > 0:
                    X_test = X_selected[test_idx]
                    y_test = y_metrics[test_idx]
                    y_pred_test = model.predict(X_test)
                    sse_test = np.sum((y_test - y_pred_test) ** 2)
                    tss_test = np.sum((y_test - np.mean(y_train)) ** 2)
                    if tss_test > 0:
                        q2 = float(1 - (sse_test / tss_test))

                return {
                    'r2': r2,
                    'q2': q2,
                    'valid': True
                }

            if self.split_method == 'kfold':
                n_folds = int(self.split_params.get('n_folds', self.cv_folds))
                shuffle = bool(self.split_params.get('shuffle', self.shuffle_cv))
                n_samples = len(y_metrics)
                n_splits = min(n_folds, n_samples)
                if n_splits < 2:
                    raise ValueError("Not enough samples for kfold metrics")
                if shuffle:
                    cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
                else:
                    cv = KFold(n_splits=n_splits, shuffle=False)

                train_r2 = []
                test_r2 = []
                for train_idx, test_idx in cv.split(X_selected):
                    X_train = X_selected[train_idx]
                    y_train = y_metrics[train_idx]
                    X_test = X_selected[test_idx]
                    y_test = y_metrics[test_idx]

                    model.fit(X_train, y_train)
                    train_r2.append(model.score(X_train, y_train))
                    test_r2.append(model.score(X_test, y_test))

                r2 = float(np.mean(train_r2)) if train_r2 else 0.0
                q2 = float(np.mean(test_r2)) if test_r2 else None

                return {
                    'r2': r2,
                    'q2': q2,
                    'valid': True
                }

            if self.split_method == 'loocv':
                if len(y_metrics) < 2:
                    raise ValueError("Not enough samples for loocv metrics")

                loo = LeaveOneOut()
                loo_predictions = []
                loo_actuals = []

                for train_idx, test_idx in loo.split(X_selected):
                    X_train = X_selected[train_idx]
                    y_train = y_metrics[train_idx]
                    X_test = X_selected[test_idx]
                    y_test = y_metrics[test_idx]

                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    loo_predictions.extend(y_pred)
                    loo_actuals.extend(y_test)

                loo_predictions = np.array(loo_predictions)
                loo_actuals = np.array(loo_actuals)
                press = np.sum((loo_actuals - loo_predictions) ** 2)
                tss = np.sum((loo_actuals - np.mean(loo_actuals)) ** 2)
                q2 = float(1 - (press / tss)) if tss > 0 else None

                model.fit(X_selected, y_metrics)
                r2 = float(model.score(X_selected, y_metrics)) if len(y_metrics) > 1 else 0.0
                return {
                    'r2': r2,
                    'q2': q2,
                    'valid': True
                }

            model.fit(X_selected, y_metrics)
            r2 = float(model.score(X_selected, y_metrics)) if len(y_metrics) > 1 else 0.0

            return {
                'r2': r2,
                'q2': r2,
                'valid': True
            }

        except Exception:
            return {
                'r2': 0.0,
                'q2': None,
                'valid': False
            }

    def _calculate_detailed_metrics(self, feature_mask, X_train, y_train):
        """
        Calculate detailed metrics for a feature subset on TRAIN data:
        - r2: R² on full TRAIN set
        - r2loo: Leave-One-Out R² on TRAIN set
        - rmse_tr: RMSE on full TRAIN set
        - ad_coverage: AD coverage for TRAIN set
        - intercept, coefficients: Model parameters

        Note: Q²ext and RMSE_ext are calculated AFTER model selection, not here.

        Returns:
        --------
        dict with keys: 'r2', 'r2loo', 'rmse_tr', 'ad_coverage', 'intercept', 'coefficients', 'valid'
        """
        selected_indices = np.where(feature_mask)[0]

        if len(selected_indices) == 0:
            return {'r2': 0.0, 'r2loo': 0.0, 'rmse_tr': 0.0, 'ad_coverage': 0.0,
                    'intercept': 0.0, 'coefficients': [], 'valid': False}

        X_selected = X_train[:, selected_indices]

        try:
            model = LinearRegression()

            # Fit model on full TRAIN set
            model.fit(X_selected, y_train)
            y_pred = model.predict(X_selected)

            # R² on full TRAIN set
            r2 = float(model.score(X_selected, y_train))

            # RMSE on full TRAIN set
            rmse_tr = float(np.sqrt(np.mean((y_train - y_pred) ** 2)))

            # R²loo (Leave-One-Out Cross-Validation on TRAIN)
            if len(y_train) >= 3:
                loo = LeaveOneOut()
                loo_predictions = []
                loo_actuals = []

                for train_idx, test_idx in loo.split(X_selected):
                    X_train_fold = X_selected[train_idx]
                    y_train_fold = y_train[train_idx]
                    X_test_fold = X_selected[test_idx]
                    y_test_fold = y_train[test_idx]

                    model_loo = LinearRegression()
                    model_loo.fit(X_train_fold, y_train_fold)
                    y_pred_loo = model_loo.predict(X_test_fold)

                    loo_predictions.extend(y_pred_loo)
                    loo_actuals.extend(y_test_fold)

                # Calculate R²loo = 1 - PRESS/TSS
                loo_predictions = np.array(loo_predictions)
                loo_actuals = np.array(loo_actuals)
                press = np.sum((loo_actuals - loo_predictions) ** 2)
                tss = np.sum((loo_actuals - np.mean(loo_actuals)) ** 2)
                r2loo = float(1 - (press / tss)) if tss > 0 else 0.0
            else:
                # Not enough samples for LOO
                r2loo = r2

            # Calculate AD coverage
            ad_coverage = self._calculate_ad_coverage(X_train, y_train, feature_mask)

            # Fit final model using statsmodels OLS (same as MLR module)
            X_with_const = sm.add_constant(X_selected)
            sm_model = sm.OLS(y_train, X_with_const).fit()
            intercept = float(sm_model.params[0])  # First param is intercept
            coefficients = [float(c) for c in sm_model.params[1:]]  # Rest are coefficients

            return {
                'r2': r2,
                'r2loo': r2loo,
                'rmse_tr': rmse_tr,
                'ad_coverage': float(ad_coverage),
                'intercept': intercept,
                'coefficients': coefficients,
                'valid': True
            }

        except Exception:
            return {'r2': 0.0, 'r2loo': 0.0, 'rmse_tr': 0.0, 'ad_coverage': 0.0,
                    'intercept': 0.0, 'coefficients': [], 'valid': False}

    def _calculate_r2cv_ext(self, feature_mask):
        """
        Calculate cross-validated R² on the external validation/test set.

        This metric (R²cv_ext) shows how well the model generalizes to unseen data
        using CV on the validation set.

        Returns:
            float or None: R²cv on validation set, or None if not available
        """
        if self._validation_X is None or self._validation_y is None:
            return None

        selected_indices = np.where(feature_mask)[0]
        if len(selected_indices) == 0:
            return None

        X_val_selected = self._validation_X[:, selected_indices]
        y_val = self._validation_y

        # Need at least 3 samples for CV
        if len(y_val) < 3:
            return None

        cv = self._cv_validation or self._make_cv(len(y_val), self.cv_folds_validation)
        if cv is None:
            return None

        try:
            model = LinearRegression()
            scores = cross_val_score(
                model, X_val_selected, y_val,
                cv=cv,
                scoring='r2',
                n_jobs=self.cv_n_jobs
            )
            if scores is None or len(scores) == 0:
                return None
            return float(np.mean(scores))
        except Exception:
            return None

    def _fitness_function(self, feature_mask, X_train, y_train, X_val=None, y_val=None):
        """
        Calculate fitness of a feature subset.

        Uses R²cv (cross-validated R² on TRAIN set) as the main criterion.
        This can be calculated using k-fold CV or sorted stratified CV depending on internal_cv_type.

        Returns higher score for better models.
        """
        selected_indices = np.where(feature_mask)[0]

        # Minimum 1 variable required
        if len(selected_indices) == 0:
            return -np.inf

        # Check target number of variables constraint
        if self.n_variables is not None:
            if len(selected_indices) != self.n_variables:
                # Penalize if not matching target number
                penalty = abs(len(selected_indices) - self.n_variables) * 0.1
                return -penalty

        # Check correlation constraint
        if not self._check_correlation(X_train, feature_mask):
            return -np.inf

        try:
            # Calculate R²cv using internal CV (k-fold or sorted)
            # Use the TRAIN data for internal CV
            if self._split_train_idx is not None and len(self._split_train_idx) > 0:
                # Use only training portion for internal CV
                X_for_cv = self._metrics_X[self._split_train_idx]
                y_for_cv = self._metrics_y[self._split_train_idx]
            else:
                # Use all data (no external split defined)
                X_for_cv = X_train
                y_for_cv = y_train

            r2cv = self._calculate_r2cv(X_for_cv, y_for_cv, feature_mask)
            return r2cv

        except Exception:
            return -np.inf

    def _create_individual(self, n_features):
        """Create a random individual (chromosome)"""
        if self.n_variables is not None:
            # Create individual with exact number of variables
            individual = np.zeros(n_features, dtype=bool)
            selected = np.random.choice(n_features, self.n_variables, replace=False)
            individual[selected] = True
        else:
            # Random number of variables
            individual = np.random.rand(n_features) > 0.5
            # Ensure at least one feature is selected
            if not individual.any():
                individual[np.random.randint(n_features)] = True
        return individual

    def _initialize_population(self, n_features):
        """Initialize population with mix of random and targeted individuals"""
        population = []

        # Number of random individuals
        n_random = int(self.population_size * self.random_models_ratio)

        # Create targeted individuals
        for _ in range(self.population_size - n_random):
            population.append(self._create_individual(n_features))

        # Create completely random individuals
        for _ in range(n_random):
            if self.n_variables is not None:
                # Even random individuals must have correct n_variables
                individual = self._create_individual(n_features)
            else:
                individual = np.random.rand(n_features) > 0.7
                if not individual.any():
                    individual[np.random.randint(n_features)] = True
            population.append(individual)

        return population

    def _tournament_selection(self, population, fitness_scores, tournament_size=3):
        """Select individual using tournament selection"""
        tournament_indices = np.random.choice(len(population), tournament_size, replace=False)
        tournament_fitness = [fitness_scores[i] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return population[winner_idx].copy()

    def _crossover(self, parent1, parent2):
        """Perform uniform crossover"""
        mask = np.random.rand(len(parent1)) > 0.5
        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)

        # Ensure at least one feature is selected
        if not child1.any():
            child1[np.random.randint(len(child1))] = True
        if not child2.any():
            child2[np.random.randint(len(child2))] = True

        # Ensure target number of variables if specified
        if self.n_variables is not None:
            child1 = self._adjust_to_n_variables(child1)
            child2 = self._adjust_to_n_variables(child2)

        return child1, child2

    def _adjust_to_n_variables(self, individual):
        """Adjust individual to have exactly n_variables features selected"""
        n_selected = np.sum(individual)
        if n_selected != self.n_variables:
            if n_selected > self.n_variables:
                # Remove random features
                selected = np.where(individual)[0]
                to_remove = np.random.choice(
                    selected,
                    n_selected - self.n_variables,
                    replace=False
                )
                individual[to_remove] = False
            else:
                # Add random features
                not_selected = np.where(~individual)[0]
                to_add = np.random.choice(
                    not_selected,
                    self.n_variables - n_selected,
                    replace=False
                )
                individual[to_add] = True
        return individual

    def _mutate(self, individual):
        """Mutate individual by flipping bits"""
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_rate:
                individual[i] = not individual[i]

        # Ensure at least one feature is selected
        if not individual.any():
            individual[np.random.randint(len(individual))] = True

        # Ensure target number of variables if specified
        if self.n_variables is not None:
            n_selected = np.sum(individual)
            if n_selected != self.n_variables:
                if n_selected > self.n_variables:
                    # Remove random features
                    selected = np.where(individual)[0]
                    to_remove = np.random.choice(
                        selected,
                        n_selected - self.n_variables,
                        replace=False
                    )
                    individual[to_remove] = False
                else:
                    # Add random features
                    not_selected = np.where(~individual)[0]
                    to_add = np.random.choice(
                        not_selected,
                        self.n_variables - n_selected,
                        replace=False
                    )
                    individual[to_add] = True

        return individual

    def fit(self, X, y, X_val=None, y_val=None):
        """
        Run genetic algorithm to find optimal feature subset

        Parameters:
        -----------
        X : array-like or DataFrame
            Training features
        y : array-like or Series
            Training target
        X_val : array-like or DataFrame, optional
            Validation features
        y_val : array-like or Series, optional
            Validation target

        Returns:
        --------
        self
        """
        self.no_models_reason_ = None
        self._validation_X = None
        self._validation_y = None
        # Convert to numpy arrays
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.tolist()
            X = X.values
        else:
            self.feature_names_ = [f"Feature_{i}" for i in range(X.shape[1])]

        if isinstance(y, (pd.Series, pd.DataFrame)):
            y = y.values.ravel()

        if X_val is not None and isinstance(X_val, pd.DataFrame):
            X_val = X_val.values
        if y_val is not None and isinstance(y_val, (pd.Series, pd.DataFrame)):
            y_val = y_val.values.ravel()

        if X_val is not None:
            self._validation_X = np.asarray(X_val)
        if y_val is not None:
            self._validation_y = np.asarray(y_val).ravel()

        if self._metrics_X is None:
            self._metrics_X = X
        else:
            if isinstance(self._metrics_X, pd.DataFrame):
                self._metrics_X = self._metrics_X.values
            else:
                self._metrics_X = np.asarray(self._metrics_X)

        if self._metrics_y is None:
            self._metrics_y = y
        else:
            if isinstance(self._metrics_y, (pd.Series, pd.DataFrame)):
                self._metrics_y = self._metrics_y.values.ravel()
            else:
                self._metrics_y = np.asarray(self._metrics_y).ravel()

        if self._metrics_X.shape[1] != X.shape[1]:
            self._metrics_X = X
        if len(self._metrics_y) != len(self._metrics_X):
            self._metrics_y = y

        self._normalize_split_indices(len(self._metrics_y))

        n_features = X.shape[1]

        # Precompute correlation matrix for faster checks
        if self.correlation_threshold < 1.0 and n_features > 1:
            corr_matrix = np.corrcoef(X, rowvar=False)
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0, posinf=0.0, neginf=0.0)
            self._corr_matrix = np.abs(corr_matrix)
        else:
            self._corr_matrix = None

        # Prepare CV splitters
        self._cv = self._make_cv(len(y), self.cv_folds)
        if self.use_validation and y_val is not None:
            self._cv_validation = self._make_cv(len(y_val), self.cv_folds_validation)
        else:
            self._cv_validation = None

        fitness_cache = {}

        # Dictionary to store all unique models with their scores
        all_models = {}  # key: tuple of feature indices, value: fitness score

        # Run algorithm with retries
        for retry in range(self.max_retries):
            print(f"GA Run {retry + 1}/{self.max_retries}")

            # Initialize population
            population = self._initialize_population(n_features)

            # Evolution loop
            generation_best_scores = []
            best_score_this_retry = -np.inf
            no_improve = 0
            last_metrics = {'r2': 0.0, 'q2': None}

            for generation in range(self.n_iterations):
                # Calculate fitness for all individuals with caching
                fitness_scores = []
                for ind in population:
                    feature_key = tuple(np.where(ind)[0])
                    if feature_key in fitness_cache:
                        score = fitness_cache[feature_key]
                    else:
                        score = self._fitness_function(ind, X, y, X_val, y_val)
                        fitness_cache[feature_key] = score
                    fitness_scores.append(score)

                # Store all unique models with valid fitness
                for ind, score in zip(population, fitness_scores):
                    if score > -np.inf:
                        # Create hashable key from feature indices
                        feature_key = tuple(np.where(ind)[0])

                        # Only store models with correct n_variables (if specified)
                        if self.n_variables is not None and len(feature_key) != self.n_variables:
                            continue

                        if feature_key not in all_models or score > all_models[feature_key]:
                            all_models[feature_key] = score

                # Track best individual
                max_fitness = max(fitness_scores)
                generation_best_scores.append(max_fitness)

                if max_fitness > self.best_score_:
                    self.best_score_ = max_fitness
                    best_idx = fitness_scores.index(max_fitness)
                    self.best_features_ = population[best_idx].copy()

                if max_fitness > best_score_this_retry + self.early_stop_min_delta:
                    best_score_this_retry = max_fitness
                    no_improve = 0
                else:
                    no_improve += 1

                # Calculate basic metrics for best individual in current generation
                best_idx_current = fitness_scores.index(max_fitness)
                best_individual_current = population[best_idx_current]
                compute_metrics = True
                if self.metrics_interval and self.metrics_interval > 0:
                    compute_metrics = (
                        generation == 0
                        or generation == self.n_iterations - 1
                        or generation % self.metrics_interval == 0
                    )
                if compute_metrics:
                    split_metrics = self._calculate_basic_metrics(best_individual_current)
                    last_metrics = split_metrics
                else:
                    split_metrics = last_metrics

                # Report progress via callback
                if self.progress_callback:
                    self.progress_callback({
                        'retry': int(retry + 1),
                        'max_retries': int(self.max_retries),
                        'generation': int(generation + 1),
                        'total_generations': int(self.n_iterations),
                        'best_score': float(max_fitness),  # This is R²cv (fitness)
                        'overall_best_score': float(self.best_score_),
                        'n_features': int(np.sum(self.best_features_)) if self.best_features_ is not None else 0,
                        'unique_models': int(len(all_models)),
                        'r2cv': float(max_fitness),  # R²cv is the fitness metric
                        'internal_cv_type': self.internal_cv_type
                    })
                elif generation % 10 == 0:
                    print(f"  Generation {generation}/{self.n_iterations}, "
                          f"Best Score: {max_fitness:.4f}")

                if self.early_stop_rounds and no_improve >= self.early_stop_rounds:
                    break

                # Create next generation
                new_population = []

                # Elitism - keep best individual
                best_idx = fitness_scores.index(max_fitness)
                new_population.append(population[best_idx].copy())

                # Generate rest of population
                while len(new_population) < self.population_size:
                    # Selection
                    parent1 = self._tournament_selection(population, fitness_scores)
                    parent2 = self._tournament_selection(population, fitness_scores)

                    # Crossover
                    child1, child2 = self._crossover(parent1, parent2)

                    # Mutation
                    child1 = self._mutate(child1)
                    child2 = self._mutate(child2)

                    new_population.extend([child1, child2])

                # Trim to population size
                population = new_population[:self.population_size]

            self.fitness_history_.extend(generation_best_scores)

            print(f"  Final Best Score: {self.best_score_:.4f}")

        # Select top N models - OPTIMIZED: first sort by fitness, then calculate metrics only for top candidates
        model_items = list(all_models.items())

        # Step 1: Sort by fitness score (R²cv) - already calculated during GA, no extra computation
        model_items_sorted_by_fitness = sorted(model_items, key=lambda x: x[1], reverse=True)

        # Step 2: Take only top candidates for detailed evaluation (10x n_best_models to have margin)
        n_candidates_to_evaluate = min(len(model_items), self.n_best_models * 10)
        top_candidates = model_items_sorted_by_fitness[:n_candidates_to_evaluate]

        print(f"\nOptimized: Evaluating {n_candidates_to_evaluate} top candidates out of {len(model_items)} unique models...")

        # Step 3: Calculate basic metrics only for top candidates
        split_metrics_cache = {}
        for feature_indices, _ in top_candidates:
            feature_mask = np.zeros(X.shape[1], dtype=bool)
            feature_mask[list(feature_indices)] = True
            split_metrics_cache[feature_indices] = self._calculate_basic_metrics(feature_mask)

        def split_sort_key(item):
            feature_indices, score = item
            metrics = split_metrics_cache.get(feature_indices, {})
            if not metrics or not metrics.get('valid'):
                return (-np.inf, -np.inf, -np.inf)
            r2 = metrics.get('r2', 0.0)
            q2 = metrics.get('q2')
            q2_rank = q2 if q2 is not None else r2
            return (q2_rank, r2, score)

        # Sort only the top candidates by split metrics
        sorted_candidates = sorted(top_candidates, key=split_sort_key, reverse=True)
        self.best_models_ = []

        print(f"\nCalculating detailed metrics for top models by split metrics...")
        for feature_indices, score in sorted_candidates:
            if len(self.best_models_) >= self.n_best_models:
                break
            # Create feature mask for this model
            feature_mask = np.zeros(X.shape[1], dtype=bool)
            feature_mask[list(feature_indices)] = True

            # Split metrics used for selection
            split_metrics = split_metrics_cache.get(feature_indices, {'r2': 0.0, 'q2': None})

            r2_split = split_metrics.get('r2')
            q2_split = split_metrics.get('q2')
            r2_below = r2_split is None or r2_split < self.min_split_r2
            q2_below = q2_split is None or q2_split < self.min_split_q2
            if r2_below and q2_below:
                if not self.best_models_:
                    self.no_models_reason_ = (
                        f"No models met split R2/Q2 >= {self.min_split_r2:.2f}. "
                        "Lower AD threshold or relax the split cutoff."
                    )
                break

            # Calculate detailed metrics for this model
            # Use training data from split (if available) for coefficient calculation
            if self._split_train_idx is not None and len(self._split_train_idx) > 0:
                X_for_metrics = self._metrics_X[self._split_train_idx]
                y_for_metrics = self._metrics_y[self._split_train_idx]
            else:
                X_for_metrics = X
                y_for_metrics = y
            detailed_metrics = self._calculate_detailed_metrics(feature_mask, X_for_metrics, y_for_metrics)
            if self.check_ad and detailed_metrics.get('ad_coverage', 0.0) < self.ad_threshold:
                continue

            # Calculate R²cv_ext on validation set (if enabled)
            r2cv_ext = None
            if self.use_validation and self._validation_X is not None and self._validation_y is not None:
                r2cv_ext = self._calculate_r2cv_ext(feature_mask)

            feature_names = [self.feature_names_[idx] for idx in feature_indices]

            # Build equation string
            intercept = detailed_metrics['intercept']
            coefficients = detailed_metrics['coefficients']
            equation_parts = [f"{intercept:.6f}"]
            for fname, coef in zip(feature_names, coefficients):
                if coef >= 0:
                    equation_parts.append(f"+ {coef:.6f}*{fname}")
                else:
                    equation_parts.append(f"- {abs(coef):.6f}*{fname}")
            equation = " ".join(equation_parts)

            # Calculate R²cv (fitness score) for this model
            if self._split_train_idx is not None and len(self._split_train_idx) > 0:
                X_for_cv = self._metrics_X[self._split_train_idx]
                y_for_cv = self._metrics_y[self._split_train_idx]
            else:
                X_for_cv = X
                y_for_cv = y
            r2cv = self._calculate_r2cv(X_for_cv, y_for_cv, feature_mask)

            self.best_models_.append({
                'r2cv': r2cv,  # CV R² (k-fold or sorted) - MAIN FITNESS METRIC
                'r2': detailed_metrics.get('r2'),  # R² on full TRAIN
                'r2loo': detailed_metrics.get('r2loo'),  # LOO R² on TRAIN
                'rmse_tr': detailed_metrics.get('rmse_tr'),  # RMSE on TRAIN
                'r2cv_ext': r2cv_ext,  # CV R² on validation/test set (if enabled)
                'ad_coverage': detailed_metrics.get('ad_coverage'),
                'intercept': intercept,
                'coefficients': coefficients,
                'equation': equation,
                'n_features': len(feature_indices),
                'feature_indices': list(feature_indices),
                'feature_names': feature_names
            })

        # Sort final models by R²cv (main fitness metric) and assign ranks
        self.best_models_.sort(key=lambda m: m['r2cv'] if m['r2cv'] is not None else -np.inf, reverse=True)
        for idx, model in enumerate(self.best_models_):
            model['rank'] = idx + 1

        if self.best_models_:
            self.best_score_ = float(self.best_models_[0]['r2cv']) if self.best_models_[0]['r2cv'] is not None else 0.0
        elif self.no_models_reason_ is None:
            if self.check_ad:
                self.no_models_reason_ = (
                    "No models met the AD threshold. Lower AD threshold or relax constraints."
                )
            else:
                self.no_models_reason_ = "No valid models found for the current constraints."

        print(f"\nTop {len(self.best_models_)} models identified:")
        for model in self.best_models_:
            r2cv_value = model.get('r2cv')
            r2cv_text = f"{r2cv_value:.4f}" if r2cv_value is not None else "N/A"
            r2_value = model.get('r2')
            r2_text = f"{r2_value:.4f}" if r2_value is not None else "N/A"
            r2loo_value = model.get('r2loo')
            r2loo_text = f"{r2loo_value:.4f}" if r2loo_value is not None else "N/A"
            rmse_tr_value = model.get('rmse_tr')
            rmse_tr_text = f"{rmse_tr_value:.4f}" if rmse_tr_value is not None else "N/A"
            r2cv_ext_value = model.get('r2cv_ext')
            r2cv_ext_text = f"{r2cv_ext_value:.4f}" if r2cv_ext_value is not None else "N/A"
            print(f"  Rank {model['rank']}: R²cv={r2cv_text}, "
                  f"R²={r2_text}, R²loo={r2loo_text}, RMSE_tr={rmse_tr_text}, "
                  f"R²cv_ext={r2cv_ext_text}, AD={model['ad_coverage']:.1f}%, "
                  f"N_features={model['n_features']}, Features={model['feature_names']}")

        return self

    def get_selected_features(self):
        """Get list of selected feature names for best model"""
        if self.best_features_ is None:
            raise ValueError("Model not fitted yet!")

        selected_indices = np.where(self.best_features_)[0]
        return [self.feature_names_[i] for i in selected_indices]

    def get_best_models(self):
        """Get list of top N models with their details"""
        if not self.best_models_:
            if self.no_models_reason_:
                raise ValueError(self.no_models_reason_)
            raise ValueError("Model not fitted yet!")

        return self.best_models_

    def plot_fitness_history(self, temp_path='temp/'):
        """Plot fitness evolution over generations"""
        os.makedirs(temp_path, exist_ok=True)

        plt.figure(figsize=(10, 6))
        plt.plot(self.fitness_history_, linewidth=2, color='#440154')
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Best Fitness (R²)', fontsize=12)
        plt.title('Genetic Algorithm Fitness Evolution', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        add_watermark_matplotlib_after_plot(plt.gcf())
        plot_path = os.path.join(temp_path, 'ga_fitness_history.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()

        return plot_path


def detect_and_remove_y_outliers(df, target_var, method='iqr', threshold=3.0):
    """
    Detect and remove outliers from target variable Y

    Tests for normality and removes outliers using specified method.

    Parameters:
    -----------
    df : DataFrame
        Input dataframe with all data
    target_var : str
        Name of target variable
    method : str
        'iqr' for Interquartile Range or 'zscore' for Z-score method
    threshold : float
        Multiplier for IQR (default 1.5) or Z-score threshold (default 3.0)

    Returns:
    --------
    dict with keys:
        - keep_indices: indices to keep
        - removed_samples: list of dicts with info about removed samples
        - normality_before: dict with normality test results before removal
        - normality_after: dict with normality test results after removal
    """
    from scipy.stats import shapiro

    y = df[target_var].copy()

    # Test normality before outlier removal
    if len(y) >= 3:
        stat_before, p_value_before = shapiro(y)
        normality_before = {
            'test': 'Shapiro-Wilk',
            'statistic': float(stat_before),
            'p_value': float(p_value_before),
            'is_normal': p_value_before > 0.05
        }
    else:
        normality_before = {'test': 'Shapiro-Wilk', 'statistic': None, 'p_value': None, 'is_normal': None}

    # Detect outliers
    if method == 'iqr':
        Q1 = y.quantile(0.25)
        Q3 = y.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outlier_mask = (y < lower_bound) | (y > upper_bound)
    elif method == 'zscore':
        z_scores = np.abs(stats.zscore(y))
        outlier_mask = z_scores > threshold
    else:
        raise ValueError(f"Unknown outlier detection method: {method}")

    # Get outlier indices
    outlier_indices = y[outlier_mask].index.tolist()
    keep_indices = y[~outlier_mask].index.tolist()

    # Create removed samples info with full row data
    removed_samples = []
    for idx in outlier_indices:
        sample_data = df.loc[idx].to_dict()
        sample_data['Sample_Index'] = int(idx)
        sample_data['Y_Value'] = float(y.loc[idx])
        removed_samples.append(sample_data)

    # Test normality after outlier removal
    y_cleaned = y[~outlier_mask]
    if len(y_cleaned) >= 3:
        stat_after, p_value_after = shapiro(y_cleaned)
        normality_after = {
            'test': 'Shapiro-Wilk',
            'statistic': float(stat_after),
            'p_value': float(p_value_after),
            'is_normal': p_value_after > 0.05
        }
    else:
        normality_after = {'test': 'Shapiro-Wilk', 'statistic': None, 'p_value': None, 'is_normal': None}

    return {
        'keep_indices': keep_indices,
        'removed_samples': removed_samples,
        'normality_before': normality_before,
        'normality_after': normality_after,
        'method': method,
        'threshold': threshold
    }


def preprocess_for_ga(df, target_var, y_transformation='none',
                     autoscale=True, remove_zero_variance=True,
                     remove_low_variance=False, variance_threshold=0.01):
    """
    Preprocess data before GA variable selection

    Parameters:
    -----------
    df : DataFrame
        Input dataframe
    target_var : str
        Target variable name
    y_transformation : str
        Transformation for Y: 'none', 'log', 'sqrt', 'square', 'inverse'
    autoscale : bool
        Whether to apply autoscaling (standardization)
    remove_zero_variance : bool
        Whether to remove zero variance features
    remove_low_variance : bool
        Whether to remove low variance features
    variance_threshold : float
        Threshold for low variance removal

    Returns:
    --------
    X : DataFrame
        Preprocessed features
    y : Series
        Preprocessed target
    removed_features : list
        List of removed feature names
    preprocessing_info : dict
        Information about preprocessing steps
    """
    preprocessing_info = {
        'y_transformation': y_transformation,
        'autoscale': autoscale,
        'removed_features': []
    }

    # Separate target and features
    y = df[target_var].copy()
    X = df.drop(columns=[target_var]).copy()

    # Apply Y transformation
    original_y = y.copy()
    if y_transformation == 'log':
        if (y <= 0).any():
            raise ValueError("Log transformation requires all positive values")
        y = np.log(y)
    elif y_transformation == 'sqrt':
        if (y < 0).any():
            raise ValueError("Square root transformation requires non-negative values")
        y = np.sqrt(y)
    elif y_transformation == 'square':
        y = y ** 2
    elif y_transformation == 'inverse':
        if (y == 0).any():
            raise ValueError("Inverse transformation requires non-zero values")
        y = 1 / y

    preprocessing_info['y_transformed'] = y
    preprocessing_info['y_original'] = original_y

    # Remove zero variance features
    removed_features = []
    if remove_zero_variance:
        zero_var_features = X.columns[X.var() == 0].tolist()
        X = X.drop(columns=zero_var_features)
        removed_features.extend(zero_var_features)
        preprocessing_info['zero_variance_removed'] = zero_var_features

    # Remove low variance features
    if remove_low_variance:
        selector = VarianceThreshold(threshold=variance_threshold)
        selector.fit(X)
        low_var_mask = selector.get_support()
        low_var_features = X.columns[~low_var_mask].tolist()
        X = X.loc[:, low_var_mask]
        removed_features.extend(low_var_features)
        preprocessing_info['low_variance_removed'] = low_var_features
        preprocessing_info['variance_threshold'] = variance_threshold

    # Autoscaling
    if autoscale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        preprocessing_info['scaler'] = scaler

    preprocessing_info['removed_features'] = removed_features
    preprocessing_info['final_features'] = X.columns.tolist()

    return X, y, removed_features, preprocessing_info


def plot_y_histogram(y, y_name='Target Variable', temp_path='temp/'):
    """
    Plot histogram of target variable to help choose transformation

    Parameters:
    -----------
    y : Series or array
        Target variable data
    y_name : str
        Name for plot title
    temp_path : str
        Path to save plot

    Returns:
    --------
    tuple
        (plot_path, statistics_dict)
        plot_path: str - Path to saved plot
        statistics_dict: dict - Dictionary with mean, median, std, skewness, kurtosis, normality test
    """
    from scipy.stats import shapiro, skew, kurtosis

    os.makedirs(temp_path, exist_ok=True)

    # Calculate statistics
    y_array = np.array(y)
    mean_val = float(np.mean(y_array))
    median_val = float(np.median(y_array))
    std_val = float(np.std(y_array))
    skewness_val = float(skew(y_array))
    kurtosis_val = float(kurtosis(y_array))

    # Normality test (Shapiro-Wilk)
    if len(y_array) >= 3:
        stat, p_value = shapiro(y_array)
        normality_pvalue = float(p_value)
        is_normal = p_value > 0.05
    else:
        normality_pvalue = 0.0
        is_normal = False

    statistics = {
        'mean': mean_val,
        'median': median_val,
        'std': std_val,
        'skewness': skewness_val,
        'kurtosis': kurtosis_val,
        'normality_pvalue': normality_pvalue,
        'is_normal': is_normal
    }

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(y, bins=30, color='#440154', alpha=0.7, edgecolor='black')
    ax1.set_xlabel(y_name, fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title(f'Histogram of {y_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(y, dist="norm", plot=ax2)
    ax2.set_title('Q-Q Plot (Normality Check)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate unique filename to avoid overwriting
    import time
    timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, f'y_histogram_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path, statistics


def plot_y_histogram_split(y_train, y_test, y_name='Target Variable', temp_path='temp/'):
    """
    Plot histograms for train and test sets separately

    Parameters:
    -----------
    y_train : Series or array
        Training target variable data
    y_test : Series or array
        Test target variable data
    y_name : str
        Name for plot title
    temp_path : str
        Path to save plot

    Returns:
    --------
    str
        Path to saved plot
    """
    os.makedirs(temp_path, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Train histogram
    axes[0, 0].hist(y_train, bins=30, color='#440154', alpha=0.7, edgecolor='black')
    axes[0, 0].set_xlabel(y_name, fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'Training Set - Histogram (n={len(y_train)})', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # Train Q-Q plot
    stats.probplot(y_train, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Training Set - Q-Q Plot', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)

    # Test histogram
    axes[1, 0].hist(y_test, bins=30, color='#35b779', alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel(y_name, fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title(f'Test Set - Histogram (n={len(y_test)})', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Test Q-Q plot
    stats.probplot(y_test, dist="norm", plot=axes[1, 1])
    axes[1, 1].set_title('Test Set - Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Generate unique filename to avoid overwriting
    import time
    timestamp = int(time.time() * 1000000)  # microseconds for uniqueness
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, f'y_histogram_split_{timestamp}.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path


def plot_y_histogram_old(y, y_name='Target Variable', temp_path='temp/'):
    """
    Plot histogram of target variable to help choose transformation

    Parameters:
    -----------
    y : array-like
        Target variable
    y_name : str
        Name of target variable
    temp_path : str
        Path to save plot

    Returns:
    --------
    plot_path : str
        Path to saved plot
    stats_dict : dict
        Dictionary with distribution statistics
    """
    os.makedirs(temp_path, exist_ok=True)

    # Calculate statistics
    stats_dict = {
        'mean': np.mean(y),
        'median': np.median(y),
        'std': np.std(y),
        'skewness': stats.skew(y),
        'kurtosis': stats.kurtosis(y),
        'min': np.min(y),
        'max': np.max(y)
    }

    # Test for normality
    _, p_value = stats.normaltest(y)
    stats_dict['normality_pvalue'] = p_value
    stats_dict['is_normal'] = p_value > 0.05

    # Create plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Histogram
    axes[0].hist(y, bins=30, color='#440154', alpha=0.7, edgecolor='black')
    axes[0].axvline(stats_dict['mean'], color='red', linestyle='--',
                   linewidth=2, label=f"Mean: {stats_dict['mean']:.2f}")
    axes[0].axvline(stats_dict['median'], color='green', linestyle='--',
                   linewidth=2, label=f"Median: {stats_dict['median']:.2f}")
    axes[0].set_xlabel(y_name, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title('Distribution of Target Variable', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Q-Q plot
    stats.probplot(y, dist="norm", plot=axes[1])
    axes[1].set_title('Q-Q Plot', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Use timestamp to ensure unique filename for each plot
    import time
    timestamp = str(int(time.time() * 1000))
    filename = f'y_distribution_{timestamp}.png'
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, filename)
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()

    return plot_path, stats_dict
