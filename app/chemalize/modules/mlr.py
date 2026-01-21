import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split, KFold, LeaveOneOut, StratifiedKFold
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy import stats
from scipy.stats import f as f_dist
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.stattools import durbin_watson
from sklearn.linear_model import LinearRegression
import os
import io
from app.utils.watermark import add_watermark_matplotlib_after_plot

def hat_matrix(X):
    """Calculate hat matrix for X"""
    X = np.asarray(X)  # Ensure X is a numpy array
    xtx = X.T @ X
    # Add small random values to diagonal for numerical stability
    np.fill_diagonal(xtx, xtx.diagonal() + np.random.uniform(0.001, 0.002, xtx.shape[0]))
    ixtx = np.linalg.inv(xtx)
    return X @ (ixtx @ X.T)

def williams_plot(result_df, model):
    """Create Williams plot data"""
    # Split result into training and test sets
    train_data = result_df[result_df['dataset'] == 'train']
    test_data = result_df[result_df['dataset'] == 'test']
    
    # Poprawione indeksowanie - użyj nazw kolumn zamiast indeksów numerycznych
    target_col = result_df.columns[1]  # Druga kolumna zawiera wartości docelowe
    
    # Wybierz tylko kolumny, które są zmiennymi modelu
    feature_cols = model.variables
    
    # Upewnij się, że wszystkie zmienne modelu są dostępne w danych
    available_cols = [col for col in feature_cols if col in train_data.columns]
    if len(available_cols) == 0:
        raise ValueError("No model variables found in data columns")
    
    X_train = train_data[available_cols]
    X_test = test_data[available_cols]
    y_train = train_data[target_col].values
    y_test = test_data[target_col].values
    
    # Calculate hat matrix based on training data only
    H_train = hat_matrix(X_train.values)
    leverage_train = np.diag(H_train)

    # For test data, calculate leverage using training data hat matrix
    # leverage_test = diag(X_test @ (X_train'X_train)^-1 @ X_test')
    X_train_np = X_train.values
    X_test_np = X_test.values
    XtX = X_train_np.T @ X_train_np
    # Add regularization for numerical stability
    reg_factor = 1e-8 * np.trace(XtX) / XtX.shape[0] if XtX.shape[0] > 0 else 1e-8
    XtX_reg = XtX + reg_factor * np.eye(XtX.shape[0])
    try:
        XtX_inv = np.linalg.inv(XtX_reg)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX_reg)
    leverage_test = np.sum((X_test_np @ XtX_inv) * X_test_np, axis=1)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    # Calculate residuals
    residual_train = y_train - y_pred_train
    residual_test = y_test - y_pred_test

    # Calculate internally studentized residuals: r_i = e_i / (s * sqrt(1 - h_i))
    # s = sqrt(MSE) based on training data
    n_train = len(y_train)
    p = len(model.variables)
    SSE_train = np.sum(residual_train ** 2)
    df = n_train - p - 1  # degrees of freedom (n - p - 1 for intercept)
    if df <= 0:
        df = 1
    MSE = SSE_train / df
    s = np.sqrt(MSE)

    # Studentized residuals for training data
    leverage_factor_train = np.sqrt(np.maximum(1 - leverage_train, 1e-10))
    s_residual_train = residual_train / (s * leverage_factor_train)

    # Studentized residuals for test data (using training MSE)
    leverage_factor_test = np.sqrt(np.maximum(1 - leverage_test, 1e-10))
    s_residual_test = residual_test / (s * leverage_factor_test)
    
    p = len(model.variables)
    n = len(X_train)
    h_star = (3 * (p + 1)) / n
    
    AD_train = 100 * np.sum((leverage_train < h_star) & (np.abs(s_residual_train) < 3)) / len(leverage_train)
    AD_test = 100 * np.sum((leverage_test < h_star) & (np.abs(s_residual_test) < 3)) / len(leverage_test)
    
    lev = np.concatenate([leverage_train, leverage_test])
    res = np.concatenate([s_residual_train, s_residual_test])
    group = np.concatenate([['Train'] * len(X_train), ['Test'] * len(X_test)])

    data_to_plot = pd.DataFrame({'lev': lev, 'res': res, 'group': group})

    # Identify outliers (points outside AD)
    # A point is outside AD if: leverage >= h_star OR |standardized residual| >= 3
    outliers_train_idx = np.where((leverage_train >= h_star) | (np.abs(s_residual_train) >= 3))[0]
    outliers_test_idx = np.where((leverage_test >= h_star) | (np.abs(s_residual_test) >= 3))[0]

    # Get original indices from result_df
    train_indices = train_data.index.tolist()
    test_indices = test_data.index.tolist()

    # Create outlier information
    outliers = []
    for idx in outliers_train_idx:
        outliers.append({
            'dataset': 'Train',
            'index': train_indices[idx],
            'leverage': float(leverage_train[idx]),
            'std_residual': float(s_residual_train[idx])
        })

    for idx in outliers_test_idx:
        outliers.append({
            'dataset': 'Test',
            'index': test_indices[idx],
            'leverage': float(leverage_test[idx]),
            'std_residual': float(s_residual_test[idx])
        })

    return {
        'ADVal': [AD_train, AD_test],
        'DTP': data_to_plot,
        'h_star': h_star,
        'outliers': outliers
    }

def plot_wp(result):
    """Plot Williams plot"""
    plt.figure(figsize=(10, 8))
    
    # Plot points
    sns.scatterplot(data=result['DTP'], x='lev', y='res', hue='group', 
                   palette={'Train': '#440154', 'Test': '#35b779'},
                   alpha=0.7, s=70)
    
    # Add threshold lines
    plt.axvline(x=result['h_star'], linestyle='dashed', color='black')
    plt.axhline(y=3, linestyle='dashed', color='gray')
    plt.axhline(y=-3, linestyle='dashed', color='gray')
    
    # Set labels and style
    plt.xlabel('Leverage', fontsize=14)
    plt.ylabel('Standardized Residual', fontsize=14)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.legend(title='', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Set axis limits
    y_min = min(-4.5, result['DTP']['res'].min())
    y_max = max(3.5, result['DTP']['res'].max())
    x_max = max(result['DTP']['lev'].max(), result['h_star']) + 0.1
    plt.ylim(y_min, y_max)
    plt.xlim(0, x_max)
    
    return plt.gcf()

class MLRModel:
    def __init__(self, variables, include_intercept=True):
        self.variables = variables
        self.include_intercept = include_intercept


def stratified_split_with_endpoints(X, y, ratio_n=3):
    """
    Stratified sorted split with min/max Y values always in test set.

    Data is sorted by Y values (descending). Min and max Y observations are
    forced into test set. Remaining samples are systematically assigned
    using 1:n ratio (every nth sample goes to test).

    Parameters:
    -----------
    X : DataFrame - features
    y : Series - target
    ratio_n : int - ratio denominator for 1:n split (default 3 for 1:3 = 25% test)
              1:2 = 50% test, 1:3 = 25% test, 1:4 = 20% test, etc.

    Returns:
    --------
    train_idx, test_idx : lists of original DataFrame indices
    """
    n_samples = len(y)

    # Sort by Y values (ascending - from smallest to largest) and get sorted indices
    sorted_indices = y.sort_values(ascending=True).index.tolist()

    # Find indices of min and max Y values (in original DataFrame)
    min_y_idx = y.idxmin()
    max_y_idx = y.idxmax()
    forced_test = [min_y_idx, max_y_idx]  # Min first (smallest), max last (largest)

    # Remove forced test indices from sorted list
    remaining_sorted = [idx for idx in sorted_indices if idx not in forced_test]

    # Systematic sampling: every nth sample goes to test (1:n ratio)
    # For 1:3 ratio, take every 3rd sample starting from position ratio_n-1
    additional_test = []
    train_idx = []

    for i, idx in enumerate(remaining_sorted):
        # Position in 1:n cycle (0, 1, 2, ... n-1, 0, 1, 2, ...)
        position_in_cycle = i % ratio_n

        # Last position in cycle goes to test (e.g., for 1:3, position 2 goes to test)
        if position_in_cycle == ratio_n - 1:
            additional_test.append(idx)
        else:
            train_idx.append(idx)

    test_idx = forced_test + additional_test

    return train_idx, test_idx


def calculate_ccc(y_true, y_pred):
    """Calculate Concordance Correlation Coefficient (CCC)"""
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()

    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)

    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))

    ccc = (2 * covariance) / (var_true + var_pred + (mean_true - mean_pred)**2)
    return ccc


def calculate_cv_metrics(X, y, n_folds=5, include_intercept=True):
    """
    Calculate cross-validation metrics: R²cv and Q²cv (5-fold)

    Returns dict with r2cv_train, r2cv_test (Q²cv)
    """
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    r2_train_scores = []
    r2_test_scores = []
    rmse_train_scores = []
    rmse_test_scores = []

    for train_idx, test_idx in kf.split(X):
        X_train_cv, X_test_cv = X.iloc[train_idx], X.iloc[test_idx]
        y_train_cv, y_test_cv = y.iloc[train_idx], y.iloc[test_idx]

        if include_intercept:
            X_train_sm = sm.add_constant(X_train_cv)
            X_test_sm = sm.add_constant(X_test_cv)
        else:
            X_train_sm = X_train_cv
            X_test_sm = X_test_cv

        model_cv = sm.OLS(y_train_cv, X_train_sm).fit()

        y_pred_train = model_cv.predict(X_train_sm)
        y_pred_test = model_cv.predict(X_test_sm)

        r2_train_scores.append(r2_score(y_train_cv, y_pred_train))
        r2_test_scores.append(r2_score(y_test_cv, y_pred_test))
        rmse_train_scores.append(np.sqrt(mean_squared_error(y_train_cv, y_pred_train)))
        rmse_test_scores.append(np.sqrt(mean_squared_error(y_test_cv, y_pred_test)))

    return {
        'r2cv_train': np.mean(r2_train_scores),
        'q2cv': np.mean(r2_test_scores),  # Q²cv is R² on test folds
        'rmse_cv_train': np.mean(rmse_train_scores),
        'rmse_cv_test': np.mean(rmse_test_scores)
    }


def calculate_loo_metrics(X, y, include_intercept=True):
    """
    Calculate Leave-One-Out metrics: Q²loo and RMSEloo
    """
    loo = LeaveOneOut()
    y_true_all = []
    y_pred_all = []

    for train_idx, test_idx in loo.split(X):
        X_train_loo, X_test_loo = X.iloc[train_idx], X.iloc[test_idx]
        y_train_loo, y_test_loo = y.iloc[train_idx], y.iloc[test_idx]

        if include_intercept:
            X_train_sm = sm.add_constant(X_train_loo)
            X_test_sm = sm.add_constant(X_test_loo)
        else:
            X_train_sm = X_train_loo
            X_test_sm = X_test_loo

        model_loo = sm.OLS(y_train_loo, X_train_sm).fit()
        y_pred = model_loo.predict(X_test_sm)

        y_true_all.append(y_test_loo.values[0])
        y_pred_all.append(y_pred[0])

    y_true_all = np.array(y_true_all)
    y_pred_all = np.array(y_pred_all)

    # Q²loo = 1 - SS_res / SS_tot
    ss_res = np.sum((y_true_all - y_pred_all)**2)
    ss_tot = np.sum((y_true_all - np.mean(y_true_all))**2)
    q2_loo = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    rmse_loo = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

    return {
        'q2_loo': q2_loo,
        'rmse_loo': rmse_loo
    }


def perform_mlr(df, target_var, selected_features, include_intercept=True,
                split_method='stratified_endpoints', split_params=None, scale_data=False,
                check_assumptions=True, detect_outliers=False, temp_path='temp/',
                predefined_train_idx=None, predefined_test_idx=None,
                ga_coefficients=None, ga_intercept=None):
    """
    Perform Multiple Linear Regression analysis with various splitting methods

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    target_var : str
        Name of the target variable
    selected_features : list
        List of feature names to use in the model
    include_intercept : bool, default=True
        Whether to include an intercept in the model
    split_method : str, default='stratified_endpoints'
        Method to split the data ('stratified_endpoints', 'random', 'stratified', 'time', 'kfold', 'loocv', 'predefined')
        Default 'stratified_endpoints' uses 1:3 split with first/last values in test set
    split_params : dict, default=None
        Parameters for the splitting method
    scale_data : bool, default=False
        Whether to standardize the features
    check_assumptions : bool, default=True
        Whether to check regression assumptions
    detect_outliers : bool, default=False
        Whether to detect outliers and influential points
    temp_path : str, default='temp/'
        Path to store temporary files
    predefined_train_idx : list, default=None
        Predefined training indices (used when split_method='predefined')
    predefined_test_idx : list, default=None
        Predefined test indices (used when split_method='predefined')
    ga_coefficients : list, default=None
        Pre-calculated coefficients from GA (skips model fitting if provided)
    ga_intercept : float, default=None
        Pre-calculated intercept from GA (skips model fitting if provided)

    Returns:
    --------
    dict
        Dictionary containing all the calculated metrics and plots
    """
    # Ensure temp directory exists
    os.makedirs(temp_path, exist_ok=True)
    
    # Set default split parameters if not provided
    if split_params is None:
        split_params = {}
    
    # Prepare data
    X = df[selected_features].copy()
    y = df[target_var].copy()
    
    # Create model
    model = MLRModel(selected_features, include_intercept)
    
    # Scale data if requested
    if scale_data:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
    
    # Split data based on the selected method
    result_df = None

    # Handle stratified split with endpoints (min/max Y in test) - DEFAULT method
    if split_method == 'stratified_endpoints':
        # Stratified sorted split with 1:n ratio (default 1:3)
        ratio_n = split_params.get('ratio_n', 3)  # 1:3 ratio by default

        train_idx, test_idx = stratified_split_with_endpoints(X, y, ratio_n=ratio_n)

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fit model using statsmodels for detailed statistics
        if include_intercept:
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
        else:
            X_train_sm = X_train
            X_test_sm = X_test

        sm_model = sm.OLS(y_train, X_train_sm).fit()

        # Get predictions
        y_pred_train = sm_model.predict(X_train_sm)
        y_pred_test = sm_model.predict(X_test_sm)

        # Create result DataFrame for all calculations
        result_df = pd.DataFrame({
            'dataset': ['train'] * len(y_train) + ['test'] * len(y_test),
            target_var: pd.concat([y_train, y_test]).reset_index(drop=True),
            'predictions': np.concatenate([y_pred_train, y_pred_test])
        })

        # Add feature columns
        for feature in selected_features:
            result_df[feature] = pd.concat([X_train[feature], X_test[feature]]).reset_index(drop=True)

    # Handle predefined indices (from GA or other sources)
    elif split_method == 'predefined' or (predefined_train_idx is not None and predefined_test_idx is not None):
        # Use predefined train/test indices
        train_idx = predefined_train_idx
        test_idx = predefined_test_idx

        X_train = X.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

        # Fit model using statsmodels for detailed statistics
        if include_intercept:
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
        else:
            X_train_sm = X_train
            X_test_sm = X_test

        sm_model = sm.OLS(y_train, X_train_sm).fit()

        # Get predictions - use GA coefficients if provided, otherwise use fitted model
        if ga_coefficients is not None and ga_intercept is not None:
            # Use GA coefficients for predictions
            ga_coef_array = np.array(ga_coefficients)
            y_pred_train = ga_intercept + X_train.values @ ga_coef_array
            if len(X_test) > 0:
                y_pred_test = ga_intercept + X_test.values @ ga_coef_array
            else:
                y_pred_test = np.array([])
        else:
            y_pred_train = sm_model.predict(X_train_sm)
            y_pred_test = sm_model.predict(X_test_sm)

        # Create result DataFrame for all calculations
        result_df = pd.DataFrame({
            'dataset': ['train'] * len(y_train) + ['test'] * len(y_test),
            target_var: pd.concat([y_train, y_test]).reset_index(drop=True),
            'predictions': np.concatenate([y_pred_train, y_pred_test])
        })

        # Add feature columns
        for feature in selected_features:
            result_df[feature] = pd.concat([X_train[feature], X_test[feature]]).reset_index(drop=True)

    elif split_method == 'random':
        # Random split
        test_size = split_params.get('test_size', 0.2)
        shuffle = split_params.get('shuffle', True)
        random_state = split_params.get('random_state', 42)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, shuffle=shuffle, random_state=random_state
        )
        
        # Fit model using statsmodels for detailed statistics
        if include_intercept:
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
        else:
            X_train_sm = X_train
            X_test_sm = X_test
            
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # Get predictions
        y_pred_train = sm_model.predict(X_train_sm)
        y_pred_test = sm_model.predict(X_test_sm)
        
        # Create result DataFrame for all calculations
        result_df = pd.DataFrame({
            'dataset': ['train'] * len(y_train) + ['test'] * len(y_test),
            target_var: pd.concat([y_train, y_test]).reset_index(drop=True),
            'predictions': np.concatenate([y_pred_train, y_pred_test])
        })
        
        # Add feature columns
        for feature in selected_features:
            result_df[feature] = pd.concat([X_train[feature], X_test[feature]]).reset_index(drop=True)
            
    elif split_method == 'stratified':
        # Stratified split based on target variable bins
        test_size = split_params.get('test_size', 0.2)
        n_bins = split_params.get('n_bins', 5)
        
        # Create bins for continuous target variable
        y_binned = pd.qcut(y, n_bins, labels=False, duplicates='drop')
        
        # Use stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y_binned, random_state=42
        )
        
        # Fit model using statsmodels
        if include_intercept:
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
        else:
            X_train_sm = X_train
            X_test_sm = X_test
            
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # Get predictions
        y_pred_train = sm_model.predict(X_train_sm)
        y_pred_test = sm_model.predict(X_test_sm)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dataset': ['train'] * len(y_train) + ['test'] * len(y_test),
            target_var: pd.concat([y_train, y_test]).reset_index(drop=True),
            'predictions': np.concatenate([y_pred_train, y_pred_test])
        })
        
        # Add feature columns
        for feature in selected_features:
            result_df[feature] = pd.concat([X_train[feature], X_test[feature]]).reset_index(drop=True)
            
    elif split_method == 'time':
        # Time-based split
        time_column = split_params.get('time_column', '')
        test_size = split_params.get('test_size', 0.2)
        
        if not time_column or time_column not in df.columns:
            raise ValueError("Time column not found in the dataset")
            
        # Sort by time column
        df_sorted = df.sort_values(by=time_column)
        
        # Split based on time
        train_size = 1 - test_size
        train_idx = int(len(df_sorted) * train_size)
        
        train_df = df_sorted.iloc[:train_idx]
        test_df = df_sorted.iloc[train_idx:]
        
        X_train = train_df[selected_features]
        y_train = train_df[target_var]
        X_test = test_df[selected_features]
        y_test = test_df[target_var]
        
        # Fit model using statsmodels
        if include_intercept:
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
        else:
            X_train_sm = X_train
            X_test_sm = X_test
            
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # Get predictions
        y_pred_train = sm_model.predict(X_train_sm)
        y_pred_test = sm_model.predict(X_test_sm)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'dataset': ['train'] * len(y_train) + ['test'] * len(y_test),
            target_var: pd.concat([y_train, y_test]).reset_index(drop=True),
            'predictions': np.concatenate([y_pred_train, y_pred_test])
        })
        
        # Add feature columns
        for feature in selected_features:
            result_df[feature] = pd.concat([X_train[feature], X_test[feature]]).reset_index(drop=True)
            
    elif split_method == 'kfold':
        # K-fold cross validation
        n_folds = split_params.get('n_folds', 5)
        shuffle = split_params.get('shuffle', True)
        
        # Initialize K-fold
        kf = KFold(n_splits=n_folds, shuffle=shuffle, random_state=42)
        
        # Lists to store cross-validation results
        cv_train_r2 = []
        cv_test_r2 = []
        cv_train_rmse = []
        cv_test_rmse = []
        
        # Prepare DataFrame for all predictions
        all_indices = []
        all_datasets = []
        all_actuals = []
        all_predictions = []
        
        # For storing the feature values for the result DataFrame
        feature_values = {feature: [] for feature in selected_features}
        
        # Perform K-fold CV
        for train_idx, test_idx in kf.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            if include_intercept:
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test)
            else:
                X_train_sm = X_train
                X_test_sm = X_test
                
            fold_model = sm.OLS(y_train, X_train_sm).fit()
            
            # Predictions
            y_pred_train = fold_model.predict(X_train_sm)
            y_pred_test = fold_model.predict(X_test_sm)
            
            # Metrics
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            # Store metrics
            cv_train_r2.append(train_r2)
            cv_test_r2.append(test_r2)
            cv_train_rmse.append(train_rmse)
            cv_test_rmse.append(test_rmse)
            
            # Store predictions for visualization
            all_indices.extend(train_idx)
            all_datasets.extend(['train'] * len(train_idx))
            all_actuals.extend(y_train.values)
            all_predictions.extend(y_pred_train)
            
            all_indices.extend(test_idx)
            all_datasets.extend(['test'] * len(test_idx))
            all_actuals.extend(y_test.values)
            all_predictions.extend(y_pred_test)
            
            # Store feature values
            for feature in selected_features:
                feature_values[feature].extend(X_train[feature].values)
                feature_values[feature].extend(X_test[feature].values)
        
        # Calculate average metrics
        cv_train_r2_mean = np.mean(cv_train_r2)
        cv_test_r2_mean = np.mean(cv_test_r2)
        cv_train_rmse_mean = np.mean(cv_train_rmse)
        cv_test_rmse_mean = np.mean(cv_test_rmse)
        
        # Create DataFrame with all results
        temp_dict = {
            'dataset': all_datasets,
            target_var: all_actuals,
            'predictions': all_predictions
        }
        
        # Add feature columns
        for feature in selected_features:
            temp_dict[feature] = feature_values[feature]
            
        # Create result DataFrame sorted by original indices
        result_df_unsorted = pd.DataFrame(temp_dict)
        
        # Sort by original indices to maintain the original order
        sort_dict = {idx: i for i, idx in enumerate(all_indices)}
        sorted_indices = [sort_dict[i] for i in range(len(sort_dict))]
        result_df = result_df_unsorted.iloc[sorted_indices].reset_index(drop=True)
        
        # Fit final model on all data for coefficients
        if include_intercept:
            X_sm = sm.add_constant(X)
        else:
            X_sm = X
            
        sm_model = sm.OLS(y, X_sm).fit()
        
    elif split_method == 'loocv':
        # Leave-One-Out Cross Validation
        loo = LeaveOneOut()
        
        # Lists to store predictions
        all_indices = []
        all_actuals = []
        all_predictions = []
        
        # For storing the feature values for the result DataFrame
        feature_values = {feature: [] for feature in selected_features}
        
        # Perform LOOCV
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # Fit model
            if include_intercept:
                X_train_sm = sm.add_constant(X_train)
                X_test_sm = sm.add_constant(X_test)
            else:
                X_train_sm = X_train
                X_test_sm = X_test
                
            fold_model = sm.OLS(y_train, X_train_sm).fit()
            
            # Predictions
            y_pred = fold_model.predict(X_test_sm)[0]
            
            # Store predictions
            all_indices.append(test_idx[0])
            all_actuals.append(y_test.values[0])
            all_predictions.append(y_pred)
            
            # Store feature values
            for feature in selected_features:
                feature_values[feature].append(X_test[feature].values[0])
        
        # Create result DataFrame
        temp_dict = {
            'dataset': ['test'] * len(all_indices),
            target_var: all_actuals,
            'predictions': all_predictions
        }
        
        # Add feature columns
        for feature in selected_features:
            temp_dict[feature] = feature_values[feature]
            
        # Create result DataFrame sorted by original indices
        result_df_unsorted = pd.DataFrame(temp_dict)
        
        # Sort by original indices to maintain the original order
        sort_dict = {idx: i for i, idx in enumerate(all_indices)}
        sorted_indices = [sort_dict[i] for i in range(len(sort_dict))]
        result_df = result_df_unsorted.iloc[sorted_indices].reset_index(drop=True)
        
        # Add training data (same as test for LOOCV but labeled as train for metrics calculations)
        train_df = result_df.copy()
        train_df['dataset'] = 'train'
        result_df = pd.concat([train_df, result_df]).reset_index(drop=True)
        
        # Fit final model on all data for coefficients
        if include_intercept:
            X_sm = sm.add_constant(X)
        else:
            X_sm = X
            
        sm_model = sm.OLS(y, X_sm).fit()
    
    elif split_method == 'systematic':
        # Systematic sampling based on sorted Y values
        step = split_params.get('step', 3)
        include_last_point = split_params.get('include_last_point', True)
        
        # Sort data based on target variable values
        sorted_indices = y.argsort().values
        X_sorted = X.iloc[sorted_indices].reset_index(drop=True)
        y_sorted = y.iloc[sorted_indices].reset_index(drop=True)
        
        # Calculate validation indices
        n = len(X_sorted)
        # Start from index 2, take every 'step' index (e.g. 2, 5, 8, 11, ...)
        validation_indices = list(range(2, n-1, step))
        
        # Add last point if requested
        if include_last_point:
            validation_indices.append(n-1)
        
        # Get training indices (all indices not in validation_indices)
        all_indices = list(range(n))
        training_indices = [i for i in all_indices if i not in validation_indices]
        
        # Split the data
        X_train = X_sorted.iloc[training_indices].reset_index(drop=True)
        X_test = X_sorted.iloc[validation_indices].reset_index(drop=True)
        y_train = y_sorted.iloc[training_indices].reset_index(drop=True)
        y_test = y_sorted.iloc[validation_indices].reset_index(drop=True)
        
        # Fit model using statsmodels
        if include_intercept:
            X_train_sm = sm.add_constant(X_train)
            X_test_sm = sm.add_constant(X_test)
        else:
            X_train_sm = X_train
            X_test_sm = X_test
            
        sm_model = sm.OLS(y_train, X_train_sm).fit()
        
        # Get predictions
        y_pred_train = sm_model.predict(X_train_sm)
        y_pred_test = sm_model.predict(X_test_sm)
        
        # Create result DataFrame for all calculations
        result_df = pd.DataFrame({
            'dataset': ['train'] * len(y_train) + ['test'] * len(y_test),
            target_var: pd.concat([y_train, y_test]).reset_index(drop=True),
            'predictions': np.concatenate([y_pred_train, y_pred_test])
        })
        
        # Add feature columns
        for feature in selected_features:
            result_df[feature] = pd.concat([X_train[feature], X_test[feature]]).reset_index(drop=True)
    
    else:
        raise ValueError(f"Unknown split method: {split_method}")
    
    # Calculate metrics for the entire dataset
    y_all = result_df.iloc[:, 1].values  # target_var column
    y_pred_all = result_df['predictions'].values
    
    # Basic metrics
    mse_all = mean_squared_error(y_all, y_pred_all)
    rmse_all = np.sqrt(mse_all)
    r2_all = r2_score(y_all, y_pred_all)
    mae_all = mean_absolute_error(y_all, y_pred_all)
    
    # Split metrics by dataset (train/test)
    train_mask = result_df["dataset"] == 'train'
    test_mask = result_df["dataset"] == 'test'
    
    # Training set metrics
    y_train = y_all[train_mask]
    y_pred_train = y_pred_all[train_mask]
    n_train = len(y_train)
    
    # R2 and Adjusted R2
    R2tr = r2_score(y_train, y_pred_train)
    p = len(selected_features)  # Number of predictors in the model
    AdjR2tr = 1 - (1 - R2tr) * (n_train - 1) / (n_train - p - 1)
    
    # Error metrics
    RMSEtr = np.sqrt(mean_squared_error(y_train, y_pred_train))
    MAEtr = mean_absolute_error(y_train, y_pred_train)
    
    # F-statistic and significance
    TSS_train = np.sum((y_train - np.mean(y_train))**2)
    RSS_train = np.sum((y_train - y_pred_train)**2)
    F_stat = ((TSS_train - RSS_train)/p) / (RSS_train/(n_train - p - 1))
    F_p_value = 1 - f_dist.cdf(F_stat, p, n_train - p - 1)
    
    # Information criteria
    AIC = n_train * np.log(RSS_train/n_train) + 2*(p + 1 if include_intercept else p)
    BIC = n_train * np.log(RSS_train/n_train) + np.log(n_train)*(p + 1 if include_intercept else p)
    
    # VIF calculation
    def calculate_vif(X):
        """Calculate VIF for each variable in X"""
        vif_dict = {}
        for i, col in enumerate(X.columns):
            # For each variable, create a linear model with all other variables as predictors
            y = X[col]
            X_others = X.drop(col, axis=1)
            # If there are no other variables, VIF is 1
            if X_others.shape[1] == 0:
                vif_dict[col] = 1.0
                continue
            # Fit linear model
            model = LinearRegression()
            model.fit(X_others, y)
            # Calculate R² and VIF
            r_squared = model.score(X_others, y)
            vif = 1 / (1 - r_squared)
            vif_dict[col] = vif
        return vif_dict
    
    # Calculate VIF for model variables
    train_data = result_df[train_mask]
    X_train_model = train_data[selected_features]
    vif_values = calculate_vif(X_train_model)
    
    # Durbin-Watson statistic
    def durbin_watson_stat(residuals):
        """Calculate Durbin-Watson statistic"""
        diff_residuals = np.diff(residuals)
        return np.sum(diff_residuals**2) / np.sum(residuals**2)
    
    residuals_train = y_train - y_pred_train
    dw_stat = durbin_watson_stat(residuals_train)
    
    # LOO validation metrics
    loo_metrics = Q2loo_calc(result_df, model)
    Q2loo = loo_metrics["Q2loo"]
    RMSEloo = loo_metrics["RMSEloo"]

    # 5-fold Cross-validation metrics (R²cv, Q²cv)
    train_data_for_cv = result_df[result_df['dataset'] == 'train']
    X_train_cv = train_data_for_cv[selected_features]
    y_train_cv = train_data_for_cv.iloc[:, 1]  # Second column is target
    cv_metrics = calculate_cv_metrics(X_train_cv, y_train_cv, n_folds=5, include_intercept=include_intercept)
    R2cv = cv_metrics['r2cv_train']
    Q2cv = cv_metrics['q2cv']

    # Test set metrics
    y_test = y_all[test_mask]
    y_pred_test = y_pred_all[test_mask]
    
    # External validation metrics
    R2ext = r2_score(y_test, y_pred_test)
    RMSEp = np.sqrt(mean_squared_error(y_test, y_pred_test))
    MAEp = mean_absolute_error(y_test, y_pred_test)
    SSE_test = np.sum((y_test - y_pred_test)**2)
    
    # Q2_test calculation
    train_mean = np.mean(y_train)  # Mean from training set
    TSS_test_vs_train_mean = np.sum((y_test - train_mean)**2)
    
    if TSS_test_vs_train_mean == 0:
        Q2_test = np.nan
    else:
        Q2_test = 1 - (SSE_test / TSS_test_vs_train_mean)
    
    # Concordance Correlation Coefficient
    mean_test = np.mean(y_test)
    mean_pred_test = np.mean(y_pred_test)
    var_test = np.var(y_test, ddof=1)
    var_pred_test = np.var(y_pred_test, ddof=1)
    covar = np.cov(y_test, y_pred_test)[0, 1]
    CCCext = (2 * covar) / (var_test + var_pred_test + (mean_test - mean_pred_test)**2)
    
    # Generate plots
    # Predicted vs Actual

    plt.figure(figsize=(10, 8))
    plt.scatter(y_train, y_pred_train, alpha=0.7, color='#440154', s=70, label='Training set')
    plt.scatter(y_test, y_pred_test, alpha=0.7, color='#35b779', s=70, label='Validation set')
    #plt.plot([min(y_all), max(y_all)], [min(y_all), max(y_all)], 'k--', lw=2, alpha=0.1) #linia plotu ale nie z tym alpha interval

    # Combine data for regplot with confidence interval
    all_actual = np.concatenate([y_train, y_test])
    all_predicted = np.concatenate([y_pred_train, y_pred_test])
    data_for_regplot = pd.DataFrame({
        'Actual': all_actual,
        'Predicted': all_predicted
    })

    # Add regression line WITH confidence interval using seaborn
    sns.regplot(data=data_for_regplot, x='Actual', y='Predicted', 
            scatter=False, ci=99, line_kws={'color': 'gray', 'alpha': 0.1})

    plt.title(f'Predicted vs Actual - {target_var}', fontsize=20)
    plt.xlabel(f'Experimental {target_var}', fontsize=20)
    plt.ylabel(f'Predicted {target_var}', fontsize=20)
    plt.legend(title='', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    add_watermark_matplotlib_after_plot(plt.gcf())
    pred_actual_plot_path = os.path.join(temp_path, 'mlr_pred_actual_plot.png')
    plt.savefig(pred_actual_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Residuals plot
    plt.figure(figsize=(10, 8))
    plt.scatter(y_pred_train, residuals_train, alpha=0.7, color='#440154', s=70, label='Training set')
    plt.scatter(y_pred_test, y_test - y_pred_test, alpha=0.7, color='#35b779', s=70, label='Validation set')
    plt.axhline(y=0, color='k', linestyle='--')
    plt.title('Residuals vs Predicted Values', fontsize=20)
    plt.xlabel('Predicted Values', fontsize=20)
    plt.ylabel('Residuals', fontsize=20)
    plt.legend(title='', fontsize=12)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    add_watermark_matplotlib_after_plot(plt.gcf())
    residuals_plot_path = os.path.join(temp_path, 'mlr_residuals_plot.png')
    plt.savefig(residuals_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Residuals histogram
    plt.figure(figsize=(10, 8))
    sns.histplot(residuals_train, kde=True, color='#440154')
    plt.axvline(x=0, color='#35b779', linestyle='--')
    plt.title('Histogram of Residuals', fontsize=20)
    plt.xlabel('Residuals', fontsize=20)
    plt.ylabel('Frequency', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    add_watermark_matplotlib_after_plot(plt.gcf())
    residuals_hist_path = os.path.join(temp_path, 'mlr_residuals_hist.png')
    plt.savefig(residuals_hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # QQ Plot
    plt.figure(figsize=(10, 8))
    stats.probplot(residuals_train, dist="norm", plot=plt)
    plt.title('Q-Q Plot of Residuals', fontsize=20)
    plt.xlabel('Theoretical Quantiles', fontsize=20)
    plt.ylabel('Sample Quantiles', fontsize=20)
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    add_watermark_matplotlib_after_plot(plt.gcf())
    qq_plot_path = os.path.join(temp_path, 'mlr_qq_plot.png')
    plt.savefig(qq_plot_path, dpi=150, bbox_inches='tight')
    plt.close()
        # Williams Plot - for Applicability Domain analysis
    try:
        # Przygotuj model dla funkcji williams_plot
        class SimplePredictionModel:
            def __init__(self, coefficients, intercept, variables):
                self.coefficients = coefficients
                self.intercept = intercept
                self.variables = variables
            
            def predict(self, X):
                # Sprawdź, czy X jest DataFrame, i przekształć na numpy array jeśli tak
                if isinstance(X, pd.DataFrame):
                    # Upewnij się, że kolumny są w tej samej kolejności co podczas treningu
                    X_ordered = X[self.variables].values if len(X.shape) > 1 else X.values
                else:
                    X_ordered = X
                
                # Dokonaj predykcji
                if len(X_ordered.shape) == 1:
                    X_ordered = X_ordered.reshape(1, -1)
                    
                preds = np.dot(X_ordered, self.coefficients)
                if self.intercept is not None:
                    preds += self.intercept
                
                return preds
        
        # Utwórz uproszczony model
        if include_intercept:
            intercept = sm_model.params[0]
            coeffs = sm_model.params[1:]
        else:
            intercept = None
            coeffs = sm_model.params
        
        simple_model = SimplePredictionModel(coeffs, intercept, selected_features)
        
        # Generuj dane wykresu Williamsa
        wp_result = williams_plot(result_df, simple_model)
        # Utwórz wykres
        wp_plot = plot_wp(wp_result)
        # Dodaj watermark i zapisz wykres do pliku
        add_watermark_matplotlib_after_plot(wp_plot)
        williams_plot_path = os.path.join(temp_path, 'mlr_williams_plot.png')
        wp_plot.savefig(williams_plot_path, dpi=150, bbox_inches='tight')
        plt.close(wp_plot)
        
        # Dodaj wartości AD do wyników
        AD_train = wp_result['ADVal'][0]
        AD_test = wp_result['ADVal'][1]
        h_star = wp_result['h_star']
        williams_outliers = wp_result['outliers']
    except Exception as e:
        print(f"Error generating Williams plot: {str(e)}")
        williams_plot_path = None
        AD_train = None
        AD_test = None
        h_star = None
        williams_outliers = []

    # Get coefficients and statistics
    if include_intercept:
        feature_names = ['Intercept'] + selected_features
        coefficients = sm_model.params.tolist()
        std_errors = sm_model.bse.tolist()
        t_values = sm_model.tvalues.tolist()
        p_values = sm_model.pvalues.tolist()
    else:
        feature_names = selected_features
        coefficients = sm_model.params.tolist()
        std_errors = sm_model.bse.tolist()
        t_values = sm_model.tvalues.tolist()
        p_values = sm_model.pvalues.tolist()

    # Override coefficients with GA values if provided
    if ga_coefficients is not None and ga_intercept is not None:
        if include_intercept:
            coefficients = [ga_intercept] + list(ga_coefficients)
        else:
            coefficients = list(ga_coefficients)

    # Compile results
    results = {
        # Main metrics - R² and Q² with stratified 1:3 split (first/last in test)
        'train_r2': R2tr,           # R² on training set
        'q2_test': Q2_test,         # Q² on test set (external validation)
        # Cross-validation metrics (5-fold)
        'r2cv': R2cv,               # R²cv (5-fold) on training data
        'q2cv': Q2cv,               # Q²cv (5-fold) on training data
        # Leave-One-Out metrics
        'q2_loo': Q2loo,            # Q²loo
        'rmse_loo': RMSEloo,        # RMSEloo
        # Error metrics
        'train_rmse': RMSEtr,       # RMSE on training
        'train_mae': MAEtr,         # MAE on training
        'test_rmse': RMSEp,         # RMSEp (prediction RMSE on test)
        'test_mae': MAEp,           # MAEp (prediction MAE on test)
        # Statistical tests
        'f_statistic': F_stat,      # F-statistic
        'f_pvalue': F_p_value,      # F p-value
        'aic': AIC,                 # AIC
        'bic': BIC,                 # BIC
        'dw_stat': dw_stat,         # Durbin-Watson
        # Other metrics
        'ccc_ext': CCCext,          # CCCext (Concordance Correlation Coefficient)
        'vif_values': vif_values,   # VIF values
        # Legacy metrics (kept for compatibility)
        'adj_r2': AdjR2tr,
        'test_r2': R2ext,
        # Coefficients and statistics
        'coefficients': coefficients,
        'std_errors': std_errors,
        't_values': t_values,
        'p_values': p_values,
        'feature_names': feature_names,
        'split_method': split_method,
        # Plots
        'mlr_pred_actual_plot': pred_actual_plot_path,
        'mlr_residuals_plot': residuals_plot_path,
        'mlr_residuals_hist': residuals_hist_path,
        'mlr_qq_plot': qq_plot_path,
        'mlr_williams_plot': williams_plot_path,
        # Williams plot metrics
        'AD_train': AD_train,
        'AD_test': AD_test,
        'h_star': h_star,
        'williams_outliers': williams_outliers
    }
    
    # Add cross-validation specific metrics
    if split_method in ['kfold']:
        results.update({
            'cv_train_r2_mean': cv_train_r2_mean,
            'cv_test_r2_mean': cv_test_r2_mean,
            'cv_train_rmse_mean': cv_train_rmse_mean,
            'cv_test_rmse_mean': cv_test_rmse_mean,
            'cv_train_r2': cv_train_r2,
            'cv_test_r2': cv_test_r2,
            'cv_train_rmse': cv_train_rmse,
            'cv_test_rmse': cv_test_rmse,
            'n_folds': split_params.get('n_folds', 5)
        })
    
    return results

def Q2loo_calc(result_df, model):
    """
    Calculate Q2loo (Leave-One-Out cross-validation R²) and RMSE_loo
    
    Parameters:
    -----------
    result_df : DataFrame
        DataFrame containing the data with columns for actual values, predictions, and dataset flag
    model : object
        Model object containing variables attribute with predictor column names
    
    Returns:
    --------
    dict
        Dictionary with Q2loo and RMSEloo metrics
    """
    # Filter to get only training data
    train_data = result_df[result_df['dataset'] == 'train']
    
    # Prepare data - directly use the model variables instead of trying to drop 'dataset'
    X_train = train_data[model.variables]  # Use only model variables
    y_train = train_data.iloc[:, 1].values  # Second column contains actual values
    
    # Initialize arrays for predictions
    n = len(y_train)
    y_pred_loo = np.zeros(n)
    
    # LOO cross-validation
    for i in range(n):
        # Split data
        X_train_loo = np.delete(X_train.values, i, axis=0)
        y_train_loo = np.delete(y_train, i)
        X_test_loo = X_train.values[i].reshape(1, -1)
        
        # Fit model
        model_loo = LinearRegression(fit_intercept=model.include_intercept)
        model_loo.fit(X_train_loo, y_train_loo)
        
        # Predict
        y_pred_loo[i] = model_loo.predict(X_test_loo)[0]
    
    # Calculate PRESS (Predictive Residual Sum of Squares)
    press = np.sum((y_train - y_pred_loo)**2)
    
    # Calculate TSS (Total Sum of Squares)
    tss = np.sum((y_train - np.mean(y_train))**2)
    
    # Calculate Q2loo
    Q2loo = 1 - (press / tss)
    
    # Calculate RMSEloo
    RMSEloo = np.sqrt(press / n)
    
    return {
        "Q2loo": Q2loo,
        "RMSEloo": RMSEloo,
        "LOO_predictions": y_pred_loo
    }


def generate_predictions_file(data_path, target_var, selected_features, include_intercept=True, temp_path='temp/'):
    """Generate a CSV file with actual values, predictions, and residuals"""
    df = pd.read_csv(data_path)
    
    # Prepare data
    X = df[selected_features]
    y = df[target_var]
    
    # Fit model
    if include_intercept:
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        predictions = model.predict(X_sm)
    else:
        model = sm.OLS(y, X).fit()
        predictions = model.predict(X)
    
    # Create predictions DataFrame
    pred_df = pd.DataFrame({
        'Actual': y,
        'Predicted': predictions,
        'Residuals': y - predictions
    })
    
    # Add features
    for feature in selected_features:
        pred_df[feature] = df[feature]
    
    # Save to temp file
    os.makedirs(temp_path, exist_ok=True)
    temp_file = os.path.join(temp_path, 'mlr_predictions.csv')
    pred_df.to_csv(temp_file, index=False)
    
    return temp_file

def generate_model_file(data_path, target_var, selected_features, include_intercept=True, temp_path='temp/'):
    """Generate a CSV file with model coefficients and statistics"""
    df = pd.read_csv(data_path)
    
    # Prepare data
    X = df[selected_features]
    y = df[target_var]
    
    # Fit model
    if include_intercept:
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        feature_names = ['Intercept'] + selected_features
    else:
        model = sm.OLS(y, X).fit()
        feature_names = selected_features
    
    # Create model summary DataFrame
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'Coefficient': model.params,
        'Std Error': model.bse,
        't-value': model.tvalues,
        'p-value': model.pvalues,
    })
    
    # Save to temp file
    os.makedirs(temp_path, exist_ok=True)
    temp_file = os.path.join(temp_path, 'mlr_model_summary.csv')
    coef_df.to_csv(temp_file, index=False)
    
    return temp_file

def generate_report(data_path, target_var, selected_features, include_intercept=True, temp_path='temp/',
                    metrics=None, plot_paths=None):
    """Generate a PDF report with model results

    Parameters:
    -----------
    data_path : str
        Path to the dataset
    target_var : str
        Target variable name
    selected_features : list
        List of feature names
    include_intercept : bool
        Whether intercept is included
    temp_path : str
        Path to store temporary files
    metrics : dict, optional
        Pre-calculated metrics from MLR analysis (train_r2, test_r2, q2_loo, etc.)
    plot_paths : dict, optional
        Paths to plot images (mlr_pred_actual_plot, mlr_williams_plot, etc.)
    """
    from fpdf import FPDF
    import datetime

    # Read data and fit model
    df = pd.read_csv(data_path)
    X = df[selected_features]
    y = df[target_var]

    # Fit model
    if include_intercept:
        X_sm = sm.add_constant(X)
        model = sm.OLS(y, X_sm).fit()
        feature_names = ['Intercept'] + selected_features
    else:
        model = sm.OLS(y, X).fit()
        feature_names = selected_features

    # Create PDF
    pdf = FPDF()
    pdf.add_page()

    # Title
    pdf.set_font('Helvetica', 'B', 16)
    pdf.cell(0, 10, 'MLR Analysis Report', ln=True, align='C')
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Generated: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', ln=True, align='C')
    pdf.ln(10)

    # Model Information
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Model Information', ln=True)
    pdf.set_font('Helvetica', '', 10)
    pdf.cell(0, 6, f'Target Variable: {target_var}', ln=True)
    pdf.cell(0, 6, f'Number of Features: {len(selected_features)}', ln=True)
    pdf.cell(0, 6, f'Features: {", ".join(selected_features)}', ln=True)
    pdf.cell(0, 6, f'Include Intercept: {"Yes" if include_intercept else "No"}', ln=True)
    pdf.cell(0, 6, f'Number of Observations: {len(y)}', ln=True)
    pdf.ln(8)

    # Regression Equation
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Regression Equation', ln=True)
    pdf.set_font('Helvetica', '', 10)

    equation_parts = [f'{target_var} = ']
    for i, (name, coef) in enumerate(zip(feature_names, model.params)):
        if i == 0 and name == 'Intercept':
            equation_parts.append(f'{coef:.4f}')
        else:
            sign = '+' if coef >= 0 else '-'
            if i > 0 or (i == 0 and name != 'Intercept'):
                equation_parts.append(f' {sign} {abs(coef):.4f} x {name}')
            else:
                equation_parts.append(f'{coef:.4f} x {name}')

    pdf.multi_cell(0, 6, ''.join(equation_parts))
    pdf.ln(8)

    # Model Performance - use pre-calculated metrics if available
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Model Performance', ln=True)
    pdf.set_font('Helvetica', '', 10)

    if metrics:
        # Training Set Metrics
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Training Set:', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5, f'  R2 (train): {metrics.get("train_r2", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  Adjusted R2: {metrics.get("adj_r2", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  RMSE (train): {metrics.get("train_rmse", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  MAE (train): {metrics.get("train_mae", 0):.4f}', ln=True)
        pdf.ln(3)

        # Validation Metrics
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Validation:', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5, f'  R2 (test): {metrics.get("test_r2", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  Q2 (LOO): {metrics.get("q2_loo", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  Q2 (external): {metrics.get("q2_test", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  RMSE (test): {metrics.get("test_rmse", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  RMSE (LOO): {metrics.get("rmse_loo", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  CCC (external): {metrics.get("ccc_ext", 0):.4f}', ln=True)
        pdf.ln(3)

        # Model Statistics
        pdf.set_font('Helvetica', 'B', 10)
        pdf.cell(0, 6, 'Model Statistics:', ln=True)
        pdf.set_font('Helvetica', '', 10)
        pdf.cell(0, 5, f'  F-statistic: {metrics.get("f_statistic", 0):.4f} (p={metrics.get("f_pvalue", 0):.2e})', ln=True)
        pdf.cell(0, 5, f'  AIC: {metrics.get("aic", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  BIC: {metrics.get("bic", 0):.4f}', ln=True)
        pdf.cell(0, 5, f'  Durbin-Watson: {metrics.get("dw_stat", 0):.4f}', ln=True)

        # Applicability Domain
        if metrics.get('AD_train') is not None:
            pdf.ln(3)
            pdf.set_font('Helvetica', 'B', 10)
            pdf.cell(0, 6, 'Applicability Domain:', ln=True)
            pdf.set_font('Helvetica', '', 10)
            pdf.cell(0, 5, f'  AD coverage (train): {metrics.get("AD_train", 0):.2f}%', ln=True)
            pdf.cell(0, 5, f'  AD coverage (test): {metrics.get("AD_test", 0):.2f}%', ln=True)
            pdf.cell(0, 5, f'  h* threshold: {metrics.get("h_star", 0):.4f}', ln=True)
    else:
        # Fallback to basic metrics from model
        predictions = model.predict(X_sm if include_intercept else X)
        r2 = r2_score(y, predictions)
        rmse = np.sqrt(mean_squared_error(y, predictions))
        mae = mean_absolute_error(y, predictions)
        pdf.cell(0, 6, f'R-squared: {r2:.4f}', ln=True)
        pdf.cell(0, 6, f'Adjusted R-squared: {model.rsquared_adj:.4f}', ln=True)
        pdf.cell(0, 6, f'RMSE: {rmse:.4f}', ln=True)
        pdf.cell(0, 6, f'MAE: {mae:.4f}', ln=True)
        pdf.cell(0, 6, f'F-statistic: {model.fvalue:.4f} (p-value: {model.f_pvalue:.6f})', ln=True)
        pdf.cell(0, 6, f'AIC: {model.aic:.4f}', ln=True)
        pdf.cell(0, 6, f'BIC: {model.bic:.4f}', ln=True)

    pdf.ln(8)

    # Coefficients Table
    pdf.set_font('Helvetica', 'B', 12)
    pdf.cell(0, 8, 'Coefficients', ln=True)
    pdf.ln(2)

    # Table settings
    col_widths = [40, 28, 24, 24, 28, 20]
    row_height = 7

    # Set fill color for header
    pdf.set_fill_color(220, 220, 220)
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_draw_color(0, 0, 0)
    pdf.set_line_width(0.3)

    # Table header with background
    headers = ['Feature', 'Coefficient', 'Std Error', 't-value', 'p-value', 'Signif.']
    for i, header in enumerate(headers):
        pdf.cell(col_widths[i], row_height, header, border=1, align='C', fill=True)
    pdf.ln()

    # Table rows
    pdf.set_font('Helvetica', '', 9)
    pdf.set_fill_color(255, 255, 255)

    for name, coef, se, t, p in zip(feature_names, model.params, model.bse, model.tvalues, model.pvalues):
        # Significance stars
        if p <= 0.001:
            sig = '***'
        elif p <= 0.01:
            sig = '**'
        elif p <= 0.05:
            sig = '*'
        elif p <= 0.1:
            sig = '.'
        else:
            sig = ''

        # Truncate long feature names
        display_name = name[:15] + '...' if len(name) > 18 else name

        pdf.cell(col_widths[0], row_height, display_name, border=1, align='L')
        pdf.cell(col_widths[1], row_height, f'{coef:.4f}', border=1, align='R')
        pdf.cell(col_widths[2], row_height, f'{se:.4f}', border=1, align='R')
        pdf.cell(col_widths[3], row_height, f'{t:.4f}', border=1, align='R')
        pdf.cell(col_widths[4], row_height, f'{p:.6f}', border=1, align='R')
        pdf.cell(col_widths[5], row_height, sig, border=1, align='C')
        pdf.ln()

    pdf.ln(3)
    pdf.set_font('Helvetica', 'I', 8)
    pdf.cell(0, 5, "Significance codes: '***' p<=0.001  '**' p<=0.01  '*' p<=0.05  '.' p<=0.1", ln=True)

    # Add plots if available
    if plot_paths:
        # New page for plots
        pdf.add_page()
        pdf.set_font('Helvetica', 'B', 12)
        pdf.cell(0, 10, 'Diagnostic Plots', ln=True, align='C')
        pdf.ln(5)

        plot_width = 90  # Width for each plot
        plot_height = 70  # Height for each plot

        # Predicted vs Actual
        if plot_paths.get('pred_actual') and os.path.exists(plot_paths['pred_actual']):
            pdf.image(plot_paths['pred_actual'], x=10, y=pdf.get_y(), w=plot_width, h=plot_height)

        # Williams Plot
        if plot_paths.get('williams') and os.path.exists(plot_paths['williams']):
            pdf.image(plot_paths['williams'], x=105, y=pdf.get_y(), w=plot_width, h=plot_height)

        pdf.ln(plot_height + 5)

        # Residuals Histogram
        if plot_paths.get('residuals_hist') and os.path.exists(plot_paths['residuals_hist']):
            pdf.image(plot_paths['residuals_hist'], x=10, y=pdf.get_y(), w=plot_width, h=plot_height)

        # Q-Q Plot
        if plot_paths.get('qq') and os.path.exists(plot_paths['qq']):
            pdf.image(plot_paths['qq'], x=105, y=pdf.get_y(), w=plot_width, h=plot_height)

    # Save PDF
    os.makedirs(temp_path, exist_ok=True)
    temp_file = os.path.join(temp_path, 'mlr_report.pdf')
    pdf.output(temp_file)

    return temp_file

def generate_outliers_file(outliers, h_star, data_path, target_var, selected_features, temp_path='temp/'):
    """Generate a CSV file with outliers from Williams Plot (Applicability Domain)

    Includes full original data for each outlier sample along with Williams Plot metrics.

    Parameters:
    -----------
    outliers : list
        List of dictionaries containing outlier information (index, dataset, leverage, std_residual)
    h_star : float
        Leverage threshold (h*)
    data_path : str
        Path to the original CSV dataset
    target_var : str
        Name of the target variable
    selected_features : list
        List of feature names used in MLR
    temp_path : str
        Path to store temporary files

    Returns:
    --------
    str
        Path to the generated CSV file
    """
    os.makedirs(temp_path, exist_ok=True)

    # If no outliers, create empty file with headers
    if not outliers or len(outliers) == 0:
        outliers_df = pd.DataFrame(columns=['Sample_Index', 'Dataset', 'Leverage', 'Std_Residual', 'Reason'])
        temp_file = os.path.join(temp_path, 'williams_outliers.csv')
        outliers_df.to_csv(temp_file, index=False)
        return temp_file

    # Load original dataset
    df_original = pd.read_csv(data_path)

    # Create list to store outlier rows with full data
    outlier_rows = []

    for outlier in outliers:
        sample_idx = outlier['index']

        # Get the full row from original dataset
        if sample_idx < len(df_original):
            row_data = df_original.iloc[sample_idx].to_dict()

            # Add Williams Plot metrics
            row_data['WP_Dataset'] = outlier['dataset']
            row_data['WP_Leverage'] = outlier['leverage']
            row_data['WP_Std_Residual'] = outlier['std_residual']

            # Determine reason
            if outlier['leverage'] >= h_star and abs(outlier['std_residual']) >= 3:
                row_data['WP_Reason'] = 'High leverage & outlier'
            elif outlier['leverage'] >= h_star:
                row_data['WP_Reason'] = 'High leverage'
            else:
                row_data['WP_Reason'] = 'Outlier residual'

            # Add sample index as first column
            row_data['Sample_Index'] = sample_idx

            outlier_rows.append(row_data)

    # Create DataFrame
    outliers_df = pd.DataFrame(outlier_rows)

    # Reorder columns: Sample_Index first, then Williams Plot metrics, then original data
    williams_cols = ['Sample_Index', 'WP_Dataset', 'WP_Leverage', 'WP_Std_Residual', 'WP_Reason']
    original_cols = [col for col in outliers_df.columns if col not in williams_cols]

    # Prioritize: Sample_Index, ID columns (if any), target_var, selected_features, then rest
    priority_cols = williams_cols.copy()

    # Try to find ID columns (common names)
    id_candidates = [col for col in original_cols if col.lower() in ['id', 'sample_id', 'name', 'compound', 'sample_name', 'sample']]
    priority_cols.extend(id_candidates)

    # Add target variable
    if target_var in original_cols and target_var not in priority_cols:
        priority_cols.append(target_var)

    # Add selected features
    for feat in selected_features:
        if feat in original_cols and feat not in priority_cols:
            priority_cols.append(feat)

    # Add remaining columns
    remaining_cols = [col for col in original_cols if col not in priority_cols]
    final_cols = priority_cols + remaining_cols

    outliers_df = outliers_df[final_cols]

    # Save to CSV
    temp_file = os.path.join(temp_path, 'williams_outliers.csv')
    outliers_df.to_csv(temp_file, index=False)

    return temp_file


def apply_custom_model(df, coefficients, intercept=None, prediction_column_name='Y_pred', temp_path='temp/'):
    """
    Apply a custom model with user-defined coefficients to generate predictions.

    This function allows users to apply pre-defined regression coefficients
    to their data without training a model (e.g., for HLB prediction using
    known equations).

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing the data
    coefficients : dict
        Dictionary mapping feature names to their coefficient values
        e.g., {'X1': 0.5, 'X2': -0.3, 'X3': 1.2}
    intercept : float, optional
        Intercept (constant) term. If None, no intercept is used.
    prediction_column_name : str, default='Y_pred'
        Name for the prediction column
    temp_path : str, default='temp/'
        Path to store temporary files

    Returns:
    --------
    dict
        Dictionary containing:
        - n_samples: number of samples
        - mean: mean of predictions
        - std: standard deviation of predictions
        - min: minimum prediction value
        - max: maximum prediction value
        - preview: list of first 20 rows as dictionaries
    """
    # Ensure temp directory exists
    os.makedirs(temp_path, exist_ok=True)

    # Validate that all required features are in the DataFrame
    missing_features = [f for f in coefficients.keys() if f not in df.columns]
    if missing_features:
        raise ValueError(f"Features not found in dataset: {missing_features}")

    # Calculate predictions
    predictions = np.zeros(len(df))

    # Add intercept if provided
    if intercept is not None:
        predictions += intercept

    # Add contribution from each feature
    for feature, coef in coefficients.items():
        predictions += df[feature].values * coef

    # Create results dictionary
    n_samples = len(predictions)
    pred_mean = float(np.mean(predictions))
    pred_std = float(np.std(predictions))
    pred_min = float(np.min(predictions))
    pred_max = float(np.max(predictions))

    # Create preview (first 20 rows)
    preview_df = df[list(coefficients.keys())].head(20).copy()
    preview_df[prediction_column_name] = predictions[:20]
    preview = preview_df.to_dict('records')

    return {
        'n_samples': n_samples,
        'mean': pred_mean,
        'std': pred_std,
        'min': pred_min,
        'max': pred_max,
        'preview': preview
    }


def generate_custom_predictions_file(data_path, coefficients, intercept=None,
                                     prediction_column_name='Y_pred', temp_path='temp/'):
    """
    Generate a CSV file with predictions from a custom model.

    Parameters:
    -----------
    data_path : str
        Path to the CSV dataset
    coefficients : dict
        Dictionary mapping feature names to their coefficient values
    intercept : float, optional
        Intercept (constant) term. If None, no intercept is used.
    prediction_column_name : str, default='Y_pred'
        Name for the prediction column
    temp_path : str, default='temp/'
        Path to store temporary files

    Returns:
    --------
    str
        Path to the generated CSV file
    """
    # Read data
    df = pd.read_csv(data_path)

    # Calculate predictions
    predictions = np.zeros(len(df))

    # Add intercept if provided
    if intercept is not None:
        predictions += intercept

    # Add contribution from each feature
    for feature, coef in coefficients.items():
        predictions += df[feature].values * coef

    # Create output DataFrame with all original columns plus predictions
    result_df = df.copy()
    result_df[prediction_column_name] = predictions

    # Reorder columns to put prediction at the beginning after any ID columns
    # Try to find ID-like columns
    id_cols = [col for col in result_df.columns if col.lower() in
               ['id', 'sample_id', 'name', 'compound', 'sample_name', 'sample', 'index']]

    # Put ID columns first, then prediction, then features used, then rest
    feature_cols = list(coefficients.keys())
    other_cols = [col for col in result_df.columns
                  if col not in id_cols and col != prediction_column_name and col not in feature_cols]

    new_order = id_cols + [prediction_column_name] + feature_cols + other_cols
    result_df = result_df[new_order]

    # Save to temp file
    os.makedirs(temp_path, exist_ok=True)
    temp_file = os.path.join(temp_path, f'{prediction_column_name}_predictions.csv')
    result_df.to_csv(temp_file, index=False)

    return temp_file