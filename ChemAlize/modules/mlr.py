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
    
    X_combined = pd.concat([X_train, X_test], ignore_index=True)
    H_combined = hat_matrix(X_combined.values)
    
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    residual_train = np.abs(y_train - y_pred_train)
    residual_test = np.abs(y_test - y_pred_test)
    s_residual_train = (residual_train - np.mean(residual_train)) / np.std(residual_train)
    s_residual_test = (residual_test - np.mean(residual_test)) / np.std(residual_test)
    
    leverage_train = np.diag(H_combined)[:len(X_train)]
    leverage_test = np.diag(H_combined)[len(X_train):]
    
    p = len(model.variables)
    n = len(X_train)
    h_star = (3 * (p + 1)) / n
    
    AD_train = 100 * np.sum((leverage_train < h_star) & (np.abs(s_residual_train) < 3)) / len(leverage_train)
    AD_test = 100 * np.sum((leverage_test < h_star) & (np.abs(s_residual_test) < 3)) / len(leverage_test)
    
    lev = np.concatenate([leverage_train, leverage_test])
    res = np.concatenate([s_residual_train, s_residual_test])
    group = np.concatenate([['Train'] * len(X_train), ['Test'] * len(X_test)])
    
    data_to_plot = pd.DataFrame({'lev': lev, 'res': res, 'group': group})
    
    return {
        'ADVal': [AD_train, AD_test], 
        'DTP': data_to_plot, 
        'h_star': h_star
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

def perform_mlr(df, target_var, selected_features, include_intercept=True, 
                split_method='random', split_params=None, scale_data=False,
                check_assumptions=True, detect_outliers=False, temp_path='temp/'):
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
    split_method : str, default='random'
        Method to split the data ('random', 'stratified', 'time', 'kfold', 'loocv')
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
    
    if split_method == 'random':
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
        # Zapisz wykres do pliku
        williams_plot_path = os.path.join(temp_path, 'mlr_williams_plot.png')
        wp_plot.savefig(williams_plot_path, dpi=150, bbox_inches='tight')
        plt.close(wp_plot)
        
        # Dodaj wartości AD do wyników
        AD_train = wp_result['ADVal'][0]
        AD_test = wp_result['ADVal'][1]
        h_star = wp_result['h_star']
    except Exception as e:
        print(f"Error generating Williams plot: {str(e)}")
        williams_plot_path = None
        AD_train = None
        AD_test = None
        h_star = None

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
    
    # Compile results
    results = {
        'train_r2': R2tr,
        'adj_r2': AdjR2tr,
        'test_r2': R2ext,
        'q2_loo': Q2loo,
        'q2_test': Q2_test,
        'train_rmse': RMSEtr,
        'test_rmse': RMSEp,
        'rmse_loo': RMSEloo,
        'train_mae': MAEtr,
        'test_mae': MAEp,
        'f_statistic': F_stat,
        'f_pvalue': F_p_value,
        'aic': AIC,
        'bic': BIC,
        'dw_stat': dw_stat,
        'vif_values': vif_values,
        'ccc_ext': CCCext,
        'coefficients': coefficients,
        'std_errors': std_errors,
        't_values': t_values,
        'p_values': p_values,
        'feature_names': feature_names,
        'split_method': split_method,
        'mlr_pred_actual_plot': pred_actual_plot_path,
        'mlr_residuals_plot': residuals_plot_path,
        'mlr_residuals_hist': residuals_hist_path,
        'mlr_qq_plot': qq_plot_path,
        'mlr_williams_plot': williams_plot_path,
        'AD_train': AD_train,
        'AD_test': AD_test,
        'h_star': h_star
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

def generate_report(data_path, target_var, selected_features, include_intercept=True, temp_path='temp/'):
    """Generate a PDF report with model results"""
    # This is a placeholder function that would generate a PDF report
    # In a real implementation, you'd use a PDF generation library like reportlab
    
    # For now, we'll just create a text file as a placeholder
    os.makedirs(temp_path, exist_ok=True)
    temp_file = os.path.join(temp_path, 'mlr_report.pdf')
    
    with open(temp_file, 'w') as f:
        f.write(f"MLR Analysis Report\n")
        f.write(f"Target Variable: {target_var}\n")
        f.write(f"Features: {', '.join(selected_features)}\n")
        f.write(f"Include Intercept: {'Yes' if include_intercept else 'No'}\n")
    
    return temp_file