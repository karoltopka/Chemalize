import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, KeepTogether
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.utils import ImageReader
import seaborn as sns
import io
from PIL import Image as PILImage
from app.utils.watermark import add_watermark_matplotlib_after_plot

# Set the Seaborn style for better looking plots
sns.set(style="whitegrid")

# Global variables to store data for file generation
_pca_model = None
_pc_scores = None
_loadings_df = None
_selected_features = None
_feature_importance = None
_df_clean = None

def perform_enhanced_pca(df, n_components=2, scale_data=True, show_variance=True,
                        show_scatter=True, show_loading=True, show_biplot=True,
                        color_by=None, pc_x_axis=1, pc_y_axis=2, pc_loadings_select=[1, 2],
                        feature_selection_method='all', top_n_features=5, loading_threshold=0.3,
                        show_top_features_plot=False, show_feature_importance=False,
                        top_n_arrows=None,
                        temp_path='temp/'):
    """
    Perform Enhanced Principal Component Analysis with advanced feature selection.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input data.
    n_components : int, default=2
        Number of principal components to retain.
    scale_data : bool, default=True
        Whether to standardize the data before PCA.
    show_variance : bool, default=True
        Whether to create a plot showing explained variance.
    show_scatter : bool, default=True
        Whether to create a scatter plot of the principal components.
    show_loading : bool, default=True
        Whether to create a loading plot showing feature contributions.
    show_biplot : bool, default=True
        Whether to create a biplot showing samples and feature vectors.
    color_by : str, default=None
        Column name to use for coloring points in the scatter plot.
    pc_x_axis : int, default=1
        Principal component for X-axis.
    pc_y_axis : int, default=2
        Principal component for Y-axis.
    pc_loadings_select : list, default=[1, 2]
        List of PC numbers to show in loadings plot.
    feature_selection_method : str, default='all'
        Method for feature selection ('all', 'top_n', 'threshold').
    top_n_features : int, default=5
        Number of top features per PC when using 'top_n' method.
    loading_threshold : float, default=0.3
        Threshold for feature loadings when using 'threshold' method.
    show_top_features_plot : bool, default=False
        Whether to create plots with selected features only.
    show_feature_importance : bool, default=False
        Whether to create feature importance plot.
    temp_path : str, default='temp/'
        Path to temporary directory for saving plots.
        
    Returns:
    --------
    dict
        Dictionary containing PCA results and plot paths.
    """
    global _pca_model, _pc_scores, _loadings_df, _selected_features, _feature_importance, _df_clean
    
    # Ensure the temporary directory exists
    os.makedirs(temp_path, exist_ok=True)
    
    # Handle categorical variables and missing values
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Drop columns with any missing values for this analysis
    df_clean = df_numeric.dropna(axis=1)
    _df_clean = df_clean
    
    if df_clean.shape[1] < 2:
        raise ValueError("Not enough numeric columns without missing values for PCA.")
    
    # Extract features (X)
    X = df_clean.values
    
    # Scale the data if requested
    scaler = None
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Initialize the PCA model
    pca_model = PCA(n_components=min(n_components, min(X.shape)))
    _pca_model = pca_model
    
    # Fit the model and transform the data
    principal_components = pca_model.fit_transform(X)
    _pc_scores = principal_components
    
    # Create a DataFrame with principal components
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)]
    )
    
    # Add color column if specified and exists in the original DataFrame
    if color_by and color_by in df.columns:
        pc_df[color_by] = df[color_by].reset_index(drop=True)
    
    # FIXED: Calculate proper normalized loadings (correlations between original variables and PCs)
    # Loadings = components * sqrt(explained_variance) - this gives correlations in [-1,1] range
    loadings_raw = pca_model.components_ * np.sqrt(pca_model.explained_variance_)[:, np.newaxis]
    
    # Alternative method if data was scaled: loadings = components * sqrt(explained_variance)
    # If data wasn't scaled, we need to account for the standard deviations
    if not scale_data:
        # When data is not scaled, we need to divide by standard deviations to get correlations
        std_devs = np.std(df_clean.values, axis=0, ddof=1)
        loadings_raw = loadings_raw / std_devs[np.newaxis, :]
    
    # Create a loadings DataFrame with proper normalized values
    loadings_df = pd.DataFrame(
        data=loadings_raw.T,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
        index=df_clean.columns
    )
    _loadings_df = loadings_df
    
    # Verify loadings are in [-1, 1] range and clip if necessary
    loadings_df = loadings_df.clip(-1, 1)
    _loadings_df = loadings_df
    
    # Calculate feature importance (sum of squared loadings across all PCs)
    feature_importance = pd.DataFrame({
        'Feature': df_clean.columns,
        'Importance': (loadings_df.abs().max(axis=1))
    }).sort_values('Importance', ascending=False).reset_index(drop=True)
    feature_importance['Rank'] = np.arange(1, len(feature_importance)+1)
    _feature_importance = feature_importance
    
    # Feature selection based on method
    selected_features = []
    feature_selection_summary = {}
    
    if feature_selection_method == 'top_n':
        # Get top N features for each selected PC
        for pc_num in pc_loadings_select:
            if pc_num <= pca_model.n_components_:
                pc_col = f'PC{pc_num}'
                top_features = loadings_df[pc_col].abs().nlargest(top_n_features).index.tolist()
                selected_features.extend(top_features)
        selected_features = list(set(selected_features))  # Remove duplicates
        feature_selection_summary = {
            'method': 'Top N features per PC',
            'n_features': top_n_features,
            'pcs': pc_loadings_select
        }
        
    elif feature_selection_method == 'threshold':
        # Get features above threshold for any PC
        for pc_num in range(1, pca_model.n_components_ + 1):
            pc_col = f'PC{pc_num}'
            above_threshold = loadings_df[loadings_df[pc_col].abs() >= loading_threshold].index.tolist()
            selected_features.extend(above_threshold)
        selected_features = list(set(selected_features))  # Remove duplicates
        feature_selection_summary = {
            'method': 'Features above threshold',
            'threshold': loading_threshold,
            'pcs': list(range(1, pca_model.n_components_ + 1))
        }
        
    else:  # 'all'
        selected_features = df_clean.columns.tolist()
        feature_selection_summary = {
            'method': 'All features',
            'n_features': len(selected_features),
            'pcs': list(range(1, pca_model.n_components_ + 1))
        }
    
    _selected_features = selected_features
    
    # Prepare result dictionary
    result = {
        'summary': [
            {
                'eigenvalue': ev,
                'explained_variance': ev_ratio * 100,
                'cumulative_variance': np.sum(pca_model.explained_variance_ratio_[:i+1]) * 100
            }
            for i, (ev, ev_ratio) in enumerate(zip(
                pca_model.explained_variance_, 
                pca_model.explained_variance_ratio_
            ))
        ],
        'selected_features_summary': feature_selection_summary,
        'selected_features_count': len(selected_features),
        'features_per_pc': f"{len(selected_features)} total features selected",
        'feature_selection_method_display': feature_selection_summary['method']
    }
    
    # Save the components and loadings to files
    pc_df.to_csv(os.path.join(temp_path, 'pca_components.csv'), index=False)
    loadings_df.to_csv(os.path.join(temp_path, 'pca_loadings.csv'))

    # Save PCA model and scaler for projecting new compounds
    pca_model_path = os.path.join(temp_path, 'pca_model.pkl')
    pca_model_data = {
        'pca_model': pca_model,
        'scaler': scaler,
        'feature_names': df_clean.columns.tolist(),
        'scale_data': scale_data,
        'n_components': pca_model.n_components_
    }
    with open(pca_model_path, 'wb') as f:
        pickle.dump(pca_model_data, f)
    result['pca_model_path'] = pca_model_path
    
    # Generate plots
    if show_variance:
        result['variance_plot'] = create_variance_plot(pca_model, temp_path)
    
    if show_scatter and pca_model.n_components >= max(pc_x_axis, pc_y_axis):
        result['scatter_plot'] = create_enhanced_scatter_plot(
            pc_df, color_by, pc_x_axis, pc_y_axis, temp_path
        )
    
    if show_loading and pc_loadings_select:
        result['loadings_plot'] = create_enhanced_loading_plot(
            loadings_df, pc_loadings_select, temp_path, top_n=50  # Dodać parametr
        )
    
    if show_biplot and pca_model.n_components >= max(pc_x_axis, pc_y_axis):
        result['biplot'] = create_enhanced_biplot(
            principal_components, loadings_raw, df_clean.columns,
            pc_x_axis, pc_y_axis, temp_path, top_n_arrows=top_n_arrows
        )
    
    if show_feature_importance:
        result['feature_importance_plot'] = create_feature_importance_plot(
            feature_importance, temp_path
        )
    
    # Create plots with selected features only (REFIT PCA on the subset)
    if show_top_features_plot and feature_selection_method != 'all' and len(selected_features) >= 2:
        # 1) Create subset
        df_sel = df_clean[selected_features].copy()

        # 2) Scale like in the base run
        X_sel = df_sel.values
        if scale_data:
            scaler_sel = StandardScaler()
            X_sel = scaler_sel.fit_transform(X_sel)

        # 3) Calculate PCA on subset
        n_comp_sel = min(n_components, X_sel.shape[1])
        pca_sel = PCA(n_components=n_comp_sel)
        pc_scores_sel = pca_sel.fit_transform(X_sel)

        # 4) Create DataFrame with scores
        pc_df_sel = pd.DataFrame(
            pc_scores_sel, columns=[f'PC{i+1}' for i in range(pca_sel.n_components_)]
        )
        
        if color_by and color_by in df.columns:
            pc_df_sel[color_by] = df[color_by].reset_index(drop=True)

        # 5) Calculate proper normalized loadings for subset
        loadings_sel_raw = pca_sel.components_ * np.sqrt(pca_sel.explained_variance_)[:, np.newaxis]
        if not scale_data:
            std_devs_sel = np.std(df_sel.values, axis=0, ddof=1)
            loadings_sel_raw = loadings_sel_raw / std_devs_sel[np.newaxis, :]

        # 6) Safety for axes
        pc_x_axis_sel = max(1, min(pc_x_axis, n_comp_sel))
        pc_y_axis_sel = max(1, min(pc_y_axis, n_comp_sel))
        if pc_x_axis_sel == pc_y_axis_sel:
            pc_y_axis_sel = pc_y_axis_sel + 1 if pc_y_axis_sel < n_comp_sel else (1 if pc_x_axis_sel != 1 else 2)

        # 7) Create plots
        if show_scatter:
            plt.figure(figsize=(10, 8))
            xcol, ycol = f'PC{pc_x_axis_sel}', f'PC{pc_y_axis_sel}'
            if color_by and color_by in pc_df_sel.columns:
                if pc_df_sel[color_by].dtype == 'object' or pd.api.types.is_categorical_dtype(pc_df_sel[color_by]):
                    categories = pc_df_sel[color_by].unique()

                    # Plot scatter points for each category
                    for cat in categories:
                        sub = pc_df_sel[pc_df_sel[color_by] == cat]
                        plt.scatter(sub[xcol], sub[ycol], label=cat, alpha=0.7)
                    
                    # Determine legend placement based on number of categories
                    n_categories = len(categories)
                    if n_categories > 6:
                        plt.legend(title=color_by, bbox_to_anchor=(0.5, -0.15), 
                                loc='upper center', ncol=min(4, n_categories))
                        plt.subplots_adjust(bottom=0.2)
                    elif n_categories > 3:
                        plt.legend(title=color_by, bbox_to_anchor=(0.5, -0.1), 
                                loc='upper center', ncol=min(3, n_categories))
                        plt.subplots_adjust(bottom=0.15)
                    else:
                        plt.legend(title=color_by, loc='best')
                else:
                    sc = plt.scatter(pc_df_sel[xcol], pc_df_sel[ycol], c=pc_df_sel[color_by], cmap='viridis', alpha=0.7)
                    plt.colorbar(sc, label=color_by)
            else:
                plt.scatter(pc_df_sel[xcol], pc_df_sel[ycol], alpha=0.7)
            plt.xlabel(f'Principal Component {pc_x_axis_sel}')
            plt.ylabel(f'Principal Component {pc_y_axis_sel}')
            plt.title(f'PCA Scatter (REFIT) — {len(selected_features)} selected features')
            plt.grid(True, linestyle='--', alpha=0.5)
            add_watermark_matplotlib_after_plot(plt.gcf())
            sel_sc_path = os.path.join(temp_path, f'pca_selected_scatter_pc{pc_x_axis_sel}_pc{pc_y_axis_sel}.png')
            plt.savefig(sel_sc_path, dpi=300, bbox_inches='tight')
            plt.close()
            result['selected_scatter_plot'] = sel_sc_path

        if show_biplot:
            plt.figure(figsize=(12, 10))
            xi, yi = pc_x_axis_sel - 1, pc_y_axis_sel - 1
            plt.scatter(pc_scores_sel[:, xi], pc_scores_sel[:, yi], alpha=0.7)
            max_x = max(abs(pc_scores_sel[:, xi]))
            max_y = max(abs(pc_scores_sel[:, yi]))
            for i, feat in enumerate(df_sel.columns):
                lx, ly = loadings_sel_raw[xi, i], loadings_sel_raw[yi, i]
                plt.arrow(0,0, lx*max_x*0.8, ly*max_y*0.8, head_width=max_x*0.02, head_length=max_y*0.02, fc='red', ec='red')
                plt.text(lx*max_x*0.85, ly*max_y*0.85, feat, color='red', ha='center', va='center', fontsize=8)
            plt.xlabel(f'Principal Component {pc_x_axis_sel}')
            plt.ylabel(f'Principal Component {pc_y_axis_sel}')
            plt.title(f'Biplot (REFIT) — {len(selected_features)} selected features')
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.axhline(0, color='k', linestyle='-', alpha=0.3)
            plt.axvline(0, color='k', linestyle='-', alpha=0.3)
            add_watermark_matplotlib_after_plot(plt.gcf())
            sel_bp_path = os.path.join(temp_path, f'pca_selected_biplot_pc{pc_x_axis_sel}_pc{pc_y_axis_sel}.png')
            plt.savefig(sel_bp_path, dpi=300, bbox_inches='tight')
            plt.close()
            result['selected_biplot'] = sel_bp_path
    
    return result


def project_new_compounds(new_df, pca_model_path, identifier_columns=None):
    """
    Project new compounds onto existing PCA space.

    Parameters:
    -----------
    new_df : pandas.DataFrame
        DataFrame with new compounds (must have same features as original PCA)
    pca_model_path : str
        Path to saved PCA model pickle file
    identifier_columns : list, optional
        List of column names to preserve as identifiers (e.g., Sample ID, Compound Name)
        If None, will auto-detect non-numeric columns

    Returns:
    --------
    pandas.DataFrame
        DataFrame with PC scores for new compounds plus identifier columns
    """
    # Load PCA model
    if not os.path.exists(pca_model_path):
        raise FileNotFoundError(f"PCA model not found at {pca_model_path}. Please run PCA analysis first.")

    with open(pca_model_path, 'rb') as f:
        pca_data = pickle.load(f)

    pca_model = pca_data['pca_model']
    scaler = pca_data['scaler']
    feature_names = pca_data['feature_names']
    scale_data = pca_data['scale_data']

    # Preserve identifier columns
    if identifier_columns is None:
        # Auto-detect non-numeric columns
        identifier_columns = new_df.select_dtypes(exclude=['number']).columns.tolist()

    identifier_data = {}
    for col in identifier_columns:
        if col in new_df.columns:
            identifier_data[col] = new_df[col].values

    # Check if all required features are present
    missing_features = [f for f in feature_names if f not in new_df.columns]
    if missing_features:
        raise ValueError(f"Missing required features in new data: {missing_features}")

    # Extract features in the same order as original PCA
    X_new = new_df[feature_names].values

    # Check for missing values
    if np.any(pd.isna(X_new)):
        raise ValueError("New data contains missing values. Please handle missing values before projection.")

    # Scale if necessary
    if scale_data and scaler is not None:
        X_new = scaler.transform(X_new)

    # Project onto PCA space
    pc_scores_new = pca_model.transform(X_new)

    # Create DataFrame with PC scores
    pc_df_new = pd.DataFrame(
        data=pc_scores_new,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)]
    )

    # Add identifier columns
    for col, values in identifier_data.items():
        pc_df_new[col] = values

    return pc_df_new


def create_variance_plot(pca_model, temp_path):
    """Create and save a plot showing explained variance by principal components."""
    plt.figure(figsize=(12, 6))
    
    # Individual explained variance
    plt.bar(
        range(1, len(pca_model.explained_variance_ratio_) + 1),
        pca_model.explained_variance_ratio_ * 100,
        alpha=0.7,
        label='Individual Explained Variance'
    )
    
    # Cumulative explained variance
    plt.step(
        range(1, len(pca_model.explained_variance_ratio_) + 1),
        np.cumsum(pca_model.explained_variance_ratio_) * 100,
        where='mid',
        label='Cumulative Explained Variance',
        color='red'
    )
    
    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance (%)')
    plt.title('Explained Variance by Principal Components')
    plt.xticks(range(1, len(pca_model.explained_variance_ratio_) + 1))
    plt.legend(loc='best')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Add watermark and save the plot
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, 'pca_variance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_enhanced_scatter_plot(pc_df, color_by, pc_x_axis, pc_y_axis, temp_path):
    """Create and save an enhanced scatter plot of specified principal components."""
    plt.figure(figsize=(10, 8))
    
    pc_x_col = f'PC{pc_x_axis}'
    pc_y_col = f'PC{pc_y_axis}'
    
    if color_by and color_by in pc_df.columns:
        # If color column is categorical
        if pc_df[color_by].dtype == 'object' or pd.api.types.is_categorical_dtype(pc_df[color_by]):
            categories = pc_df[color_by].unique()

            # Plot scatter points for each category
            for category in categories:
                subset = pc_df[pc_df[color_by] == category]
                plt.scatter(subset[pc_x_col], subset[pc_y_col], label=category, alpha=0.7)
            
            # Determine legend placement based on number of categories
            n_categories = len(categories)
            if n_categories > 6:
                plt.legend(title=color_by, bbox_to_anchor=(0.5, -0.15), 
                          loc='upper center', ncol=min(4, n_categories))
                plt.subplots_adjust(bottom=0.2)
            elif n_categories > 3:
                plt.legend(title=color_by, bbox_to_anchor=(0.5, -0.1), 
                          loc='upper center', ncol=min(3, n_categories))
                plt.subplots_adjust(bottom=0.15)
            else:
                plt.legend(title=color_by, loc='best')
        else:
            # If color column is numeric
            scatter = plt.scatter(pc_df[pc_x_col], pc_df[pc_y_col], c=pc_df[color_by], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label=color_by)
    else:
        plt.scatter(pc_df[pc_x_col], pc_df[pc_y_col], alpha=0.7)
    
    plt.xlabel(f'Principal Component {pc_x_axis}')
    plt.ylabel(f'Principal Component {pc_y_axis}')
    plt.title(f'PCA Scatter Plot (PC{pc_x_axis} vs PC{pc_y_axis})')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add watermark and save the plot
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, f'pca_scatter_pc{pc_x_axis}_pc{pc_y_axis}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_enhanced_loading_plot(loadings_df, pc_loadings_select, temp_path, top_n=50):
    """Create and save a plot showing feature loadings for selected principal components."""
    
    # Select only the requested PCs
    pc_columns = [f'PC{i}' for i in pc_loadings_select if f'PC{i}' in loadings_df.columns]
    loadings_subset = loadings_df[pc_columns].copy()
    
    # Sort by the sum of absolute loadings across selected PCs and take top N, siła korelacji czyli np 0.6+0.6 będzie wyej niz 0.9+0.1
    loadings_subset['abs_sum'] = loadings_subset.abs().sum(axis=1)
    loadings_subset = loadings_subset.sort_values('abs_sum', ascending=False).head(top_n)
    loadings_subset = loadings_subset.drop('abs_sum', axis=1)
    
    n_features = loadings_subset.shape[0]
    height_in = max(8, min(20, n_features * 0.4))  # Zwiększ wysokość na cechę
    plt.figure(figsize=(14, height_in))

    # Zawsze pokazuj wartości dla top N (bo to już ograniczona lista)
    sns.heatmap(
        loadings_subset, cmap='coolwarm', center=0,
        annot=True, fmt=".3f",  # Zawsze True, 3 miejsca po przecinku
        linewidths=0.5, cbar_kws={'label': 'Loading Value (Correlation)'},
        annot_kws={"size": 8}  # Mniejsza czcionka żeby się zmieściło
    )
    plt.title(f'Top {n_features} Feature Loadings for {", ".join(pc_columns)}')
    plt.xlabel('Principal Components')
    plt.ylabel('Features')
    plt.tight_layout()
    
    # Add watermark and save the plot
    add_watermark_matplotlib_after_plot(plt.gcf())
    pc_names = "_".join([f"pc{i}" for i in pc_loadings_select])
    plot_path = os.path.join(temp_path, f'pca_loadings_{pc_names}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_enhanced_biplot(pc_scores, loadings, feature_names, pc_x_axis, pc_y_axis, temp_path, top_n_arrows=None):
    """Create and save an enhanced biplot showing both samples and feature vectors.

    Parameters:
    -----------
    pc_scores : array
        Principal component scores (samples).
    loadings : array
        Loading matrix.
    feature_names : list
        Names of features.
    pc_x_axis : int
        Principal component for X-axis.
    pc_y_axis : int
        Principal component for Y-axis.
    temp_path : str
        Path to save the plot.
    top_n_arrows : int, optional
        Number of top features (by loading magnitude) to show as arrows.
        If None, all features are shown.
    """
    plt.figure(figsize=(12, 10))

    # Adjust indices for 0-based indexing
    x_idx = pc_x_axis - 1
    y_idx = pc_y_axis - 1

    # Plot the scores (samples)
    plt.scatter(pc_scores[:, x_idx], pc_scores[:, y_idx], alpha=0.7)

    # Plot feature vectors
    max_x = max(abs(pc_scores[:, x_idx]))
    max_y = max(abs(pc_scores[:, y_idx]))

    # Calculate loading magnitude for each feature (Euclidean distance from origin)
    loading_magnitudes = np.sqrt(loadings[x_idx, :]**2 + loadings[y_idx, :]**2)

    # Select top N features if specified
    if top_n_arrows is not None and top_n_arrows > 0:
        # Get indices of top N features by loading magnitude
        top_indices = np.argsort(loading_magnitudes)[::-1][:top_n_arrows]
        selected_features = [(i, feature_names[i], loadings[x_idx, i], loadings[y_idx, i])
                           for i in top_indices]
    else:
        # Use all features
        selected_features = [(i, feature_names[i], loadings[x_idx, i], loadings[y_idx, i])
                           for i in range(len(feature_names))]

    # Draw arrows for selected features
    for i, feature, loading_x, loading_y in selected_features:
        plt.arrow(
            0, 0,  # Start at origin
            loading_x * max_x * 0.8,  # Scale to match scores
            loading_y * max_y * 0.8,
            head_width=max_x*0.02, head_length=max_y*0.02, fc='red', ec='red'
        )
        plt.text(
            loading_x * max_x * 0.85,
            loading_y * max_y * 0.85,
            feature, color='red', ha='center', va='center', fontsize=8
        )

    plt.xlabel(f'Principal Component {pc_x_axis}')
    plt.ylabel(f'Principal Component {pc_y_axis}')
    if top_n_arrows is not None and top_n_arrows > 0:
        plt.title(f'PCA Biplot (PC{pc_x_axis} vs PC{pc_y_axis}) — Top {top_n_arrows} features')
    else:
        plt.title(f'PCA Biplot (PC{pc_x_axis} vs PC{pc_y_axis})')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()

    # Add watermark and save the plot
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, f'pca_biplot_pc{pc_x_axis}_pc{pc_y_axis}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()

    return plot_path

def create_feature_importance_plot(feature_importance, temp_path):
    top_features = feature_importance.head(20)
    height_in = max(8, min(12, len(top_features) * 0.4))
    plt.figure(figsize=(12, height_in))
    
    # Create horizontal bar plot
    plt.barh(range(len(top_features)), top_features['Importance'])
    plt.yticks(range(len(top_features)), top_features['Feature'])
    plt.xlabel('Feature Importance (Sum of Squared Loadings)')
    plt.ylabel('Features')
    plt.title('Top 20 Feature Importance in PCA')
    plt.gca().invert_yaxis()  # Highest importance at top
    plt.grid(True, linestyle='--', alpha=0.5, axis='x')
    plt.tight_layout()
    
    # Add watermark and save the plot
    add_watermark_matplotlib_after_plot(plt.gcf())
    plot_path = os.path.join(temp_path, 'pca_feature_importance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_components_file(dataset_path, temp_path='temp/'):
    """Generate a CSV file with the principal component scores."""
    return os.path.join(temp_path, 'pca_components.csv')

def generate_loadings_file(dataset_path, temp_path='temp/'):
    """Generate a CSV file with the loadings."""
    return os.path.join(temp_path, 'pca_loadings.csv')

def generate_feature_importance_file(temp_path='temp/'):
    """Generate a CSV file with feature importance scores."""
    global _feature_importance
    
    if _feature_importance is not None:
        # Add loadings for each PC to the feature importance table
        importance_with_loadings = _feature_importance.copy()
        
        if _loadings_df is not None:
            # Add individual PC loadings
            for col in _loadings_df.columns:
                importance_with_loadings[f'{col}_Loading'] = [
                    _loadings_df.loc[feature, col] for feature in importance_with_loadings['Feature']
                ]
        
        file_path = os.path.join(temp_path, 'feature_importance.csv')
        importance_with_loadings.to_csv(file_path, index=False)
        return file_path
    
    return None

def generate_selected_features_file(dataset_path, temp_path='temp/', sort_by_loading=True, sort_pc=1):
    """Generate a CSV file with data for selected features only."""
    global _selected_features, _df_clean, _loadings_df, _pc_scores
    
    if _selected_features is not None and _df_clean is not None:
        selected_data = _df_clean[_selected_features].copy()
        
        # Add PC scores
        if _pc_scores is not None:
            pc_df = pd.DataFrame(
                _pc_scores,
                columns=[f'PC{i+1}' for i in range(_pc_scores.shape[1])]
            )
            selected_data = pd.concat([selected_data, pc_df], axis=1)
        
        file_path = os.path.join(temp_path, 'selected_features_data.csv')
        selected_data.to_csv(file_path, index=False)
        
        # Generate additional file with loading information for selected features
        if _loadings_df is not None:
            loadings_info = generate_selected_features_with_loadings(temp_path, sort_by_loading, sort_pc)
        
        return file_path
    
    return None

def generate_selected_features_with_loadings(temp_path='temp/', sort_by_loading=True, sort_pc=1):
    """Generate a detailed CSV with loading information for selected features."""
    global _selected_features, _loadings_df, _feature_importance
    
    if _selected_features is None or _loadings_df is None:
        return None
    
    # Get loadings for selected features only
    selected_loadings = _loadings_df.loc[_selected_features].copy()
    
    # Add feature importance scores
    if _feature_importance is not None:
        # Create a mapping from feature name to importance
        importance_dict = dict(zip(_feature_importance['Feature'], _feature_importance['Importance']))
        selected_loadings['Feature_Importance'] = [
            importance_dict.get(feat, 0) for feat in selected_loadings.index
        ]
        
        # Add rank information
        rank_dict = dict(zip(_feature_importance['Feature'], _feature_importance['Rank']))
        selected_loadings['Importance_Rank'] = [
            rank_dict.get(feat, len(_feature_importance)) for feat in selected_loadings.index
        ]
    
    # Add maximum absolute loading across all PCs
    pc_columns = [col for col in selected_loadings.columns if col.startswith('PC')]
    if pc_columns:
        selected_loadings['Max_Abs_Loading'] = selected_loadings[pc_columns].abs().max(axis=1)
        selected_loadings['Max_Loading_PC'] = selected_loadings[pc_columns].abs().idxmax(axis=1)
        
        # Add the actual max loading value (with sign)
        selected_loadings['Max_Loading_Value'] = [
            selected_loadings.loc[idx, selected_loadings.loc[idx, 'Max_Loading_PC']]
            for idx in selected_loadings.index
        ]
    
    # Add selection reason (why was this feature selected?)
    selected_loadings['Selection_Reason'] = ''
    for idx in selected_loadings.index:
        reasons = []
        for pc_col in pc_columns:
            loading_val = abs(selected_loadings.loc[idx, pc_col])
            if loading_val >= 0.3:  # Default threshold
                reasons.append(f"{pc_col}:{loading_val:.3f}")
        selected_loadings.loc[idx, 'Selection_Reason'] = '; '.join(reasons)
    
    # Sort the dataframe
    if sort_by_loading and f'PC{sort_pc}' in selected_loadings.columns:
        # Sort by absolute loading for specified PC
        selected_loadings = selected_loadings.reindex(
            selected_loadings[f'PC{sort_pc}'].abs().sort_values(ascending=False).index
        )
    elif 'Feature_Importance' in selected_loadings.columns:
        # Sort by feature importance
        selected_loadings = selected_loadings.sort_values('Feature_Importance', ascending=False)
    elif 'Max_Abs_Loading' in selected_loadings.columns:
        # Sort by maximum absolute loading
        selected_loadings = selected_loadings.sort_values('Max_Abs_Loading', ascending=False)
    
    # Reset index to make feature names a column
    selected_loadings = selected_loadings.reset_index()
    selected_loadings.rename(columns={'index': 'Feature_Name'}, inplace=True)
    
    # Reorder columns for better readability
    base_columns = ['Feature_Name']
    if 'Importance_Rank' in selected_loadings.columns:
        base_columns.append('Importance_Rank')
    if 'Feature_Importance' in selected_loadings.columns:
        base_columns.append('Feature_Importance')
    if 'Max_Abs_Loading' in selected_loadings.columns:
        base_columns.extend(['Max_Abs_Loading', 'Max_Loading_PC', 'Max_Loading_Value'])
    
    pc_columns = [col for col in selected_loadings.columns if col.startswith('PC')]
    other_columns = ['Selection_Reason']
    
    final_columns = base_columns + pc_columns + other_columns
    selected_loadings = selected_loadings[final_columns]
    
    # Save to CSV
    file_path = os.path.join(temp_path, 'selected_features_with_loadings.csv')
    selected_loadings.to_csv(file_path, index=False)
    
    # Also create a summary file with just the key information
    summary_columns = ['Feature_Name', 'Importance_Rank', 'Feature_Importance', 
                      'Max_Abs_Loading', 'Max_Loading_PC', 'Selection_Reason']
    summary_columns = [col for col in summary_columns if col in selected_loadings.columns]
    
    summary_df = selected_loadings[summary_columns].copy()
    summary_path = os.path.join(temp_path, 'selected_features_summary.csv')
    summary_df.to_csv(summary_path, index=False)
    
    return file_path

def generate_enhanced_report(dataset_path, temp_path='temp/'):
    """Generate an enhanced PDF report summarizing the PCA results."""
    global _pca_model, _feature_importance, _selected_features
    
    # Read the dataset
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else pd.read_excel(dataset_path)
    
    # Read the PCA results
    pc_df = pd.read_csv(os.path.join(temp_path, 'pca_components.csv'))
    loadings_df = pd.read_csv(os.path.join(temp_path, 'pca_loadings.csv'), index_col=0)
    
    # Create a PDF report
    report_path = os.path.join(temp_path, 'enhanced_pca_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph("Enhanced Principal Component Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Dataset Information
    elements.append(Paragraph("Dataset Information", styles['Heading1']))
    elements.append(Paragraph(f"Number of observations: {len(df)}", styles['Normal']))
    elements.append(Paragraph(f"Number of variables: {len(df.columns)}", styles['Normal']))
    elements.append(Paragraph(f"Number of PC components: {len(pc_df.columns)}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # Feature Selection Summary
    if _selected_features:
        elements.append(Paragraph("Feature Selection", styles['Heading1']))
        elements.append(Paragraph(f"Number of selected features: {len(_selected_features)}", styles['Normal']))
        elements.append(Paragraph(f"Selected features: {', '.join(_selected_features[:10])}" + 
                                ("..." if len(_selected_features) > 10 else ""), styles['Normal']))
        elements.append(Spacer(1, 12))
    
    # Explained Variance Summary
    if _pca_model:
        elements.append(Paragraph("Explained Variance Summary", styles['Heading1']))
        
        # Create a table with variance information
        variance_data = [['Component', 'Explained Variance (%)', 'Cumulative Variance (%)']]
        for i, (ev_ratio) in enumerate(_pca_model.explained_variance_ratio_):
            cumulative = np.sum(_pca_model.explained_variance_ratio_[:i+1]) * 100
            variance_data.append([f'PC{i+1}', f'{ev_ratio*100:.2f}', f'{cumulative:.2f}'])
        
        variance_table = Table(variance_data)
        variance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 14),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(variance_table)
        elements.append(Spacer(1, 12))
    
    # Add plots if they exist
    plot_files = [
        ('pca_variance.png', 'Explained Variance'),
        ('pca_feature_importance.png', 'Feature Importance'),
        ('pca_scatter_pc1_pc2.png', 'PCA Scatter Plot'),
        ('pca_loadings_pc1_pc2.png', 'Feature Loadings'),
        ('pca_biplot_pc1_pc2.png', 'PCA Biplot')
    ]

    # Maximum dimensions for plots in the PDF (in points, safe area within letter page margins)
    max_width = 450   # Safe width for letter size with margins
    max_height = 550  # Safe height to fit on page with headers/text

    for plot_file, plot_title in plot_files:
        plot_path = os.path.join(temp_path, plot_file)
        if os.path.exists(plot_path):
            # Get actual image dimensions to preserve aspect ratio
            with PILImage.open(plot_path) as img:
                img_width, img_height = img.size
                aspect_ratio = img_height / img_width

                # Calculate scaling to fit both width and height constraints
                # Scale by width
                scale_by_width = max_width / img_width if img_width > max_width else 1.0
                # Scale by height
                scale_by_height = max_height / img_height if img_height > max_height else 1.0

                # Use the more restrictive (smaller) scale factor to ensure it fits both constraints
                scale_factor = min(scale_by_width, scale_by_height, 1.0)

                display_width = img_width * scale_factor
                display_height = img_height * scale_factor

            # Keep title and image together on the same page
            plot_elements = [
                Paragraph(plot_title, styles['Heading2']),
                Spacer(1, 6),
                Image(plot_path, width=display_width, height=display_height)
            ]
            elements.append(KeepTogether(plot_elements))
            elements.append(Spacer(1, 12))
    
    # Feature Importance Table
    if _feature_importance is not None:
        elements.append(Paragraph("Top 10 Most Important Features", styles['Heading2']))
        
        importance_data = [['Rank', 'Feature', 'Importance Score']]
        for i, row in _feature_importance.head(10).iterrows():
            importance_data.append([str(row['Rank']), row['Feature'], f"{row['Importance']:.4f}"])
        
        importance_table = Table(importance_data)
        importance_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, 0), 12),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black)
        ]))
        elements.append(importance_table)
    
    # Build the PDF
    doc.build(elements)
    
    return report_path

# Legacy functions for backward compatibility
def perform_pca(*args, **kwargs):
    """Legacy function for backward compatibility."""
    return perform_enhanced_pca(*args, **kwargs)

def create_scatter_plot(pc_df, color_by, temp_path):
    """Legacy function for backward compatibility."""
    return create_enhanced_scatter_plot(pc_df, color_by, 1, 2, temp_path)

def create_loading_plot(loadings, feature_names, temp_path):
    """Legacy function for backward compatibility."""
    loadings_df = pd.DataFrame(
        data=loadings[:2, :].T,
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    return create_enhanced_loading_plot(loadings_df, [1, 2], temp_path)

def create_biplot(pc_scores, loadings, feature_names, temp_path):
    """Legacy function for backward compatibility."""
    return create_enhanced_biplot(pc_scores, loadings, feature_names, 1, 2, temp_path)

def generate_report(dataset_path, temp_path='temp/'):
    """Legacy function for backward compatibility."""
    return generate_enhanced_report(dataset_path, temp_path)