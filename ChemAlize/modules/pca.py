import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image
from reportlab.lib.styles import getSampleStyleSheet
import seaborn as sns
import io

# Set the Seaborn style for better looking plots
sns.set(style="whitegrid")

def perform_pca(df, n_components=2, scale_data=True, show_variance=True, 
               show_scatter=True, show_loading=True, show_biplot=True, 
               color_by=None, temp_path='temp/'):
    """
    Perform Principal Component Analysis on the given DataFrame.
    
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
        Whether to create a scatter plot of the first two principal components.
    show_loading : bool, default=True
        Whether to create a loading plot showing feature contributions.
    show_biplot : bool, default=True
        Whether to create a biplot showing samples and feature vectors.
    color_by : str, default=None
        Column name to use for coloring points in the scatter plot.
    temp_path : str, default='temp/'
        Path to temporary directory for saving plots.
        
    Returns:
    --------
    dict
        Dictionary containing PCA results and plot paths.
    """
    # Ensure the temporary directory exists
    os.makedirs(temp_path, exist_ok=True)
    
    # Handle categorical variables and missing values
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Drop columns with any missing values for this analysis
    df_clean = df_numeric.dropna(axis=1)
    
    if df_clean.shape[1] < 2:
        raise ValueError("Not enough numeric columns without missing values for PCA.")
    
    # Extract features (X)
    X = df_clean.values
    
    # Scale the data if requested
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Initialize the PCA model
    pca_model = PCA(n_components=min(n_components, min(X.shape)))
    
    # Fit the model and transform the data
    principal_components = pca_model.fit_transform(X)
    
    # Create a DataFrame with principal components
    pc_df = pd.DataFrame(
        data=principal_components,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)]
    )
    
    # Add color column if specified and exists in the original DataFrame
    if color_by and color_by in df.columns:
        pc_df[color_by] = df[color_by].reset_index(drop=True)
    
    # Get the loadings
    loadings = pca_model.components_
    
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
        ]
    }
    
    # Save the components and loadings to files
    pc_df.to_csv(os.path.join(temp_path, 'pca_components.csv'), index=False)
    
    # Create a loadings DataFrame
    loadings_df = pd.DataFrame(
        data=loadings.T,
        columns=[f'PC{i+1}' for i in range(pca_model.n_components_)],
        index=df_clean.columns
    )
    loadings_df.to_csv(os.path.join(temp_path, 'pca_loadings.csv'))
    
    # Generate plots
    if show_variance:
        result['variance_plot'] = create_variance_plot(pca_model, temp_path)
    
    if show_scatter and pca_model.n_components >= 2:
        result['scatter_plot'] = create_scatter_plot(pc_df, color_by, temp_path)
    
    if show_loading and pca_model.n_components >= 2:
        result['loadings_plot'] = create_loading_plot(loadings, df_clean.columns, temp_path)
    
    if show_biplot and pca_model.n_components >= 2:
        result['biplot'] = create_biplot(principal_components, loadings, df_clean.columns, temp_path)
    
    return result

def create_variance_plot(pca_model, temp_path):
    """Create and save a plot showing explained variance by principal components."""
    plt.figure(figsize=(10, 6))
    
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
    
    # Save the plot
    plot_path = os.path.join(temp_path, 'pca_variance.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_scatter_plot(pc_df, color_by, temp_path):
    """Create and save a scatter plot of the first two principal components."""
    plt.figure(figsize=(10, 8))
    
    if color_by and color_by in pc_df.columns:
        # If color column is categorical
        if pc_df[color_by].dtype == 'object' or pd.api.types.is_categorical_dtype(pc_df[color_by]):
            categories = pc_df[color_by].unique()
            for category in categories:
                subset = pc_df[pc_df[color_by] == category]
                plt.scatter(subset['PC1'], subset['PC2'], label=category, alpha=0.7)
            plt.legend(title=color_by)
        else:
            # If color column is numeric
            scatter = plt.scatter(pc_df['PC1'], pc_df['PC2'], c=pc_df[color_by], cmap='viridis', alpha=0.7)
            plt.colorbar(scatter, label=color_by)
    else:
        plt.scatter(pc_df['PC1'], pc_df['PC2'], alpha=0.7)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Scatter Plot')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(temp_path, 'pca_scatter.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_loading_plot(loadings, feature_names, temp_path):
    """Create and save a plot showing feature loadings for the first two principal components."""
    plt.figure(figsize=(12, 10))
    
    # Create a heatmap of loadings
    loadings_df = pd.DataFrame(
        data=loadings[:2, :].T,  # Take only first two components
        columns=['PC1', 'PC2'],
        index=feature_names
    )
    
    # Sort by absolute loading values on PC1
    loadings_df = loadings_df.reindex(loadings_df['PC1'].abs().sort_values(ascending=False).index)
    
    sns.heatmap(loadings_df, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5)
    plt.title('Feature Loadings for PC1 and PC2')
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(temp_path, 'pca_loadings.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def create_biplot(pc_scores, loadings, feature_names, temp_path):
    """Create and save a biplot showing both samples and feature vectors."""
    plt.figure(figsize=(12, 10))
    
    # Plot the scores (samples)
    plt.scatter(pc_scores[:, 0], pc_scores[:, 1], alpha=0.7)
    
    # Plot feature vectors
    for i, (feature, loading_x, loading_y) in enumerate(zip(feature_names, loadings[0, :], loadings[1, :])):
        plt.arrow(
            0, 0,  # Start at origin
            loading_x * max(abs(pc_scores[:, 0])) * 0.8,  # Scale to match scores
            loading_y * max(abs(pc_scores[:, 1])) * 0.8,
            head_width=0.1, head_length=0.1, fc='red', ec='red'
        )
        plt.text(
            loading_x * max(abs(pc_scores[:, 0])) * 0.85,
            loading_y * max(abs(pc_scores[:, 1])) * 0.85,
            feature, color='red', ha='center', va='center'
        )
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('PCA Biplot')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    plt.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    plt.tight_layout()
    
    # Save the plot
    plot_path = os.path.join(temp_path, 'pca_biplot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return plot_path

def generate_components_file(dataset_path, temp_path='temp/'):
    """Generate a CSV file with the principal component scores."""
    # The file was already saved during analysis
    return os.path.join(temp_path, 'pca_components.csv')

def generate_loadings_file(dataset_path, temp_path='temp/'):
    """Generate a CSV file with the loadings."""
    # The file was already saved during analysis
    return os.path.join(temp_path, 'pca_loadings.csv')

def generate_report(dataset_path, temp_path='temp/'):
    """Generate a PDF report summarizing the PCA results."""
    # Read the dataset
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else \
         pd.read_excel(dataset_path)
    
    # Read the PCA results
    pc_df = pd.read_csv(os.path.join(temp_path, 'pca_components.csv'))
    loadings_df = pd.read_csv(os.path.join(temp_path, 'pca_loadings.csv'), index_col=0)
    
    # Create a PDF report
    report_path = os.path.join(temp_path, 'pca_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph("Principal Component Analysis Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Dataset Information
    elements.append(Paragraph("Dataset Information", styles['Heading1']))
    elements.append(Paragraph(f"Number of observations: {len(df)}", styles['Normal']))
    elements.append(Paragraph(f"Number of variables: {len(df.columns)}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # PCA Results
    elements.append(Paragraph("PCA Results", styles['Heading1']))
    elements.append(Paragraph(f"Number of principal components: {len(pc_df.columns)}", styles['Normal']))
    
    # Add the variance plot if it exists
    variance_plot_path = os.path.join(temp_path, 'pca_variance.png')
    if os.path.exists(variance_plot_path):
        elements.append(Paragraph("Explained Variance", styles['Heading2']))
        elements.append(Image(variance_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the scatter plot if it exists
    scatter_plot_path = os.path.join(temp_path, 'pca_scatter.png')
    if os.path.exists(scatter_plot_path):
        elements.append(Paragraph("PCA Scatter Plot", styles['Heading2']))
        elements.append(Image(scatter_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the loadings plot if it exists
    loadings_plot_path = os.path.join(temp_path, 'pca_loadings.png')
    if os.path.exists(loadings_plot_path):
        elements.append(Paragraph("Feature Loadings", styles['Heading2']))
        elements.append(Image(loadings_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the biplot if it exists
    biplot_path = os.path.join(temp_path, 'pca_biplot.png')
    if os.path.exists(biplot_path):
        elements.append(Paragraph("PCA Biplot", styles['Heading2']))
        elements.append(Image(biplot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path