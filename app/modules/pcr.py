import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import seaborn as sns
import math
import io

# Set the Seaborn style for better looking plots
sns.set(style="whitegrid")

def perform_pcr(df, target_var, n_components=2, test_size=0.2, scale_data=True, 
               optimize_components=False, compare_with_linear=True, 
               show_variance=True, show_pred_actual=True, show_residuals=True, 
               temp_path='temp/'):
    """
    Perform Principal Component Regression on the given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input data.
    target_var : str
        Name of the target variable.
    n_components : int, default=2
        Number of principal components to use in regression.
    test_size : float, default=0.2
        Proportion of data to use for testing.
    scale_data : bool, default=True
        Whether to standardize the data before PCA.
    optimize_components : bool, default=False
        Whether to optimize the number of components using cross-validation.
    compare_with_linear : bool, default=True
        Whether to compare PCR with standard linear regression.
    show_variance : bool, default=True
        Whether to create a plot showing explained variance.
    show_pred_actual : bool, default=True
        Whether to create plots comparing predicted and actual values.
    show_residuals : bool, default=True
        Whether to create residual plots.
    temp_path : str, default='temp/'
        Path to temporary directory for saving plots.
        
    Returns:
    --------
    dict
        Dictionary containing PCR results and plot paths.
    """
    # Ensure the temporary directory exists
    os.makedirs(temp_path, exist_ok=True)
    
    # Handle target variable
    if target_var not in df.columns:
        raise ValueError(f"Target variable '{target_var}' not found in the dataset.")
    
    # Split features and target
    y = df[target_var].values
    
    # Handle categorical variables and missing values
    df_numeric = df.select_dtypes(include=[np.number])
    df_numeric = df_numeric.drop(columns=[target_var], errors='ignore')
    
    # Drop columns with any missing values for this analysis
    df_clean = df_numeric.dropna(axis=1)
    
    if df_clean.shape[1] < 1:
        raise ValueError("Not enough numeric columns without missing values for PCR.")
    
    # Extract features (X)
    X = df_clean.values
    feature_names = df_clean.columns
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    # Scale the data if requested
    if scale_data:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    
    # Perform standard linear regression for comparison if requested
    lr_results = {}
    if compare_with_linear:
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Predictions
        y_train_pred_lr = lr.predict(X_train)
        y_test_pred_lr = lr.predict(X_test)
        
        # Metrics
        lr_train_r2 = r2_score(y_train, y_train_pred_lr)
        lr_test_r2 = r2_score(y_test, y_test_pred_lr)
        lr_train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred_lr))
        lr_test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred_lr))
        lr_test_mae = mean_absolute_error(y_test, y_test_pred_lr)
        
        lr_results = {
            'lr_train_r2': lr_train_r2,
            'lr_test_r2': lr_test_r2,
            'lr_train_rmse': lr_train_rmse,
            'lr_test_rmse': lr_test_rmse,
            'lr_test_mae': lr_test_mae
        }
    
    # Initialize the PCA model
    pca_model = PCA(n_components=min(n_components, min(X_train.shape)))
    
    # Fit the PCA model on the training data
    pca_model.fit(X_train)
    
    # Transform the data
    X_train_pca = pca_model.transform(X_train)
    X_test_pca = pca_model.transform(X_test)
    
    # Optimize the number of components if requested
    if optimize_components:
        # Try different numbers of components
        max_components = min(X_train.shape)
        n_components_range = range(1, min(max_components + 1, 11))  # Up to 10 components or max available
        cv_scores = []
        
        for n in n_components_range:
            # Use only the first n components
            X_train_pca_n = X_train_pca[:, :n]
            
            # Create and fit a linear regression model
            lr = LinearRegression()
            # Perform cross-validation
            scores = cross_val_score(lr, X_train_pca_n, y_train, cv=5, scoring='r2')
            cv_scores.append(np.mean(scores))
        
        # Find the optimal number of components
        optimal_n = n_components_range[np.argmax(cv_scores)]
        
        # Update the number of components
        n_components = optimal_n
        
        # Update the transformed data
        X_train_pca = X_train_pca[:, :n_components]
        X_test_pca = X_test_pca[:, :n_components]
        
        # Create the optimization plot
        plt.figure(figsize=(10, 6))
        plt.plot(n_components_range, cv_scores, 'o-')
        plt.xlabel('Number of Components')
        plt.ylabel('Cross-Validation R² Score')
        plt.title('PCA Component Optimization')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        optimization_plot_path = os.path.join(temp_path, 'pcr_optimization.png')
        plt.savefig(optimization_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        optimization_plot_path = None
        X_train_pca = X_train_pca[:, :n_components]
        X_test_pca = X_test_pca[:, :n_components]
    
    # Fit linear regression on the PCA components
    pcr = LinearRegression()
    pcr.fit(X_train_pca, y_train)
    
    # Make predictions
    y_train_pred = pcr.predict(X_train_pca)
    y_test_pred = pcr.predict(X_test_pca)
    
    # Calculate metrics
    pcr_train_r2 = r2_score(y_train, y_train_pred)
    pcr_test_r2 = r2_score(y_test, y_test_pred)
    pcr_train_rmse = math.sqrt(mean_squared_error(y_train, y_train_pred))
    pcr_test_rmse = math.sqrt(mean_squared_error(y_test, y_test_pred))
    pcr_test_mae = mean_absolute_error(y_test, y_test_pred)
    
    # Calculate the total variance explained by the selected components
    total_variance_explained = np.sum(pca_model.explained_variance_ratio_[:n_components]) * 100
    
    # Create plots
    plot_paths = {}
    
    # Explained variance plot
    if show_variance:
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
        
        # Mark the selected number of components
        plt.axvline(x=n_components, color='green', linestyle='--', 
                   label=f'Selected Components: {n_components}')
        
        plt.xlabel('Principal Component')
        plt.ylabel('Explained Variance (%)')
        plt.title('Explained Variance by Principal Components')
        plt.xticks(range(1, len(pca_model.explained_variance_ratio_) + 1))
        plt.legend(loc='best')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        variance_plot_path = os.path.join(temp_path, 'pcr_variance.png')
        plt.savefig(variance_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['pcr_variance_plot'] = variance_plot_path
    
    # Predicted vs Actual plot
    if show_pred_actual:
        plt.figure(figsize=(12, 10))
        
        # Determine plot limits
        min_val = min(np.min(y_test), np.min(y_test_pred))
        max_val = max(np.max(y_test), np.max(y_test_pred))
        
        # Add some margin
        margin = (max_val - min_val) * 0.1
        min_val -= margin
        max_val += margin
        
        # Plot the perfect prediction line
        plt.plot([min_val, max_val], [min_val, max_val], 'k--', label='Perfect Prediction')
        
        # Plot PCR predictions
        plt.scatter(y_test, y_test_pred, alpha=0.7, label='PCR')
        
        # Plot Linear Regression predictions if available
        if compare_with_linear:
            plt.scatter(y_test, y_test_pred_lr, alpha=0.7, marker='x', label='Linear Regression')
        
        plt.xlabel(f'Actual {target_var}')
        plt.ylabel(f'Predicted {target_var}')
        plt.title('Predicted vs Actual Values')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        pred_actual_plot_path = os.path.join(temp_path, 'pcr_pred_actual.png')
        plt.savefig(pred_actual_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['pcr_pred_actual_plot'] = pred_actual_plot_path
    
    # Residuals plot
    if show_residuals:
        plt.figure(figsize=(12, 6))
        
        # Calculate residuals
        residuals = y_test - y_test_pred
        
        # Plot residuals
        plt.scatter(y_test_pred, residuals, alpha=0.7)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel(f'Predicted {target_var}')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        residuals_plot_path = os.path.join(temp_path, 'pcr_residuals.png')
        plt.savefig(residuals_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        plot_paths['pcr_residuals_plot'] = residuals_plot_path
    
    # Save all predictions and results
    results_df = pd.DataFrame({
        'Actual': y_test,
        'Predicted_PCR': y_test_pred
    })
    
    if compare_with_linear:
        results_df['Predicted_LR'] = y_test_pred_lr
    
    results_df.to_csv(os.path.join(temp_path, 'pcr_predictions.csv'), index=False)
    
    # Save PCR coefficients in the original feature space
    # Transform PCR coefficients back to the original feature space
    pcr_coef_pca = pcr.coef_
    pcr_coef_orig = np.zeros(pca_model.components_.shape[1])
    
    for i in range(n_components):
        pcr_coef_orig += pcr_coef_pca[i] * pca_model.components_[i, :]
    
    coef_df = pd.DataFrame({
        'Feature': feature_names,
        'PCR_Coefficient': pcr_coef_orig
    })
    
    if compare_with_linear:
        coef_df['LR_Coefficient'] = lr.coef_
    
    coef_df.to_csv(os.path.join(temp_path, 'pcr_coefficients.csv'), index=False)
    
    # Combine all results
    results = {
        'pcr_train_r2': pcr_train_r2,
        'pcr_test_r2': pcr_test_r2,
        'pcr_train_rmse': pcr_train_rmse,
        'pcr_test_rmse': pcr_test_rmse,
        'pcr_test_mae': pcr_test_mae,
        'n_components': n_components,
        'total_variance_explained': total_variance_explained
    }
    
    if optimize_components:
        results['optimization_plot'] = optimization_plot_path
    
    # Add linear regression results if available
    if compare_with_linear:
        results.update(lr_results)
    
    # Add plot paths
    results.update(plot_paths)
    
    return results

def generate_predictions_file(dataset_path, target_var, temp_path='temp/'):
    """Generate a CSV file with the PCR predictions."""
    # The file was already saved during analysis
    return os.path.join(temp_path, 'pcr_predictions.csv')

def generate_model_file(dataset_path, target_var, temp_path='temp/'):
    """Generate a CSV file with the PCR coefficients."""
    # The file was already saved during analysis
    return os.path.join(temp_path, 'pcr_coefficients.csv')

def generate_report(dataset_path, target_var, temp_path='temp/'):
    """Generate a PDF report summarizing the PCR results."""
    # Read the dataset
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else \
         pd.read_excel(dataset_path)
    
    # Read the PCR results
    predictions_df = pd.read_csv(os.path.join(temp_path, 'pcr_predictions.csv'))
    coefficients_df = pd.read_csv(os.path.join(temp_path, 'pcr_coefficients.csv'))
    
    # Create a PDF report
    report_path = os.path.join(temp_path, 'pcr_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph("Principal Component Regression Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Dataset Information
    elements.append(Paragraph("Dataset Information", styles['Heading1']))
    elements.append(Paragraph(f"Number of observations: {len(df)}", styles['Normal']))
    elements.append(Paragraph(f"Number of variables: {len(df.columns)}", styles['Normal']))
    elements.append(Paragraph(f"Target variable: {target_var}", styles['Normal']))
    elements.append(Spacer(1, 12))
    
    # PCR Results
    elements.append(Paragraph("PCR Results", styles['Heading1']))
    
    # Add the variance plot if it exists
    variance_plot_path = os.path.join(temp_path, 'pcr_variance.png')
    if os.path.exists(variance_plot_path):
        elements.append(Paragraph("Explained Variance", styles['Heading2']))
        elements.append(Image(variance_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the predicted vs actual plot if it exists
    pred_actual_plot_path = os.path.join(temp_path, 'pcr_pred_actual.png')
    if os.path.exists(pred_actual_plot_path):
        elements.append(Paragraph("Predicted vs Actual Values", styles['Heading2']))
        elements.append(Image(pred_actual_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the residuals plot if it exists
    residuals_plot_path = os.path.join(temp_path, 'pcr_residuals.png')
    if os.path.exists(residuals_plot_path):
        elements.append(Paragraph("Residuals Plot", styles['Heading2']))
        elements.append(Image(residuals_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the optimization plot if it exists
    optimization_plot_path = os.path.join(temp_path, 'pcr_optimization.png')
    if os.path.exists(optimization_plot_path):
        elements.append(Paragraph("Component Optimization", styles['Heading2']))
        elements.append(Image(optimization_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Model Coefficients
    elements.append(Paragraph("Model Coefficients", styles['Heading2']))
    
    # Convert coefficients to a table
    data = [['Feature', 'PCR Coefficient']]
    if 'LR_Coefficient' in coefficients_df.columns:
        data[0].append('LR Coefficient')
    
    for _, row in coefficients_df.iterrows():
        data_row = [row['Feature'], f"{row['PCR_Coefficient']:.4f}"]
        if 'LR_Coefficient' in coefficients_df.columns:
            data_row.append(f"{row['LR_Coefficient']:.4f}")
        data.append(data_row)
    
    # Create the table
    coef_table = Table(data)
    coef_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(coef_table)
    elements.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path