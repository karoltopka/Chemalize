import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
import seaborn as sns
import io

# Set the Seaborn style for better looking plots
sns.set(style="whitegrid")

def perform_clustering(df, method='kmeans', n_clusters=3, h_n_clusters=None, eps=0.5, min_samples=5, 
                     linkage_method='ward', scale_data=True, pca_visualization=True,
                     temp_path='temp/', index_column=None, label_density=10,
                     feature_selection='all', selected_features=None):
    """
    Perform clustering analysis on the given DataFrame.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        The input data.
    method : str, default='kmeans'
        Clustering method to use ('kmeans', 'dbscan', or 'hierarchical').
    n_clusters : int, default=3
        Number of clusters for KMeans clustering.
    h_n_clusters : int, default=None
        Number of clusters for Hierarchical clustering. If None, uses n_clusters.
    eps : float, default=0.5
        The maximum distance between samples for DBSCAN.
    min_samples : int, default=5
        The number of samples in a neighborhood for DBSCAN.
    linkage_method : str, default='ward'
        The linkage method for hierarchical clustering.
    scale_data : bool, default=True
        Whether to standardize the data before clustering.
    pca_visualization : bool, default=True
        Whether to use PCA for visualizing high-dimensional data.
    temp_path : str, default='temp/'
        Path to temporary directory for saving plots.
    index_column : str, default=None
        Column to use as index labels in the plots. If None, numeric indices are used.
    label_density : int, default=10
        Percentage of points to label (1-100). Lower values produce clearer plots.
        
    Returns:
    --------
    dict
        Dictionary containing clustering results and plot paths.
    """
    # Use h_n_clusters for hierarchical if provided, otherwise fall back to n_clusters
    if method == 'hierarchical' and h_n_clusters is not None:
        actual_n_clusters = h_n_clusters
    else:
        actual_n_clusters = n_clusters
    
    # Ensure the temporary directory exists
    os.makedirs(temp_path, exist_ok=True)
    
    # Store the index values if an index column is specified
    index_values = None
    if index_column and index_column in df.columns:
        # Make sure we handle numeric and non-numeric indices
        index_values = df[index_column].astype(str).values
    
    # Handle categorical variables and missing values
    df_numeric = df.select_dtypes(include=[np.number])
    
    # Filter columns based on feature selection
    if feature_selection == 'select' and selected_features and len(selected_features) > 0:
        # Only keep columns that are in selected_features and are numeric
        valid_features = [col for col in selected_features if col in df_numeric.columns]
        if len(valid_features) >= 2:  # Need at least 2 features for clustering
            df_clean = df_numeric[valid_features].dropna(axis=1)
        else:
            # Fall back to all numeric features if not enough valid features selected
            df_clean = df_numeric.dropna(axis=1)
    else:
        # Use all numeric columns
        df_clean = df_numeric.dropna(axis=1)
    
    if df_clean.shape[1] < 2:
        raise ValueError("Not enough numeric columns without missing values for clustering.")
    
    # Extract features (X)
    X = df_clean.values
    feature_names = df_clean.columns
    
    # Scale the data if requested
    if scale_data:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
    
    # Perform clustering based on the selected method
    if method == 'kmeans':
        # KMeans clustering
        model = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init=10)
        labels = model.fit_predict(X)
        centers = model.cluster_centers_
        
        # Calculate clustering metrics
        if actual_n_clusters > 1 and actual_n_clusters < len(X):
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
        else:
            silhouette = None
            calinski = None
        
        # Additional KMeans specific results
        inertia = model.inertia_
        
        # Determine optimal number of clusters (Elbow method)
        k_range = range(2, min(11, len(X)))
        inertia_values = []
        
        for k in k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertia_values.append(km.inertia_)
        
        # Create elbow plot
        plt.figure(figsize=(10, 6))
        plt.plot(k_range, inertia_values, 'o-')
        plt.xlabel('Number of Clusters')
        plt.ylabel('Inertia')
        plt.title('Elbow Method for Optimal k')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        elbow_plot_path = os.path.join(temp_path, 'clustering_elbow.png')
        plt.savefig(elbow_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
    elif method == 'dbscan':
        # DBSCAN clustering
        model = DBSCAN(eps=eps, min_samples=min_samples)
        labels = model.fit_predict(X)
        
        # Calculate clustering metrics
        # Note: DBSCAN doesn't require a specific number of clusters
        actual_n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        if actual_n_clusters > 1:
            silhouette = silhouette_score(X, labels) if len(set(labels)) > 1 else None
            calinski = calinski_harabasz_score(X, labels) if len(set(labels)) > 1 else None
        else:
            silhouette = None
            calinski = None
        
        # Count number of noise points
        n_noise = list(labels).count(-1)
        
        # Create eps distance plot
        from sklearn.neighbors import NearestNeighbors
        
        # Find the optimal eps value
        neighbors = NearestNeighbors(n_neighbors=min_samples)
        neighbors_fit = neighbors.fit(X)
        distances, _ = neighbors_fit.kneighbors(X)
        distances = np.sort(distances[:, min_samples-1])
        
        plt.figure(figsize=(10, 6))
        plt.plot(distances)
        plt.xlabel('Points')
        plt.ylabel(f'Distance to {min_samples}th nearest neighbor')
        plt.title('k-distance Graph')
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        eps_plot_path = os.path.join(temp_path, 'clustering_eps.png')
        plt.savefig(eps_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # No explicit centers for DBSCAN
        centers = None
        elbow_plot_path = None
        inertia = None
        
    elif method == 'hierarchical':
        # For hierarchical clustering, we'll use the same linkage matrix for both
        # the cluster assignment and dendrogram visualization to ensure consistency
        Z = linkage(X, method=linkage_method)
        
        # Get cluster labels from the linkage matrix with the same number of clusters
        # This ensures the dendrogram and cluster assignment use the same approach
        labels = fcluster(Z, actual_n_clusters, criterion='maxclust') - 1  # Subtract 1 to start from 0
        
        # For comparison, create the AgglomerativeClustering model too
        model = AgglomerativeClustering(n_clusters=actual_n_clusters, linkage=linkage_method)
        model_labels = model.fit_predict(X)
        
        # Calculate clustering metrics
        if actual_n_clusters > 1 and actual_n_clusters < len(X):
            silhouette = silhouette_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
        else:
            silhouette = None
            calinski = None
        
        # Create dendrogram with color threshold to match the number of clusters
        plt.figure(figsize=(12, 8))
        
        # Get the color threshold from the linkage matrix that gives us the desired clusters
        # For multi-cluster case (>2), we need to carefully set the threshold
        if actual_n_clusters > 1:
            # Get the distances at which clusters were merged
            distances = Z[:, 2]
            sorted_distances = np.sort(distances)
            
            # Find the gap between merge distances that creates our desired number of clusters
            # We need to find the (n_clusters-1)th largest distance
            idx = len(distances) - (actual_n_clusters - 1)
            if idx >= 0 and idx < len(distances):
                # Set threshold just below this distance to get exactly n_clusters
                color_threshold = sorted_distances[idx] - 0.0001
            else:
                # Fallback for edge cases
                color_threshold = 0.7 * np.max(distances)
        else:
            color_threshold = 0
        
        # Generate the dendrogram with the precise color threshold
        # Use index_values as labels if available
        leaf_labels = None
        if index_values is not None:
            leaf_labels = index_values
        
        dendrogram(
            Z,
            truncate_mode='level', 
            p=5,
            color_threshold=color_threshold,  # This ensures consistent coloring with clusters
            above_threshold_color='grey',
            labels=leaf_labels  # Use custom labels if available
        )
        plt.title('Hierarchical Clustering Dendrogram')
        if index_values is not None:
            plt.xlabel(f'Sample ({index_column})')
        else:
            plt.xlabel('Sample index')
        plt.ylabel('Distance')
        
        # Rotate x-axis labels if we're using custom labels
        if index_values is not None:
            plt.xticks(rotation=90, fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        dendrogram_path = os.path.join(temp_path, 'clustering_dendrogram.png')
        plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # No explicit centers for hierarchical clustering
        centers = None
        elbow_plot_path = None
        inertia = None
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # Create a DataFrame with cluster labels
    result_df = df_clean.copy()
    result_df['Cluster'] = labels
    
    # Add the index column to the results if specified
    if index_column and index_column in df.columns:
        # Check if the index column was included in df_clean (numeric columns)
        if index_column not in result_df.columns:
            # Create a row mapping from original df to result_df
            # We need to match rows since df_clean might not have all rows from df if there were any NaN values
            # This is a simple approach assuming the order of rows is preserved
            if len(df) == len(result_df):
                # If no rows were dropped, simply add the index column
                result_df[index_column] = df[index_column].values
            else:
                # If some rows were dropped, this requires more careful matching
                # For simplicity, we'll create a DataFrame with the index column and cluster assignments
                # that we can later join with the original df
                cluster_df = pd.DataFrame({
                    'Cluster': labels
                })
                # Store the clusters separately to later join with original df
                cluster_df.to_csv(os.path.join(temp_path, 'cluster_assignments.csv'), index=False)
                
                # Create a full result with all original columns plus clusters
                full_result_df = df.copy()
                # Add the cluster assignments
                full_result_df['Cluster'] = np.nan  # Initialize with NaN
                full_result_df.loc[result_df.index, 'Cluster'] = labels
                
                # Use this as our new result_df
                result_df = full_result_df
        
        # Move the index column to the first position
        cols = result_df.columns.tolist()
        if index_column in cols:
            cols.remove(index_column)
            cols = [index_column] + cols
            result_df = result_df[cols]
    
    # Save the clustering results
    result_df.to_csv(os.path.join(temp_path, 'clustering_results.csv'), index=False)
    
    # Visualize the clusters using PCA if requested
    if pca_visualization:
        # Apply PCA to reduce dimensionality to 2 components
        pca_model = PCA(n_components=2)
        X_pca = pca_model.fit_transform(X)
        
        # Create a scatter plot with cluster colors
        plt.figure(figsize=(10, 8))
        
        # For DBSCAN, handle noise points differently
        if method == 'dbscan':
            # Plot noise points as black
            noise_mask = (labels == -1)
            plt.scatter(X_pca[noise_mask, 0], X_pca[noise_mask, 1], 
                       c='black', marker='x', label='Noise')
            
            # Plot clustered points
            for cluster_id in range(actual_n_clusters):
                cluster_mask = (labels == cluster_id)
                plt.scatter(X_pca[cluster_mask, 0], X_pca[cluster_mask, 1], 
                          label=f'Cluster {cluster_id}', alpha=0.7)
            
            # Add index labels if requested - for all clusters
            if index_values is not None:
                # Calculate the number of points to label based on the density
                n_total_points = len(X_pca)
                n_labels = max(5, int(n_total_points * label_density / 100))  # Ensure at least 5 labels
                
                # Select points to label, ensuring a good distribution across clusters
                label_indices = []
                
                # First, handle noise points if they exist
                if np.sum(noise_mask) > 0:
                    noise_indices = np.where(noise_mask)[0]
                    n_noise_labels = max(1, int(np.sum(noise_mask) * label_density / 100))
                    noise_selected = np.random.choice(noise_indices, 
                                                    size=min(n_noise_labels, len(noise_indices)), 
                                                    replace=False)
                    label_indices.extend(noise_selected)
                
                # Then allocate labels to each cluster proportionally
                remaining_labels = n_labels - len(label_indices)
                for cluster_id in range(actual_n_clusters):
                    cluster_mask = (labels == cluster_id)
                    if np.sum(cluster_mask) > 0:
                        cluster_indices = np.where(cluster_mask)[0]
                        # Allocate labels proportionally to cluster size
                        n_cluster_labels = max(1, int(np.sum(cluster_mask) / (n_total_points - np.sum(noise_mask)) * remaining_labels))
                        cluster_selected = np.random.choice(cluster_indices, 
                                                          size=min(n_cluster_labels, len(cluster_indices)), 
                                                          replace=False)
                        label_indices.extend(cluster_selected)
                
                # Add annotations for selected points
                for i in label_indices:
                    plt.annotate(str(index_values[i]), 
                               (X_pca[i, 0], X_pca[i, 1]),
                               fontsize=8, alpha=0.8)
                
        else:
            # For KMeans and Hierarchical, use a fixed set of distinct colors for better visualization
            cmap = plt.cm.get_cmap('tab10', max(5, actual_n_clusters))
            
            # Create the scatter plot with these colors
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.7)
            
            # Add index labels if requested
            if index_values is not None:
                # Calculate the number of points to label based on the density
                n_total_points = len(X_pca)
                n_labels = max(5, int(n_total_points * label_density / 100))  # Ensure at least 5 labels
                
                # Select points to label, ensuring a good distribution across clusters
                label_indices = []
                
                # Allocate labels to each cluster proportionally
                for cluster_id in range(actual_n_clusters):
                    cluster_mask = (labels == cluster_id)
                    if np.sum(cluster_mask) > 0:
                        cluster_indices = np.where(cluster_mask)[0]
                        # Allocate labels proportionally to cluster size
                        n_cluster_labels = max(1, int(np.sum(cluster_mask) / n_total_points * n_labels))
                        cluster_selected = np.random.choice(cluster_indices, 
                                                          size=min(n_cluster_labels, len(cluster_indices)), 
                                                          replace=False)
                        label_indices.extend(cluster_selected)
                
                # Add annotations for selected points
                for i in label_indices:
                    plt.annotate(str(index_values[i]), 
                               (X_pca[i, 0], X_pca[i, 1]),
                               fontsize=8, alpha=0.8)
            
            # Plot cluster centers for KMeans
            if method == 'kmeans' and centers is not None:
                centers_pca = pca_model.transform(centers)
                plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', marker='X', 
                          s=100, label='Centroids')
        
        plt.xlabel('Principal Component 1')
        plt.ylabel('Principal Component 2')
        plt.title(f'Cluster Visualization with PCA ({method.capitalize()})')
        
        # Create a proper colorbar for consistent colors
        if method != 'dbscan':
            cbar = plt.colorbar(scatter, label='Cluster')
            cbar.set_ticks(np.arange(actual_n_clusters))
            cbar.set_ticklabels([f'Cluster {i}' for i in range(actual_n_clusters)])
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        cluster_plot_path = os.path.join(temp_path, 'clustering_pca.png')
        plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        cluster_plot_path = None
    
    # Create a cluster profile plot showing feature means for each cluster
    if method != 'dbscan' or actual_n_clusters > 0:
        # Group by cluster and calculate mean for each feature - only for numeric columns
        # Exclude the index column and any other non-numeric columns
        numeric_cols = result_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Make sure 'Cluster' is included
        if 'Cluster' not in numeric_cols:
            numeric_cols.append('Cluster')
        
        # If index_column exists and is not numeric, remove it from the list
        if index_column in numeric_cols and index_column != 'Cluster':
            numeric_cols.remove(index_column)
        
        # Group by cluster and calculate mean only for numeric columns
        cluster_profiles = result_df[numeric_cols].groupby('Cluster').mean()
        
        # Create a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_profiles, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5)
        plt.title('Cluster Profiles (Feature Means by Cluster)')
        plt.tight_layout()
        
        # Save the plot
        profile_plot_path = os.path.join(temp_path, 'clustering_profiles.png')
        plt.savefig(profile_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        profile_plot_path = None
    
    # Create cluster size distribution plot
    plt.figure(figsize=(10, 6))
    cluster_sizes = result_df['Cluster'].value_counts().sort_index()
    
    # For DBSCAN, handle noise points differently
    if method == 'dbscan' and -1 in cluster_sizes.index:
        noise_size = cluster_sizes[-1]
        cluster_sizes = cluster_sizes.drop(-1)
        
        # Create the bar chart without noise points
        cluster_sizes.plot(kind='bar', alpha=0.7)
        
        # Add noise points as a different colored bar
        if not cluster_sizes.empty:
            plt.bar(-1, noise_size, color='black', alpha=0.7)
            plt.xticks(range(-1, len(cluster_sizes)), ['Noise'] + [f'Cluster {i}' for i in cluster_sizes.index])
        else:
            plt.bar(0, noise_size, color='black', alpha=0.7)
            plt.xticks([0], ['Noise'])
    else:
        # Use same colormap as in the PCA visualization for consistency
        cmap = plt.cm.get_cmap('tab10', max(5, actual_n_clusters))
        
        # Create individual bars with colors matching the clusters
        for i, (cluster, size) in enumerate(cluster_sizes.items()):
            plt.bar(i, size, color=cmap(cluster), alpha=0.7)
        
        plt.xticks(range(len(cluster_sizes)), [f'Cluster {i}' for i in cluster_sizes.index])
    
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    size_plot_path = os.path.join(temp_path, 'clustering_sizes.png')
    plt.savefig(size_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Prepare the results
    results = {
        'method': method,
        'n_clusters': actual_n_clusters,
        'silhouette': silhouette,
        'calinski': calinski,
        'cluster_plot': cluster_plot_path,
        'profile_plot': profile_plot_path,
        'size_plot': size_plot_path,
        'index_column': index_column,  # Return the index column used
    }
    
    # Add method-specific results
    if method == 'kmeans':
        results.update({
            'inertia': inertia,
            'elbow_plot': elbow_plot_path,
        })
    elif method == 'dbscan':
        results.update({
            'eps': eps,
            'min_samples': min_samples,
            'n_noise': n_noise,
            'eps_plot': eps_plot_path,
        })
    elif method == 'hierarchical':
        results.update({
            'h_n_clusters': actual_n_clusters,  # Store as h_n_clusters for hierarchical
            'linkage_method': linkage_method,
            'dendrogram': dendrogram_path,
        })
    
    return results

def generate_results_file(dataset_path, temp_path='temp/'):
    """Generate a CSV file with the clustering results."""
    # The file was already saved during analysis
    return os.path.join(temp_path, 'clustering_results.csv')

def generate_report(dataset_path, method='kmeans', temp_path='temp/', index_column=None):
    """Generate a PDF report summarizing the clustering results."""
    # Read the dataset
    df = pd.read_csv(dataset_path) if dataset_path.endswith('.csv') else \
         pd.read_excel(dataset_path)
    
    # Read the clustering results
    results_df = pd.read_csv(os.path.join(temp_path, 'clustering_results.csv'))
    
    # Create a PDF report
    report_path = os.path.join(temp_path, 'clustering_report.pdf')
    doc = SimpleDocTemplate(report_path, pagesize=letter)
    styles = getSampleStyleSheet()
    elements = []
    
    # Title
    elements.append(Paragraph(f"{method.capitalize()} Clustering Report", styles['Title']))
    elements.append(Spacer(1, 12))
    
    # Dataset Information
    elements.append(Paragraph("Dataset Information", styles['Heading1']))
    elements.append(Paragraph(f"Number of observations: {len(df)}", styles['Normal']))
    elements.append(Paragraph(f"Number of variables: {len(df.columns)}", styles['Normal']))
    
    # Add index column info if specified
    if index_column:
        elements.append(Paragraph(f"Index Column: {index_column}", styles['Normal']))
    
    elements.append(Spacer(1, 12))
    
    # Clustering Results
    elements.append(Paragraph("Clustering Results", styles['Heading1']))
    
    # Cluster counts
    elements.append(Paragraph("Cluster Distribution", styles['Heading2']))
    cluster_counts = results_df['Cluster'].value_counts().sort_index()
    
    # Format cluster counts as a table
    data = [['Cluster', 'Count', 'Percentage']]
    for cluster, count in cluster_counts.items():
        percentage = (count / len(results_df)) * 100
        data.append([f"Cluster {cluster}", str(count), f"{percentage:.2f}%"])
    
    # Create the table
    cluster_table = Table(data)
    cluster_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    
    elements.append(cluster_table)
    elements.append(Spacer(1, 12))
    
    # Add the cluster visualization plot if it exists
    cluster_plot_path = os.path.join(temp_path, 'clustering_pca.png')
    if os.path.exists(cluster_plot_path):
        elements.append(Paragraph("Cluster Visualization", styles['Heading2']))
        elements.append(Image(cluster_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the cluster profiles plot if it exists
    profile_plot_path = os.path.join(temp_path, 'clustering_profiles.png')
    if os.path.exists(profile_plot_path):
        elements.append(Paragraph("Cluster Profiles", styles['Heading2']))
        elements.append(Image(profile_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add the cluster sizes plot if it exists
    size_plot_path = os.path.join(temp_path, 'clustering_sizes.png')
    if os.path.exists(size_plot_path):
        elements.append(Paragraph("Cluster Sizes", styles['Heading2']))
        elements.append(Image(size_plot_path, width=400, height=300))
        elements.append(Spacer(1, 12))
    
    # Add method-specific plots
    if method == 'kmeans':
        elbow_plot_path = os.path.join(temp_path, 'clustering_elbow.png')
        if os.path.exists(elbow_plot_path):
            elements.append(Paragraph("Elbow Method for Optimal k", styles['Heading2']))
            elements.append(Image(elbow_plot_path, width=400, height=300))
            elements.append(Spacer(1, 12))
    elif method == 'dbscan':
        eps_plot_path = os.path.join(temp_path, 'clustering_eps.png')
        if os.path.exists(eps_plot_path):
            elements.append(Paragraph("k-distance Graph for Optimal Eps", styles['Heading2']))
            elements.append(Image(eps_plot_path, width=400, height=300))
            elements.append(Spacer(1, 12))
    elif method == 'hierarchical':
        dendrogram_path = os.path.join(temp_path, 'clustering_dendrogram.png')
        if os.path.exists(dendrogram_path):
            elements.append(Paragraph("Hierarchical Clustering Dendrogram", styles['Heading2']))
            elements.append(Image(dendrogram_path, width=400, height=300))
            elements.append(Spacer(1, 12))
    
    # Build the PDF
    doc.build(elements)
    
    return report_path