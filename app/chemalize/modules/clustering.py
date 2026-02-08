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
from app.utils.watermark import add_watermark_matplotlib_after_plot

# Set the Seaborn style for better looking plots
sns.set(style="whitegrid")


import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

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
        add_watermark_matplotlib_after_plot(plt.gcf())
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
        add_watermark_matplotlib_after_plot(plt.gcf())
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
        
        color_threshold = max(color_threshold, 0)
        
        # Lift ties and log-scale distances for clearer lower branches
        Z_plot = _prepare_linkage_for_plot(Z)
        plot_color_threshold = color_threshold
        y_label = 'Distance'
        if Z_plot.size > 0:
            if plot_color_threshold > 0:
                plot_color_threshold = np.log1p(plot_color_threshold)
            y_label = 'Distance (log scaled)'

        # Set custom color palette for dendrogram using tab20 colors
        from scipy.cluster.hierarchy import set_link_color_palette
        tab20_colors = plt.cm.tab20.colors  # Get tab20 colors
        # Convert RGB tuples to hex strings for scipy
        tab20_hex = ['#%02x%02x%02x' % (int(r*255), int(g*255), int(b*255))
                     for r, g, b in tab20_colors[:actual_n_clusters]]
        set_link_color_palette(tab20_hex)

        # Generate the dendrogram with the precise color threshold
        # Use index_values as labels if available
        leaf_labels = None
        if index_values is not None:
            leaf_labels = index_values

        dendrogram_kwargs = {
            'color_threshold': plot_color_threshold,  # This ensures consistent coloring with clusters
            'above_threshold_color': 'grey',
            'labels': leaf_labels  # Use custom labels if available
        }
        max_full_leaves = 200
        if len(X) > max_full_leaves:
            dendrogram_kwargs.update({
                'truncate_mode': 'level',
                'p': min(30, len(X)),
            })
        dendrogram(Z_plot, **dendrogram_kwargs)

        # Reset color palette to default after drawing
        set_link_color_palette(None)
        plt.title('Hierarchical Clustering Dendrogram')
        if index_values is not None:
            plt.xlabel(f'Sample ({index_column})')
        else:
            plt.xlabel('Sample index')
        plt.ylabel(y_label)
        
        # Rotate x-axis labels if we're using custom labels
        if index_values is not None:
            plt.xticks(rotation=90, fontsize=8)
        
        plt.tight_layout()
        
        # Save the plot
        add_watermark_matplotlib_after_plot(plt.gcf())
        dendrogram_path = os.path.join(temp_path, 'clustering_dendrogram.png')
        plt.savefig(dendrogram_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # No explicit centers for hierarchical clustering
        centers = None
        elbow_plot_path = None
        inertia = None
        
    else:
        raise ValueError(f"Unsupported clustering method: {method}")
    
    # ===== POPRAWIONA CZĘŚĆ: Tworzenie result_df z index_column =====
    # Create a DataFrame with cluster labels based on df_clean
    result_df = df_clean.copy()
    result_df['Cluster'] = labels
    
    # Create full result DataFrame with all original columns
    if index_column and index_column in df.columns:
        # Start with the original DataFrame
        full_result_df = df.copy()
        
        # Add Cluster column (initially NaN for all rows)
        full_result_df['Cluster'] = np.nan
        
        # Assign cluster labels only to rows that were used in clustering
        # Match by index position
        full_result_df.loc[result_df.index, 'Cluster'] = labels
        
        # Move index_column to the first position
        cols = full_result_df.columns.tolist()
        if index_column in cols:
            cols.remove(index_column)
            cols = [index_column] + cols
            full_result_df = full_result_df[cols]
        
        # Use this as our result_df
        result_df = full_result_df
    else:
        # If no index_column specified, just use the df_clean based result
        pass
    # ==============================================================
    
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
            # For KMeans and Hierarchical, use a FIXED colormap so colors stay consistent
            # Get discrete colors from tab20 (use only first N colors for N clusters)
            from matplotlib.colors import ListedColormap
            tab20_colors = plt.cm.tab20.colors  # Get all 20 colors from tab20
            cluster_colors = [tab20_colors[i % 20] for i in range(actual_n_clusters)]  # Use only first N
            cmap = ListedColormap(cluster_colors)  # Create discrete colormap

            # Create the scatter plot with discrete colors
            scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap=cmap, alpha=0.7,
                                vmin=-0.5, vmax=actual_n_clusters-0.5)
            
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
        
        # Create a proper colorbar showing only actual clusters
        if method != 'dbscan':
            cbar = plt.colorbar(scatter, label='Cluster', ticks=np.arange(actual_n_clusters))
            cbar.set_ticklabels([f'Cluster {i}' for i in range(actual_n_clusters)])
        
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        
        # Save the plot
        add_watermark_matplotlib_after_plot(plt.gcf())
        cluster_plot_path = os.path.join(temp_path, 'clustering_pca.png')
        plt.savefig(cluster_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        cluster_plot_path = None
    
    # Create a cluster profile plot showing feature means for each cluster
    if method != 'dbscan' or actual_n_clusters > 0:
        # For profile plot, use only the rows that were actually clustered
        # Create a temporary dataframe with just the clustered data
        profile_df = df_clean.copy()
        profile_df['Cluster'] = labels
        
        # Group by cluster and calculate mean for each feature
        cluster_profiles = profile_df.groupby('Cluster').mean()
        
        # Create a heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(cluster_profiles, cmap='coolwarm', center=0, annot=True, fmt=".2f", linewidths=0.5)
        plt.title('Cluster Profiles (Feature Means by Cluster)')
        plt.tight_layout()
        
        # Save the plot
        add_watermark_matplotlib_after_plot(plt.gcf())
        profile_plot_path = os.path.join(temp_path, 'clustering_profiles.png')
        plt.savefig(profile_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        profile_plot_path = None
    
    # Create cluster size distribution plot
    plt.figure(figsize=(10, 6))
    
    # For size distribution, count only the clustered rows
    cluster_series = pd.Series(labels)
    cluster_sizes = cluster_series.value_counts().sort_index()
    
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
        # Use FIXED colors from tab20 to match the PCA scatter plot
        tab20_colors = plt.cm.tab20.colors  # Get all 20 colors from tab20

        # Create individual bars with colors matching the clusters
        # Use cluster number directly as index into tab20_colors
        for i, (cluster, size) in enumerate(cluster_sizes.items()):
            # Use the same color as in the PCA plot - cluster number as index
            color = tab20_colors[cluster % 20]  # Get color by cluster number (not position!)
            plt.bar(i, size, color=color, alpha=0.7)

        plt.xticks(range(len(cluster_sizes)), [f'Cluster {i}' for i in cluster_sizes.index])
    
    plt.xlabel('Cluster')
    plt.ylabel('Number of Samples')
    plt.title('Cluster Size Distribution')
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save the plot
    add_watermark_matplotlib_after_plot(plt.gcf())
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


def _lift_linkage_distances(linkage_matrix, max_extra_ratio=0.2, min_gap_abs=1e-6):
    """
    Adjust linkage distances for plotting so ties don't collapse into flat lines.

    This keeps the hierarchy structure but enforces a small vertical gap
    between a merge and its children for visibility.
    """
    Z_plot = np.array(linkage_matrix, copy=True)
    if Z_plot.size == 0:
        return Z_plot

    n_merges = Z_plot.shape[0]
    max_dist = np.max(Z_plot[:, 2])
    if max_dist <= 0:
        target_extra = 1.0
    else:
        target_extra = max_dist * max_extra_ratio
    min_gap = max(target_extra / max(n_merges, 1), min_gap_abs)

    n_leaves = n_merges + 1
    for i in range(n_merges):
        left = int(Z_plot[i, 0])
        right = int(Z_plot[i, 1])
        left_height = 0.0 if left < n_leaves else Z_plot[left - n_leaves, 2]
        right_height = 0.0 if right < n_leaves else Z_plot[right - n_leaves, 2]
        min_height = max(left_height, right_height) + min_gap
        if Z_plot[i, 2] < min_height:
            Z_plot[i, 2] = min_height

    return Z_plot


def _prepare_linkage_for_plot(linkage_matrix, max_extra_ratio=0.2, min_gap_abs=0.02):
    """
    Prepare linkage distances for plotting: log-scale and lift ties.
    """
    Z_plot = np.array(linkage_matrix, copy=True)
    if Z_plot.size == 0:
        return Z_plot

    Z_plot[:, 2] = np.log1p(Z_plot[:, 2])
    Z_plot = _lift_linkage_distances(Z_plot, max_extra_ratio=max_extra_ratio, min_gap_abs=min_gap_abs)
    return Z_plot


def _create_dendrogram_traces(linkage_matrix, leaf_positions, orientation='bottom', line_width=2, color='#636EFA'):
    """
    Create dendrogram traces manually from linkage matrix with custom leaf positions.

    This ensures the dendrogram aligns properly with heatmap cells.

    Parameters:
    -----------
    linkage_matrix : ndarray
        The linkage matrix from scipy.cluster.hierarchy.linkage
    leaf_positions : list
        The x (or y) positions where leaves should be placed (matching heatmap)
    orientation : str
        'bottom' for column dendrogram (leaves at bottom), 'left' for row dendrogram (leaves at left)
    line_width : int
        Width of dendrogram lines
    color : str
        Color of dendrogram lines

    Returns:
    --------
    list of go.Scatter traces
    """
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import dendrogram

    n_leaves = len(leaf_positions)

    # Lift ties and apply log scaling so low merges are visible
    linkage_plot = _prepare_linkage_for_plot(linkage_matrix)

    # Get dendrogram structure from scipy (without plotting)
    dendro_data = dendrogram(linkage_plot, no_plot=True)

    # scipy dendrogram returns icoord (x) and dcoord (y) for each link
    # icoord: x-coordinates of the four points of each link
    # dcoord: y-coordinates (heights) of the four points of each link
    icoord = dendro_data['icoord']
    dcoord = dendro_data['dcoord']

    # scipy uses positions 5, 15, 25, ... for leaves (step of 10, starting at 5)
    # We need to map these to our leaf_positions
    scipy_leaf_positions = [5 + 10 * i for i in range(n_leaves)]

    # Create mapping from scipy positions to our positions
    def map_position(scipy_pos):
        # Find closest scipy leaf position
        for i, sp in enumerate(scipy_leaf_positions):
            if abs(scipy_pos - sp) < 0.1:
                return leaf_positions[i]
        # For non-leaf positions (internal nodes), interpolate
        # Find which leaves this position is between
        for i in range(len(scipy_leaf_positions) - 1):
            if scipy_leaf_positions[i] < scipy_pos < scipy_leaf_positions[i + 1]:
                # Linear interpolation
                ratio = (scipy_pos - scipy_leaf_positions[i]) / (scipy_leaf_positions[i + 1] - scipy_leaf_positions[i])
                return leaf_positions[i] + ratio * (leaf_positions[i + 1] - leaf_positions[i])
        # Fallback - linear scaling
        scipy_min, scipy_max = scipy_leaf_positions[0], scipy_leaf_positions[-1]
        pos_min, pos_max = leaf_positions[0], leaf_positions[-1]
        return pos_min + (scipy_pos - scipy_min) / (scipy_max - scipy_min) * (pos_max - pos_min)

    traces = []

    for i in range(len(icoord)):
        # Map x coordinates (leaf positions)
        mapped_x = [map_position(x) for x in icoord[i]]
        # y coordinates are already lifted/log-scaled
        mapped_y = list(dcoord[i])

        if orientation == 'bottom':
            # Column dendrogram: x = leaf positions, y = heights
            trace = go.Scatter(
                x=mapped_x,
                y=mapped_y,
                mode='lines',
                line=dict(color=color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            )
        else:  # orientation == 'left'
            # Row dendrogram: x = heights (reversed), y = leaf positions
            trace = go.Scatter(
                x=mapped_y,  # heights become x
                y=mapped_x,  # leaf positions become y
                mode='lines',
                line=dict(color=color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            )

        traces.append(trace)

    return traces


def _create_dendrogram_traces_colored(linkage_matrix, leaf_positions, leaf_labels, color_labels,
                                       orientation='bottom', line_width=2, default_color='#888888',
                                       custom_colors=None):
    """
    Create dendrogram traces with branches colored by category.

    Branches are colored based on the category of leaves they connect.
    If a branch connects leaves of different categories, it uses the default color.

    Parameters:
    -----------
    linkage_matrix : ndarray
        The linkage matrix from scipy.cluster.hierarchy.linkage
    leaf_positions : list
        The x (or y) positions where leaves should be placed (matching heatmap)
    leaf_labels : list
        Labels for each leaf (used for matching with color_labels)
    color_labels : dict
        Dictionary mapping leaf labels to their category (e.g., {'sample1': 'TypeA', 'sample2': 'TypeB'})
    orientation : str
        'bottom' for column dendrogram (leaves at bottom), 'left' for row dendrogram (leaves at left)
    line_width : int
        Width of dendrogram lines
    default_color : str
        Color for branches connecting different categories
    custom_colors : dict
        Dictionary mapping category names to hex color strings.
        E.g., {'TypeA': '#ff0000', 'TypeB': '#00ff00'}
        If None, uses default tab10 color palette.

    Returns:
    --------
    tuple (list of go.Scatter traces, dict of category colors)
    """
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import dendrogram

    n_leaves = len(leaf_positions)

    # Get unique categories and assign colors
    unique_categories = list(set(color_labels.values()))
    unique_categories.sort()  # Sort for consistent coloring

    # Default colors (tab10 palette)
    default_palette = [
        '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
        '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]

    # Build category_colors dict
    category_colors = {}
    for i, cat in enumerate(unique_categories):
        if custom_colors and cat in custom_colors:
            # Use custom color if provided
            category_colors[cat] = custom_colors[cat]
        else:
            # Fallback to default palette
            category_colors[cat] = default_palette[i % len(default_palette)]

    # Convert color_labels keys to strings for consistent matching
    color_labels_str = {str(k): v for k, v in color_labels.items()}

    # Build category sets for each cluster using linkage matrix
    # Cluster IDs: 0 to n_leaves-1 are original leaves, n_leaves+ are merged clusters
    cluster_categories = {}

    # Initialize leaves with their categories
    for i, label in enumerate(leaf_labels):
        label_str = str(label)
        if label_str in color_labels_str:
            cluster_categories[i] = {color_labels_str[label_str]}
        else:
            cluster_categories[i] = set()

    # Process linkage matrix to build category sets for merged clusters
    for i, row in enumerate(linkage_matrix):
        left_id = int(row[0])
        right_id = int(row[1])
        new_id = n_leaves + i

        left_cats = cluster_categories.get(left_id, set())
        right_cats = cluster_categories.get(right_id, set())
        cluster_categories[new_id] = left_cats | right_cats

    # Lift ties and apply log scaling so low merges are visible
    linkage_plot = _prepare_linkage_for_plot(linkage_matrix)

    # Get dendrogram structure from scipy (without plotting)
    dendro_data = dendrogram(linkage_plot, no_plot=True)

    icoord = dendro_data['icoord']
    dcoord = dendro_data['dcoord']
    leaves_order = dendro_data['leaves']

    # scipy uses positions 5, 15, 25, ... for leaves
    scipy_leaf_positions = [5 + 10 * i for i in range(n_leaves)]

    # Create mapping from scipy positions to our positions
    def map_position(scipy_pos):
        for i, sp in enumerate(scipy_leaf_positions):
            if abs(scipy_pos - sp) < 0.1:
                return leaf_positions[i]
        for i in range(len(scipy_leaf_positions) - 1):
            if scipy_leaf_positions[i] < scipy_pos < scipy_leaf_positions[i + 1]:
                ratio = (scipy_pos - scipy_leaf_positions[i]) / (scipy_leaf_positions[i + 1] - scipy_leaf_positions[i])
                return leaf_positions[i] + ratio * (leaf_positions[i + 1] - leaf_positions[i])
        scipy_min, scipy_max = scipy_leaf_positions[0], scipy_leaf_positions[-1]
        pos_min, pos_max = leaf_positions[0], leaf_positions[-1]
        return pos_min + (scipy_pos - scipy_min) / (scipy_max - scipy_min) * (pos_max - pos_min)

    # Match dendrogram links to linkage matrix rows by height
    # Sort linkage rows by height and dendrogram links by height
    linkage_heights = [(i, linkage_plot[i, 2]) for i in range(len(linkage_plot))]
    linkage_heights.sort(key=lambda x: x[1])

    dendro_heights = [(i, max(dcoord[i])) for i in range(len(dcoord))]
    dendro_heights.sort(key=lambda x: x[1])

    # Create mapping: dendrogram link index -> linkage row index
    dendro_to_linkage = {}
    for (dendro_idx, _), (linkage_idx, _) in zip(dendro_heights, linkage_heights):
        dendro_to_linkage[dendro_idx] = linkage_idx

    # Debug: show some cluster categories
    single_cat_clusters = [(k, v) for k, v in cluster_categories.items() if len(v) == 1]
    print(f"DEBUG: Clusters with single category: {len(single_cat_clusters)} out of {len(cluster_categories)}")
    print(f"DEBUG: Sample single-cat clusters: {single_cat_clusters[:5]}")

    traces = []

    for link_idx in range(len(icoord)):
        ic = icoord[link_idx]
        dc = dcoord[link_idx]

        # Get the corresponding linkage row to find left and right children
        linkage_row_idx = dendro_to_linkage.get(link_idx, None)

        if linkage_row_idx is not None:
            left_child = int(linkage_matrix[linkage_row_idx, 0])
            right_child = int(linkage_matrix[linkage_row_idx, 1])
            left_cats = cluster_categories.get(left_child, set())
            right_cats = cluster_categories.get(right_child, set())
        else:
            left_cats = set()
            right_cats = set()

        # Determine colors for left leg, right leg, and top bar
        left_color = category_colors.get(list(left_cats)[0], default_color) if len(left_cats) == 1 else default_color
        right_color = category_colors.get(list(right_cats)[0], default_color) if len(right_cats) == 1 else default_color
        # Top bar is colored only if both sides have the same single category
        if len(left_cats) == 1 and len(right_cats) == 1 and left_cats == right_cats:
            top_color = left_color
        else:
            top_color = default_color

        # U-shape points: [x0,y0], [x1,y1], [x2,y2], [x3,y3]
        # x0,y0 = bottom-left, x1,y1 = top-left, x2,y2 = top-right, x3,y3 = bottom-right
        # Left leg: (x0,y0) to (x1,y1)
        # Top bar: (x1,y1) to (x2,y2)
        # Right leg: (x2,y2) to (x3,y3)

        if orientation == 'bottom':
            # Left leg
            traces.append(go.Scatter(
                x=[map_position(ic[0]), map_position(ic[1])],
                y=[dc[0], dc[1]],
                mode='lines',
                line=dict(color=left_color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
            # Top bar
            traces.append(go.Scatter(
                x=[map_position(ic[1]), map_position(ic[2])],
                y=[dc[1], dc[2]],
                mode='lines',
                line=dict(color=top_color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
            # Right leg
            traces.append(go.Scatter(
                x=[map_position(ic[2]), map_position(ic[3])],
                y=[dc[2], dc[3]],
                mode='lines',
                line=dict(color=right_color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
        else:
            # Left leg (for 'left' orientation, x and y are swapped)
            traces.append(go.Scatter(
                x=[dc[0], dc[1]],
                y=[map_position(ic[0]), map_position(ic[1])],
                mode='lines',
                line=dict(color=left_color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
            # Top bar
            traces.append(go.Scatter(
                x=[dc[1], dc[2]],
                y=[map_position(ic[1]), map_position(ic[2])],
                mode='lines',
                line=dict(color=top_color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
            # Right leg
            traces.append(go.Scatter(
                x=[dc[2], dc[3]],
                y=[map_position(ic[2]), map_position(ic[3])],
                mode='lines',
                line=dict(color=right_color, width=line_width),
                hoverinfo='skip',
                showlegend=False
            ))
    return traces, category_colors


def generate_twoway_hca_heatmap(df, selected_variables, grouping_column, row_linkage='ward',
                                col_linkage='ward', temp_path='temp/', height_scale=100, width_scale=100,
                                row_color_column=None, show_zeros=False, custom_colors=None,
                                x_axis_font_size=None, y_axis_font_size=None,
                                endpoint_column=None, endpoint_data=None, endpoint_is_numeric=False):
    """
    Generate an interactive two-way hierarchical clustering heatmap using Plotly.

    X axis: Variables (selected by user)
    Y axis: Groups (e.g., regions, clusters) - mean values for each group

    Parameters:
    -----------
    df : pandas.DataFrame
        The input data
    selected_variables : list
        List of variable names to include in the heatmap (X axis)
    grouping_column : str
        Column name to group samples by (Y axis) - e.g., 'region', 'Cluster', etc.
    row_linkage : str, default='ward'
        Linkage method for row (group) clustering
    col_linkage : str, default='ward'
        Linkage method for column (variable) clustering
    temp_path : str, default='temp/'
        Path to temporary directory for saving plots
    height_scale : int, default=100
        Scale for top dendrogram height (heatmap cells stay constant)
    width_scale : int, default=100
        Scale for left dendrogram width (heatmap cells stay constant)
    row_color_column : str, default=None
        Column name to use for coloring the row dendrogram branches.
        Each unique value in this column will get a different color.
        If None, dendrogram uses a single color.
    show_zeros : bool, default=False
        If True, display crossed markers on cells where the original
        (pre-scaled) value is exactly 0.
    custom_colors : dict, default=None
        Dictionary mapping category names to hex color strings.
        E.g., {'TypeA': '#ff0000', 'TypeB': '#00ff00'}
        If None, uses default tab10 color palette.
    x_axis_font_size : int, default=None
        Font size for X-axis labels (variables). If None, auto-calculated based on number of columns.
    y_axis_font_size : int, default=None
        Font size for Y-axis labels (groups). If None, auto-calculated based on number of rows.
    endpoint_column : str, default=None
        Name of the external endpoint column to display as a color strip next to the heatmap.
    endpoint_data : pandas.Series, default=None
        The endpoint data series, aligned with df rows. Produced by coloring.prepare_color_data().
    endpoint_is_numeric : bool, default=False
        Whether the endpoint data is numeric (gradient) or categorical (discrete colors).

    Returns:
    --------
    str
        HTML string containing the Plotly heatmap
    """
    import plotly.figure_factory as ff
    import plotly.graph_objects as go
    from scipy.cluster.hierarchy import linkage as scipy_linkage

    # Ensure the temporary directory exists
    os.makedirs(temp_path, exist_ok=True)

    # Check if grouping column exists
    if grouping_column not in df.columns:
        raise ValueError(f"Grouping column '{grouping_column}' not found in dataset")

    # Filter to only selected variables that exist in the dataframe
    available_vars = [var for var in selected_variables if var in df.columns and var != grouping_column]

    if len(available_vars) == 0:
        raise ValueError("None of the selected variables exist in the dataset")

    if len(available_vars) < 2:
        raise ValueError("At least 2 variables are required for column clustering")

    # Get the data for heatmap (grouping column + selected variables)
    cols_to_use = [grouping_column] + available_vars
    heatmap_data = df[cols_to_use].copy()

    # Remove rows with any NaN values
    heatmap_data = heatmap_data.dropna()

    if len(heatmap_data) == 0:
        raise ValueError("No data available after removing missing values")

    # Group by the grouping column and calculate mean for each group
    grouped_data = heatmap_data.groupby(grouping_column)[available_vars].mean()

    if len(grouped_data) < 2:
        raise ValueError(f"At least 2 groups are required for row clustering. Found {len(grouped_data)} groups in '{grouping_column}'")

    # Aggregate endpoint data by grouping column if provided
    endpoint_grouped = None
    endpoint_categories = None
    endpoint_cat_colors = None
    if endpoint_column is not None and endpoint_data is not None:
        # Build a temporary df for aggregation
        ep_df = pd.DataFrame({
            grouping_column: heatmap_data[grouping_column].values,
            'endpoint': endpoint_data.reindex(heatmap_data.index).values
        })
        if endpoint_is_numeric:
            endpoint_grouped = ep_df.groupby(grouping_column)['endpoint'].mean()
        else:
            # Mode for categorical data
            endpoint_grouped = ep_df.groupby(grouping_column)['endpoint'].agg(
                lambda x: x.mode().iloc[0] if not x.mode().empty else None
            )

    # Get group labels (Y axis)
    group_labels = grouped_data.index.astype(str).tolist()

    # Standardize the data for better visualization
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    grouped_data_scaled = scaler.fit_transform(grouped_data.values)

    # Perform hierarchical clustering on rows (groups)
    row_linkage_matrix = scipy_linkage(grouped_data_scaled, method=row_linkage)

    # Perform hierarchical clustering on columns (variables)
    col_linkage_matrix = scipy_linkage(grouped_data_scaled.T, method=col_linkage)

    # Create row dendrogram to get row order
    row_fig = ff.create_dendrogram(
        grouped_data_scaled,
        orientation='left',
        labels=group_labels,
        linkagefun=lambda x: scipy_linkage(x, method=row_linkage)
    )

    # Get the reordered indices/positions from the dendrogram
    row_dendro_leaves = row_fig['layout']['yaxis']['ticktext']
    row_dendro_positions = row_fig['layout']['yaxis']['tickvals']
    row_order = [group_labels.index(label) for label in row_dendro_leaves]

    # Create column dendrogram to get column order
    col_fig = ff.create_dendrogram(
        grouped_data_scaled.T,
        orientation='bottom',
        labels=available_vars,
        linkagefun=lambda x: scipy_linkage(x, method=col_linkage)
    )

    col_dendro_leaves = col_fig['layout']['xaxis']['ticktext']
    col_dendro_positions = col_fig['layout']['xaxis']['tickvals']
    col_order = [available_vars.index(label) for label in col_dendro_leaves]

    # Reorder the heatmap data (both scaled and original)
    heatmap_reordered = grouped_data_scaled[row_order, :][:, col_order]
    heatmap_original_reordered = grouped_data.values[row_order, :][:, col_order]
    reordered_row_labels = [group_labels[i] for i in row_order]
    reordered_col_labels = [available_vars[i] for i in col_order]
    reordered_row_positions = [row_dendro_positions[i] for i in range(len(row_dendro_positions))]
    reordered_col_positions = [col_dendro_positions[i] for i in range(len(col_dendro_positions))]

    # Create the final combined figure with dendrograms and heatmap
    from plotly.subplots import make_subplots

    # Create dendrogram traces using custom function for proper alignment
    col_dendro_traces = _create_dendrogram_traces(
        col_linkage_matrix,
        reordered_col_positions,
        orientation='bottom',
        line_width=2,
        color="#000000"
    )

    # Build color mapping for row dendrogram if row_color_column is specified
    category_colors = None
    if row_color_column and row_color_column in df.columns:
        # Build mapping from group labels to their color category
        # For each group (from grouping_column), get the most common value of row_color_column
        color_mapping = {}
        for group_label in group_labels:
            group_mask = df[grouping_column].astype(str) == str(group_label)
            if group_mask.any():
                # Get the most common value of row_color_column for this group
                color_values = df.loc[group_mask, row_color_column].dropna()
                if len(color_values) > 0:
                    most_common = color_values.mode()
                    if len(most_common) > 0:
                        color_mapping[group_label] = str(most_common.iloc[0])

        if color_mapping:
            print(f"DEBUG color_mapping: {color_mapping}")
            print(f"DEBUG group_labels: {group_labels}")
            row_dendro_traces, category_colors = _create_dendrogram_traces_colored(
                row_linkage_matrix,
                reordered_row_positions,
                group_labels,  # Original order labels
                color_mapping,
                orientation='left',
                line_width=2,
                default_color='#888888',
                custom_colors=custom_colors
            )
        else:
            row_dendro_traces = _create_dendrogram_traces(
                row_linkage_matrix,
                reordered_row_positions,
                orientation='left',
                line_width=2,
                color='#1f77b4'
            )
    else:
        row_dendro_traces = _create_dendrogram_traces(
            row_linkage_matrix,
            reordered_row_positions,
            orientation='left',
            line_width=2,
            color="#000000"
        )

    # Create the main heatmap
    heatmap_text = [[reordered_col_labels[j] for j in range(len(reordered_col_labels))]
                    for _ in reordered_row_labels]
    # Customdata contains [group_label, original_value] for each cell
    heatmap_customdata = [[[reordered_row_labels[i], heatmap_original_reordered[i, j]]
                           for j in range(len(reordered_col_labels))]
                          for i in range(len(reordered_row_labels))]

    heatmap = go.Heatmap(
        z=heatmap_reordered,
        x=reordered_col_positions,
        y=reordered_row_positions,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title=dict(text="Z-Score", side='top'), x=1.02, y=1.004, yanchor='top'),
        hovertemplate='Variable: %{text}<br>Group: %{customdata[0]}<br>Original Value: %{customdata[1]:.4f}<br>Z-Score: %{z:.2f}<extra></extra>',
        text=heatmap_text,
        customdata=heatmap_customdata
    )

    # Calculate proportions with dynamic cell sizing based on number of variables
    n_rows = len(reordered_row_labels)
    n_cols = len(reordered_col_labels)

    # Cell sizing - minimum size to fit labels, chart can expand as needed
    # Wider cells for fewer variables, narrower for many
    if n_cols > 50:
        cell_width = 25  # Narrow cells for many variables
    elif n_cols > 30:
        cell_width = 30
    else:
        cell_width = 35  # Standard width

    if n_rows > 30:
        cell_height = 35  # Shorter cells for many groups
    elif n_rows > 15:
        cell_height = 45
    else:
        cell_height = 50  # Standard height

    heatmap_height = n_rows * cell_height
    heatmap_width = n_cols * cell_width

    # Dendrogram size scales with parameters (larger base)
    base_top_dendro = 300
    base_left_dendro = 250
    top_dendro_height = int(base_top_dendro * height_scale / 100)
    left_dendro_width = int(base_left_dendro * width_scale / 100)

    # Calculate proportions
    top_ratio = top_dendro_height / (top_dendro_height + heatmap_height)
    left_ratio = left_dendro_width / (left_dendro_width + heatmap_width)

    # Calculate space needed for Y-axis labels based on longest label width
    max_label_len = max(len(str(label)) for label in reordered_row_labels)
    # Estimate label width in pixels: ~6-7px per character at font size 10-11
    y_font_size = 9 if n_rows > 30 else (10 if n_rows > 15 else 11)
    char_width = y_font_size * 0.65  # Average character width relative to font size
    estimated_label_width = max_label_len * char_width + 20  # +20px padding
    # Calculate preliminary total width
    preliminary_width = max(800, heatmap_width + left_dendro_width + 150)
    # Convert to ratio - use 1.5x multiplier for safety margin since labels extend leftward
    label_space = max(0.04, min(0.20, (estimated_label_width * 1.5) / preliminary_width))

    # Determine if endpoint strip is needed
    has_endpoint = (endpoint_column is not None and endpoint_data is not None and endpoint_grouped is not None)

    if has_endpoint:
        # Reorder endpoint_grouped to match row_order
        endpoint_reordered_values = endpoint_grouped.reindex(
            [grouped_data.index[i] for i in row_order]
        ).values

        if not endpoint_is_numeric:
            # Build categorical color map
            unique_cats = sorted(set(str(v) for v in endpoint_reordered_values if v is not None and str(v) != 'nan'))
            from app.chemalize.visualization.coloring import generate_distinct_colors
            cat_colors = generate_distinct_colors(len(unique_cats))
            endpoint_cat_colors = {cat: cat_colors[i] for i, cat in enumerate(unique_cats)}
            endpoint_categories = unique_cats

    # Compute endpoint strip x-position: just after the last heatmap column
    # Place it in the same subplot so there is no subplot-boundary gap.
    if has_endpoint:
        col_spacing = reordered_col_positions[1] - reordered_col_positions[0] if len(reordered_col_positions) > 1 else 10
        endpoint_x_pos = reordered_col_positions[-1] + col_spacing

    # Always use 2x2 grid (endpoint strip goes into the heatmap subplot)
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[top_ratio, 1 - top_ratio],
        column_widths=[left_ratio, 1 - left_ratio],
        specs=[[None, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        horizontal_spacing=label_space,
        vertical_spacing=0
    )

    # Add column dendrogram at top
    for trace in col_dendro_traces:
        fig.add_trace(trace, row=1, col=2)

    # Add row dendrogram on left
    for trace in row_dendro_traces:
        fig.add_trace(trace, row=2, col=1)

    # Add main heatmap
    fig.add_trace(heatmap, row=2, col=2)

    # Add legend traces for dendrogram colors if coloring is enabled
    show_legend = False
    max_category_len = 0
    if category_colors:
        show_legend = True
        max_category_len = max(len(str(cat)) for cat in category_colors.keys())
        for cat_name, cat_color in category_colors.items():
            # Add invisible scatter trace for legend
            fig.add_trace(go.Scatter(
                x=[None], y=[None],
                mode='markers',
                marker=dict(size=10, color=cat_color),
                name=str(cat_name),
                showlegend=True
            ))

    # Add markers for zero values if requested
    zeros_marked = False
    if show_zeros:
        # Find positions where original values are 0
        zero_x_positions = []
        zero_y_positions = []
        zero_hover_texts = []

        for i in range(len(reordered_row_labels)):
            for j in range(len(reordered_col_labels)):
                if heatmap_original_reordered[i, j] == 0:
                    zero_x_positions.append(reordered_col_positions[j])
                    zero_y_positions.append(reordered_row_positions[i])
                    # Get Z-Score for this cell
                    z_score = heatmap_reordered[i, j]
                    zero_hover_texts.append(
                        f"Variable: {reordered_col_labels[j]}<br>"
                        f"Group: {reordered_row_labels[i]}<br>"
                        f"Original Value: 0.0000<br>"
                        f"Z-Score: {z_score:.2f}"
                    )

        if zero_x_positions:
            zeros_marked = True
            # Add scatter markers with "x" symbol for zeros (not in legend)
            fig.add_trace(go.Scatter(
                x=zero_x_positions,
                y=zero_y_positions,
                mode='markers',
                marker=dict(
                    symbol='x',
                    size=12,
                    color='black',
                    line=dict(width=2, color='black')
                ),
                name='0',
                showlegend=False,
                hovertemplate='%{text}<extra></extra>',
                text=zero_hover_texts
            ), row=2, col=2)

    # Get the axis references for the heatmap subplot (row=2, col=2)
    # In 2x2 layout: heatmap is x3/y3; in 2x3 layout: heatmap is x4/y4
    heatmap_xref = fig.get_subplot(2, 2).xaxis.plotly_name  # e.g., 'xaxis3' or 'xaxis4'
    heatmap_yref = fig.get_subplot(2, 2).yaxis.plotly_name
    heatmap_xmatch = heatmap_xref.replace('axis', '')  # e.g., 'x3' or 'x4'
    heatmap_ymatch = heatmap_yref.replace('axis', '')

    # Add endpoint strip if available (same subplot as heatmap, col=2)
    if has_endpoint:
        if endpoint_is_numeric:
            ep_z = [[v] for v in endpoint_reordered_values]
            ep_text = [[f'{endpoint_column}: {v:.4f}' if v is not None and not (isinstance(v, float) and np.isnan(v)) else f'{endpoint_column}: N/A'] for v in endpoint_reordered_values]
            ep_heatmap = go.Heatmap(
                z=ep_z,
                x0=endpoint_x_pos,
                dx=col_spacing,
                y=reordered_row_positions,
                colorscale='Viridis',
                colorbar=dict(
                    title=dict(text=endpoint_column, side='right'),
                    x=1.08, y=1.004, yanchor='top'
                ),
                hovertemplate='%{text}<br>Group: %{customdata}<extra></extra>',
                text=ep_text,
                customdata=reordered_row_labels,
                showscale=True
            )
            fig.add_trace(ep_heatmap, row=2, col=2)
        else:
            cat_to_idx = {cat: i for i, cat in enumerate(endpoint_categories)}
            ep_z = [[cat_to_idx.get(str(v), -1)] for v in endpoint_reordered_values]
            ep_text = [[f'{endpoint_column}: {v}'] for v in endpoint_reordered_values]

            n_cats = len(endpoint_categories)
            discrete_colorscale = []
            for i, cat in enumerate(endpoint_categories):
                lower = i / n_cats
                upper = (i + 1) / n_cats
                color = endpoint_cat_colors[cat]
                discrete_colorscale.append([lower, color])
                discrete_colorscale.append([upper, color])

            ep_heatmap = go.Heatmap(
                z=ep_z,
                x0=endpoint_x_pos,
                dx=col_spacing,
                y=reordered_row_positions,
                colorscale=discrete_colorscale,
                zmin=0, zmax=n_cats,
                showscale=False,
                hovertemplate='%{text}<br>Group: %{customdata}<extra></extra>',
                text=ep_text,
                customdata=reordered_row_labels
            )
            fig.add_trace(ep_heatmap, row=2, col=2)

            show_legend = True
            ep_legend_labels = [f'{endpoint_column}: {c}' for c in endpoint_categories]
            max_category_len = max(max_category_len, max((len(l) for l in ep_legend_labels), default=0))
            for cat_name in endpoint_categories:
                fig.add_trace(go.Scatter(
                    x=[None], y=[None],
                    mode='markers',
                    marker=dict(size=10, color=endpoint_cat_colors[cat_name], symbol='square'),
                    name=f'{endpoint_column}: {cat_name}',
                    showlegend=True
                ))

        # Endpoint label will be added as a separate annotation below the strip
        # (not as an x-axis tick, so it can be horizontal while other ticks are angled)

    # Total figure size = heatmap + dendrogram + padding (no max limit - let it expand)
    final_height = max(500, heatmap_height + top_dendro_height + 150)
    # Add a bit of extra width for the endpoint strip column + colorbar
    endpoint_extra_width = (cell_width + 100) if has_endpoint else 0
    final_width = max(800, heatmap_width + left_dendro_width + 150 + endpoint_extra_width)

    # Add extra width for legend if showing (based on longest category name)
    if show_legend:
        # Estimate legend width: ~8px per character + padding
        legend_width = max(150, max_category_len * 8 + 80)
        final_width += legend_width

    # Bottom margin for -45° angled labels - will be auto-adjusted by automargin
    bottom_margin = 50

    # Build title with optional color column info
    title_text = f'Two-Way Hierarchical Clustering Heatmap<br><sub>Groups by {grouping_column} | Row: {row_linkage} linkage | Column: {col_linkage} linkage | {n_cols} variables × {n_rows} groups'
    if row_color_column and category_colors:
        title_text += f' | Colored by {row_color_column}'
    if zeros_marked:
        title_text += ' | ✕ = 0'
    if has_endpoint:
        ep_type = 'numeric' if endpoint_is_numeric else 'categorical'
        title_text += f' | Endpoint: {endpoint_column} ({ep_type})'
    title_text += '</sub>'

    # Adjust right margin for legend (based on longest category name)
    # Colorbar takes ~60px, then we need space for legend
    if show_legend:
        right_margin = max(200, max_category_len * 8 + 140)  # Extra space for colorbar + legend
    else:
        right_margin = 80  # Just colorbar

    fig.update_layout(
        title=title_text,
        height=final_height,
        width=final_width,
        showlegend=show_legend,
        legend=dict(
            title=dict(text=row_color_column if row_color_column else (endpoint_column if has_endpoint else "Category")),
            orientation='v',
            yanchor='top',
            y=1.0,
            xanchor='left',
            x=1.10  # Position to the right of colorbar (which is at x=1.02)
        ) if show_legend else None,
        hovermode='closest',
        plot_bgcolor='white',
        dragmode='pan',
        margin=dict(t=120, l=120, r=right_margin, b=bottom_margin)
    )

    # Update axes for column dendrogram - share X with heatmap, Y starts from 0
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, matches=heatmap_xmatch, row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, row=1, col=2,
                     rangemode='nonnegative', automargin=False)

    # Update axes for row dendrogram - share Y with heatmap, X starts from 0 (reversed)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed', rangemode='tozero', row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, matches=heatmap_ymatch, row=2, col=1)

    # Update axes for main heatmap (x3/y3)
    # Always use slight angle (-45°) for X-axis labels, adjust font size based on count
    x_tickangle = -45

    # X-axis font size: use custom value if provided, otherwise auto-calculate
    if x_axis_font_size is not None:
        x_tickfont = dict(size=x_axis_font_size)
    elif n_cols > 50:
        x_tickfont = dict(size=8)
    elif n_cols > 30:
        x_tickfont = dict(size=9)
    elif n_cols > 15:
        x_tickfont = dict(size=10)
    else:
        x_tickfont = dict(size=11)

    x_ticklabel_standoff = -20

    fig.update_xaxes(showticklabels=True, showgrid=False, zeroline=False, row=2, col=2,
                     tickangle=x_tickangle, side='bottom', tickmode='array',
                     tickvals=reordered_col_positions, ticktext=reordered_col_labels, type='linear',
                     tickfont=x_tickfont, ticks='', ticklabelstandoff=x_ticklabel_standoff, automargin=True)

    # Y-axis font size: use custom value if provided, otherwise auto-calculate
    if y_axis_font_size is not None:
        y_tickfont = dict(size=y_axis_font_size)
    else:
        y_tickfont = dict(size=9) if n_rows > 30 else (dict(size=10) if n_rows > 15 else dict(size=11))

    fig.update_yaxes(showticklabels=True, showgrid=False, zeroline=False, row=2, col=2, side='left',
                     tickmode='array', tickvals=reordered_row_positions, ticktext=reordered_row_labels,
                     autorange='reversed', type='linear', tickfont=y_tickfont, ticklabelstandoff=0, automargin=False)

    # Add annotation above the row dendrogram to label the grouping column
    fig.add_annotation(
        text=str(grouping_column),
        xref='paper', yref='paper',
        x=-0.12, y=1.2,
        xanchor='left', yanchor='bottom',
        yshift=10,
        showarrow=False,
        align='left',
        font=dict(size=12, color='black')
    )

    # Add endpoint column label below the strip (X axis), horizontal, bold
    if has_endpoint:
        # Keep endpoint label in sync with custom X-axis label spacing.
        endpoint_label_yshift = -10 - x_ticklabel_standoff + 10
        fig.add_annotation(
            text='<b>y</b>',
            x=endpoint_x_pos,
            xref=heatmap_xmatch,
            y=0,
            yref='paper',
            yshift=endpoint_label_yshift,
            xanchor='center', yanchor='top',
            showarrow=False,
            textangle=0,
            font=dict(size=int(x_tickfont['size'] * 1.5), color='black')
        )

    # Convert to HTML with higher resolution image export config
    config = {
        'toImageButtonOptions': {
            'format': 'png',
            'filename': 'twoway_hca_heatmap',
            'scale': 2  # 2x resolution
        },
        'displaylogo': False
    }
    html_string = fig.to_html(include_plotlyjs='cdn', div_id='twoway_hca_plot', config=config)

    return html_string
