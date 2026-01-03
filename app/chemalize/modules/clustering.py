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


def generate_twoway_hca_heatmap(df, selected_variables, grouping_column, row_linkage='ward',
                                col_linkage='ward', temp_path='temp/', height_scale=100, width_scale=100):
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

    # Reorder the heatmap data
    heatmap_reordered = grouped_data_scaled[row_order, :][:, col_order]
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
        color='#1f77b4'
    )

    row_dendro_traces = _create_dendrogram_traces(
        row_linkage_matrix,
        reordered_row_positions,
        orientation='left',
        line_width=2,
        color='#1f77b4'
    )

    # Create the main heatmap
    heatmap_text = [[reordered_col_labels[j] for j in range(len(reordered_col_labels))]
                    for _ in reordered_row_labels]
    heatmap_groups = [[label for _ in reordered_col_labels] for label in reordered_row_labels]

    heatmap = go.Heatmap(
        z=heatmap_reordered,
        x=reordered_col_positions,
        y=reordered_row_positions,
        colorscale='RdBu_r',
        zmid=0,
        colorbar=dict(title="Scaled Mean Value", x=1.1),
        hovertemplate='Variable: %{text}<br>Group: %{customdata}<br>Mean Value (scaled): %{z:.2f}<extra></extra>',
        text=heatmap_text,
        customdata=heatmap_groups
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

    # Create a subplot layout: row dendrogram | heatmap
    #                          column dendrogram (above heatmap)
    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[top_ratio, 1 - top_ratio],
        column_widths=[left_ratio, 1 - left_ratio],
        specs=[[None, {'type': 'xy'}],
               [{'type': 'xy'}, {'type': 'xy'}]],
        horizontal_spacing=label_space,  # Gap for labels between dendrogram and heatmap
        vertical_spacing=0.02
    )

    # Add column dendrogram at top
    for trace in col_dendro_traces:
        fig.add_trace(trace, row=1, col=2)

    # Add row dendrogram on left
    for trace in row_dendro_traces:
        fig.add_trace(trace, row=2, col=1)

    # Add main heatmap
    fig.add_trace(heatmap, row=2, col=2)

    # Total figure size = heatmap + dendrogram + padding (no max limit - let it expand)
    final_height = max(500, heatmap_height + top_dendro_height + 150)
    final_width = max(800, heatmap_width + left_dendro_width + 150)

    # Adjust bottom margin based on label angle (more space for vertical labels)
    bottom_margin = 150 if n_cols > 30 else (120 if n_cols > 15 else 100)

    fig.update_layout(
        title=f'Two-Way Hierarchical Clustering Heatmap<br><sub>Groups by {grouping_column} | Row: {row_linkage} linkage | Column: {col_linkage} linkage | {n_cols} variables × {n_rows} groups</sub>',
        height=final_height,
        width=final_width,
        showlegend=False,
        hovermode='closest',
        plot_bgcolor='white',
        dragmode='pan',
        margin=dict(t=150, l=120, r=60, b=bottom_margin)
    )

    # Update axes for column dendrogram - share X with heatmap, Y starts from 0
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, matches='x3', row=1, col=2)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, rangemode='tozero', row=1, col=2)

    # Update axes for row dendrogram - share Y with heatmap, X starts from 0 (reversed)
    fig.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, autorange='reversed', rangemode='tozero', row=2, col=1)
    fig.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, matches='y3', row=2, col=1)

    # Update axes for main heatmap (x3/y3)
    # Adjust label angle and font based on number of variables - always show ALL labels
    if n_cols > 50:
        x_tickangle = -90  # Vertical for many variables
        x_tickfont = dict(size=8)
    elif n_cols > 30:
        x_tickangle = -90
        x_tickfont = dict(size=9)
    elif n_cols > 15:
        x_tickangle = -60
        x_tickfont = dict(size=10)
    else:
        x_tickangle = -45
        x_tickfont = dict(size=11)

    fig.update_xaxes(showticklabels=True, showgrid=False, zeroline=False, row=2, col=2,
                     tickangle=x_tickangle, side='bottom', tickmode='array',
                     tickvals=reordered_col_positions, ticktext=reordered_col_labels, type='linear',
                     tickfont=x_tickfont)

    # Y-axis labels (groups) - adjust font size if many groups
    y_tickfont = dict(size=9) if n_rows > 30 else (dict(size=10) if n_rows > 15 else dict(size=11))
    # Calculate standoff based on label length - longer labels need more space from dendrogram
    y_standoff = max(5, int(max_label_len * 0.5))
    fig.update_yaxes(showticklabels=True, showgrid=False, zeroline=False, row=2, col=2, side='left',
                     tickmode='array', tickvals=reordered_row_positions, ticktext=reordered_row_labels,
                     autorange='reversed', type='linear', tickfont=y_tickfont, ticklabelstandoff=y_standoff)

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
