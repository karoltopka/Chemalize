"""
Watermark utility module for ChemAlize application.
Provides functions to add watermarks to Matplotlib and Plotly plots.

Usage:
    1. Place your watermark image in: app/static/img/watermarks/watermark.png
    2. For Matplotlib plots: call add_watermark_matplotlib(fig) or add_watermark_matplotlib(plt.gca())
    3. For Plotly plots: call add_watermark_plotly(fig)
"""
import os
from PIL import Image
import base64
import io

# Path to watermark images directory
WATERMARK_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'static', 'img', 'watermarks')

# Default watermark filename
DEFAULT_WATERMARK = 'watermark.png'


def get_watermark_path(filename=None):
    """
    Get the full path to a watermark file.
    
    Parameters:
    -----------
    filename : str, optional
        Name of the watermark file. Defaults to 'watermark.png'.
        
    Returns:
    --------
    str
        Full path to the watermark file.
    """
    if filename is None:
        filename = DEFAULT_WATERMARK
    return os.path.join(WATERMARK_DIR, filename)


def watermark_exists(filename=None):
    """
    Check if the watermark file exists.
    
    Parameters:
    -----------
    filename : str, optional
        Name of the watermark file. Defaults to 'watermark.png'.
        
    Returns:
    --------
    bool
        True if the watermark file exists, False otherwise.
    """
    return os.path.exists(get_watermark_path(filename))


def add_watermark_matplotlib(fig_or_ax, watermark_file=None, alpha=0.15, zorder=0):
    """
    Add a watermark as full background to a Matplotlib figure or axes.
    The watermark covers the entire plot area.
    
    Parameters:
    -----------
    fig_or_ax : matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes to add the watermark to.
    watermark_file : str, optional
        Name of the watermark file in the watermarks directory.
        Defaults to 'watermark.png'.
    alpha : float, optional
        Transparency of the watermark (0.0 to 1.0). Default is 0.15.
    zorder : int, optional
        Z-order for the watermark (lower = behind plot elements). Default is 0.
        
    Returns:
    --------
    bool
        True if watermark was added successfully, False otherwise.
        
    Example:
    --------
    >>> import matplotlib.pyplot as plt
    >>> from app.utils.watermark import add_watermark_matplotlib
    >>> 
    >>> fig, ax = plt.subplots()
    >>> ax.plot([1, 2, 3], [1, 4, 9])
    >>> add_watermark_matplotlib(fig)
    >>> plt.savefig('my_plot.png')
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    watermark_path = get_watermark_path(watermark_file)
    
    if not os.path.exists(watermark_path):
        print(f"Warning: Watermark file not found at {watermark_path}")
        return False
    
    try:
        # Load the watermark image
        watermark_img = Image.open(watermark_path)
        
        # Convert to RGBA if needed
        if watermark_img.mode != 'RGBA':
            watermark_img = watermark_img.convert('RGBA')
        
        # Get the figure and axes
        if hasattr(fig_or_ax, 'get_axes'):
            fig = fig_or_ax
            axes_list = fig.get_axes()
        else:
            ax = fig_or_ax
            fig = ax.get_figure()
            axes_list = [ax]
        
        if not axes_list:
            return False
        
        # Add watermark to each axes
        for ax in axes_list:
            # Get the axes extent in figure coordinates
            bbox = ax.get_position()
            
            # Calculate pixel dimensions for the axes area
            fig_width_pixels = int(fig.get_figwidth() * fig.dpi)
            fig_height_pixels = int(fig.get_figheight() * fig.dpi)
            
            ax_width_pixels = int(bbox.width * fig_width_pixels)
            ax_height_pixels = int(bbox.height * fig_height_pixels)
            
            # Resize watermark to fit the entire axes area
            watermark_resized = watermark_img.resize(
                (ax_width_pixels, ax_height_pixels), 
                Image.Resampling.LANCZOS
            )
            
            # Apply alpha
            watermark_array = np.array(watermark_resized, dtype=np.float32)
            watermark_array[:, :, 3] = watermark_array[:, :, 3] * alpha
            watermark_array = watermark_array.astype(np.uint8)
            
            # Add the watermark as a background image covering the entire plot
            ax.imshow(
                watermark_array,
                extent=ax.get_xlim() + ax.get_ylim(),
                aspect='auto',
                zorder=zorder,
                alpha=1.0  # Alpha is already applied to the image
            )
            
            # Update axis limits might change after imshow, restore them
            ax.autoscale(False)
        
        return True
        
    except Exception as e:
        print(f"Error adding watermark: {str(e)}")
        return False


def add_watermark_matplotlib_after_plot(fig_or_ax, watermark_file=None, alpha=0.15):
    """
    Add a watermark as full background AFTER all plotting is done.
    Call this right before plt.savefig().
    
    The watermark maintains its original aspect ratio and is centered in the plot.
    
    Parameters:
    -----------
    fig_or_ax : matplotlib.figure.Figure or matplotlib.axes.Axes
        The figure or axes to add the watermark to.
    watermark_file : str, optional
        Name of the watermark file. Defaults to 'watermark.png'.
    alpha : float, optional
        Transparency (0.0 to 1.0). Default is 0.15.
        
    Returns:
    --------
    bool
        True if watermark was added successfully, False otherwise.
    """
    import matplotlib.pyplot as plt
    from matplotlib.transforms import Bbox, TransformedBbox, BboxTransformTo
    from matplotlib.image import BboxImage
    import numpy as np
    
    watermark_path = get_watermark_path(watermark_file)
    
    if not os.path.exists(watermark_path):
        print(f"Warning: Watermark file not found at {watermark_path}")
        return False
    
    try:
        watermark_img = Image.open(watermark_path)
        
        if watermark_img.mode != 'RGBA':
            watermark_img = watermark_img.convert('RGBA')
        
        # Get figure and axes
        if hasattr(fig_or_ax, 'get_axes'):
            fig = fig_or_ax
            axes_list = fig.get_axes()
        else:
            ax = fig_or_ax
            fig = ax.get_figure()
            axes_list = [ax]
        
        if not axes_list:
            return False
        
        # Apply alpha to image
        watermark_array = np.array(watermark_img, dtype=np.float32)
        watermark_array[:, :, 3] = watermark_array[:, :, 3] * alpha
        watermark_array = watermark_array.astype(np.uint8)
        
        # Get original image aspect ratio
        img_height, img_width = watermark_array.shape[:2]
        img_aspect = img_width / img_height
        
        for ax in axes_list:
            # Get axes dimensions from figure coordinates (more reliable than window_extent before savefig)
            ax_bbox = ax.get_position()
            fig_width = fig.get_figwidth()
            fig_height = fig.get_figheight()
            ax_width = ax_bbox.width * fig_width
            ax_height = ax_bbox.height * fig_height
            ax_aspect = ax_width / ax_height
            
            # Calculate the position to center the watermark while maintaining aspect ratio
            if img_aspect > ax_aspect:
                # Image is wider than axes - fit to width
                w = 1.0
                h = ax_aspect / img_aspect
                x0 = 0.0
                y0 = (1.0 - h) / 2.0
            else:
                # Image is taller than axes - fit to height
                h = 1.0
                w = img_aspect / ax_aspect
                x0 = (1.0 - w) / 2.0
                y0 = 0.0
            
            # Create a BboxImage that maintains aspect ratio
            bbox_image = BboxImage(
                TransformedBbox(Bbox([[x0, y0], [x0 + w, y0 + h]]), ax.transAxes),
                zorder=-1,  # Very low zorder to be behind everything
                alpha=1.0   # Alpha is already applied to the image array
            )
            bbox_image.set_data(watermark_array)
            ax.add_artist(bbox_image)
        
        return True
        
    except Exception as e:
        print(f"Error adding watermark: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def add_watermark_plotly(fig, watermark_file=None, opacity=0.15):
    """
    Add a watermark as full background to a Plotly figure.
    The watermark covers the entire plot area.
    
    Parameters:
    -----------
    fig : plotly.graph_objects.Figure
        The Plotly figure to add the watermark to.
    watermark_file : str, optional
        Name of the watermark file in the watermarks directory.
        Defaults to 'watermark.png'.
    opacity : float, optional
        Transparency of the watermark (0.0 to 1.0). Default is 0.15.
        
    Returns:
    --------
    plotly.graph_objects.Figure
        The figure with the watermark added.
        
    Example:
    --------
    >>> import plotly.graph_objects as go
    >>> from app.utils.watermark import add_watermark_plotly
    >>> 
    >>> fig = go.Figure(data=go.Scatter(x=[1, 2, 3], y=[1, 4, 9]))
    >>> fig = add_watermark_plotly(fig)
    >>> fig.show()
    """
    watermark_path = get_watermark_path(watermark_file)
    
    if not os.path.exists(watermark_path):
        print(f"Warning: Watermark file not found at {watermark_path}")
        return fig
    
    try:
        # Load and encode the watermark image
        with open(watermark_path, 'rb') as f:
            watermark_data = f.read()
        
        # Detect image type
        if watermark_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif watermark_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        
        # Encode as base64
        encoded = base64.b64encode(watermark_data).decode('utf-8')
        image_source = f"data:{mime_type};base64,{encoded}"
        
        # Add image covering the entire plot area (0 to 1 in paper coordinates)
        fig.add_layout_image(
            dict(
                source=image_source,
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=1,
                sizey=1,
                xanchor="left",
                yanchor="top",
                opacity=opacity,
                layer="below"  # Put behind plot elements
            )
        )
        
        return fig
        
    except Exception as e:
        print(f"Error adding watermark: {str(e)}")
        return fig


def get_watermark_base64(watermark_file=None):
    """
    Get the watermark image as a base64 encoded string.
    Useful for embedding in HTML or other contexts.
    
    Parameters:
    -----------
    watermark_file : str, optional
        Name of the watermark file. Defaults to 'watermark.png'.
        
    Returns:
    --------
    str or None
        Base64 encoded image data URI, or None if file not found.
    """
    watermark_path = get_watermark_path(watermark_file)
    
    if not os.path.exists(watermark_path):
        return None
    
    try:
        with open(watermark_path, 'rb') as f:
            watermark_data = f.read()
        
        if watermark_path.lower().endswith('.png'):
            mime_type = 'image/png'
        elif watermark_path.lower().endswith(('.jpg', '.jpeg')):
            mime_type = 'image/jpeg'
        else:
            mime_type = 'image/png'
        
        encoded = base64.b64encode(watermark_data).decode('utf-8')
        return f"data:{mime_type};base64,{encoded}"
        
    except Exception as e:
        print(f"Error encoding watermark: {str(e)}")
        return None
