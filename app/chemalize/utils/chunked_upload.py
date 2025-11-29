"""
Chunked file upload utilities for handling large files.
Important for Cloudflare Tunnel which has 100MB request limit.
"""
import os
from werkzeug.utils import secure_filename
from flask import current_app


def save_chunked_upload(file_storage, destination_path, chunk_size=None):
    """
    Save uploaded file in chunks to handle large files efficiently.

    Args:
        file_storage: FileStorage object from request.files
        destination_path: Full path where to save the file
        chunk_size: Size of chunks in bytes (default from config)

    Returns:
        tuple: (success: bool, file_path: str, error_message: str)
    """
    if chunk_size is None:
        chunk_size = current_app.config.get('UPLOAD_CHUNK_SIZE', 5242880)  # 5MB default

    try:
        # Ensure directory exists
        os.makedirs(os.path.dirname(destination_path), exist_ok=True)

        # Write file in chunks
        with open(destination_path, 'wb') as f:
            while True:
                chunk = file_storage.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)

        # Get file size for verification
        file_size = os.path.getsize(destination_path)
        file_size_mb = file_size / (1024 * 1024)

        return True, destination_path, f"File uploaded successfully ({file_size_mb:.2f} MB)"

    except Exception as e:
        return False, None, f"Upload failed: {str(e)}"


def validate_file_upload(file_storage, allowed_extensions=None, max_size=None):
    """
    Validate uploaded file before processing.

    Args:
        file_storage: FileStorage object from request.files
        allowed_extensions: Set of allowed extensions (e.g., {'.csv', '.xlsx'})
        max_size: Maximum file size in bytes

    Returns:
        tuple: (valid: bool, error_message: str)
    """
    if not file_storage or not file_storage.filename:
        return False, "No file selected"

    filename = secure_filename(file_storage.filename)

    # Check extension
    if allowed_extensions:
        file_ext = os.path.splitext(filename)[1].lower()
        if file_ext not in allowed_extensions:
            return False, f"File type not allowed. Allowed: {', '.join(allowed_extensions)}"

    # Check file size (if we can get it from headers)
    if max_size:
        # Note: file_storage.content_length might be None
        if hasattr(file_storage, 'content_length') and file_storage.content_length:
            if file_storage.content_length > max_size:
                max_mb = max_size / (1024 * 1024)
                return False, f"File too large. Maximum size: {max_mb:.0f} MB"

    return True, ""


def get_file_info(file_storage):
    """Get information about uploaded file."""
    info = {
        'filename': secure_filename(file_storage.filename),
        'content_type': file_storage.content_type,
        'size': getattr(file_storage, 'content_length', None),
    }

    if info['size']:
        info['size_mb'] = info['size'] / (1024 * 1024)

    return info
