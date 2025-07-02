"""
Input validation utilities for security and data integrity.

This module provides validators for user inputs to prevent injection attacks
and ensure data quality.
"""

import re
from typing import List, Optional
from urllib.parse import urlparse


def validate_username(username: str) -> bool:
    """
    Validate social media username.
    
    Parameters
    ----------
    username : str
        Username to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Allow alphanumeric, underscore, dot, max 30 chars
    pattern = r'^[a-zA-Z0-9_.]{1,30}$'
    return bool(re.match(pattern, username))


def validate_platform(platform: str) -> bool:
    """
    Validate platform name.
    
    Parameters
    ----------
    platform : str
        Platform name to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    allowed_platforms = ["instagram", "tiktok"]
    return platform.lower() in allowed_platforms


def validate_limit(limit: int, min_val: int = 1, max_val: int = 500) -> bool:
    """
    Validate numeric limit.
    
    Parameters
    ----------
    limit : int
        Limit value to validate.
    min_val : int
        Minimum allowed value.
    max_val : int
        Maximum allowed value.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    return isinstance(limit, int) and min_val <= limit <= max_val


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent path traversal.
    
    Parameters
    ----------
    filename : str
        Filename to sanitize.
        
    Returns
    -------
    str
        Sanitized filename.
    """
    # Remove path separators and special characters
    sanitized = re.sub(r'[^\w\s.-]', '', filename)
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    # Replace spaces with underscores
    sanitized = sanitized.replace(' ', '_')
    # Limit length
    return sanitized[:255]


def validate_url(url: str) -> bool:
    """
    Validate URL format.
    
    Parameters
    ----------
    url : str
        URL to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def validate_date_format(date_str: str) -> bool:
    """
    Validate date string format.
    
    Parameters
    ----------
    date_str : str
        Date string to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Allow formats like "7 days", "2024-01-01", etc.
    patterns = [
        r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
        r'^\d+\s+days?$',        # N days
        r'^\d+\s+weeks?$',       # N weeks
        r'^\d+\s+months?$'       # N months
    ]
    
    return any(re.match(pattern, date_str) for pattern in patterns)


def validate_config(config: dict) -> tuple[bool, Optional[str]]:
    """
    Validate configuration dictionary.
    
    Parameters
    ----------
    config : dict
        Configuration to validate.
        
    Returns
    -------
    tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    # Check required keys
    required_keys = ["instagram", "tiktok"]
    for key in required_keys:
        if key not in config:
            return False, f"Missing required key: {key}"
        
        if not isinstance(config[key], list):
            return False, f"{key} must be a list"
            
        # Validate each username
        for username in config[key]:
            if not validate_username(username):
                return False, f"Invalid username: {username}"
    
    # Validate settings if present
    if "settings" in config:
        settings = config["settings"]
        
        if "instagram_posts_limit" in settings:
            if not validate_limit(settings["instagram_posts_limit"]):
                return False, "Invalid instagram_posts_limit"
                
        if "tiktok_videos_limit" in settings:
            if not validate_limit(settings["tiktok_videos_limit"]):
                return False, "Invalid tiktok_videos_limit"
    
    return True, None


def validate_api_token(token: str) -> bool:
    """
    Validate API token format.
    
    Parameters
    ----------
    token : str
        API token to validate.
        
    Returns
    -------
    bool
        True if valid, False otherwise.
    """
    # Basic validation - adjust pattern based on actual token format
    if not token or len(token) < 10:
        return False
    
    # Check for common placeholder values
    invalid_tokens = ["your_token_here", "xxx", "placeholder", "test"]
    if token.lower() in invalid_tokens:
        return False
    
    return True


def create_safe_path(base_dir: str, *parts: str) -> str:
    """
    Create a safe path that stays within base directory.
    
    Parameters
    ----------
    base_dir : str
        Base directory path.
    *parts : str
        Path components to join.
        
    Returns
    -------
    str
        Safe path within base directory.
        
    Raises
    ------
    ValueError
        If resulting path escapes base directory.
    """
    from pathlib import Path
    
    base = Path(base_dir).resolve()
    path = base.joinpath(*parts).resolve()
    
    # Ensure path is within base directory
    try:
        path.relative_to(base)
        return str(path)
    except ValueError:
        raise ValueError("Path traversal attempt detected")