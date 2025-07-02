"""
Data validation utilities for social media analytics.

This module provides validation functions for ensuring data quality
and consistency across platforms.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, ValidationError


def validate_url(url: str, platform: str) -> bool:
    """
    Validate social media URL format.
    
    Parameters
    ----------
    url : str
        URL to validate.
    platform : str
        Platform name ('instagram' or 'tiktok').
    
    Returns
    -------
    bool
        True if URL is valid for the platform.
    """
    platform = platform.lower()
    
    patterns = {
        'instagram': r'^https?://(?:www\.)?instagram\.com/(?:p|reel)/[\w-]+/?$',
        'tiktok': r'^https?://(?:www\.)?tiktok\.com/@[\w.]+/video/\d+/?$'
    }
    
    pattern = patterns.get(platform)
    if not pattern:
        return False
    
    return bool(re.match(pattern, url))


def validate_username(username: str, platform: str) -> bool:
    """
    Validate username format for platform.
    
    Parameters
    ----------
    username : str
        Username to validate.
    platform : str
        Platform name.
    
    Returns
    -------
    bool
        True if username is valid.
    """
    if not username:
        return False
    
    # Remove @ if present
    username = username.lstrip('@')
    
    # Platform-specific rules
    if platform.lower() == 'instagram':
        # Instagram: 1-30 chars, letters, numbers, periods, underscores
        pattern = r'^[\w.]{1,30}$'
    elif platform.lower() == 'tiktok':
        # TikTok: 2-24 chars, letters, numbers, underscores, periods
        pattern = r'^[\w.]{2,24}$'
    else:
        return False
    
    return bool(re.match(pattern, username))


def validate_hashtag(hashtag: str) -> bool:
    """
    Validate hashtag format.
    
    Parameters
    ----------
    hashtag : str
        Hashtag to validate.
    
    Returns
    -------
    bool
        True if hashtag is valid.
    """
    # Remove # if present
    hashtag = hashtag.lstrip('#')
    
    # Must contain only letters, numbers, underscores
    # Must not be empty
    return bool(re.match(r'^[\w]+$', hashtag)) and len(hashtag) > 0


def validate_timestamp_range(
    start_date: str,
    end_date: str
) -> Tuple[bool, Optional[str]]:
    """
    Validate timestamp range.
    
    Parameters
    ----------
    start_date : str
        Start date in ISO format.
    end_date : str
        End date in ISO format.
    
    Returns
    -------
    Tuple[bool, Optional[str]]
        (is_valid, error_message)
    """
    try:
        start = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        
        if start > end:
            return False, "Start date must be before end date"
        
        if start > datetime.now():
            return False, "Start date cannot be in the future"
        
        return True, None
        
    except (ValueError, AttributeError) as e:
        return False, f"Invalid date format: {str(e)}"


def validate_dataframe_schema(
    df: pd.DataFrame,
    required_columns: List[str]
) -> Tuple[bool, List[str]]:
    """
    Validate DataFrame has required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    required_columns : List[str]
        List of required column names.
    
    Returns
    -------
    Tuple[bool, List[str]]
        (is_valid, missing_columns)
    """
    missing_columns = [col for col in required_columns if col not in df.columns]
    return len(missing_columns) == 0, missing_columns


def validate_numeric_columns(
    df: pd.DataFrame,
    numeric_columns: List[str]
) -> Dict[str, Dict[str, Any]]:
    """
    Validate numeric columns for data quality issues.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to validate.
    numeric_columns : List[str]
        List of columns that should be numeric.
    
    Returns
    -------
    Dict[str, Dict[str, Any]]
        Validation results for each column.
    """
    results = {}
    
    for col in numeric_columns:
        if col not in df.columns:
            results[col] = {'exists': False}
            continue
        
        column_data = df[col]
        
        # Try to convert to numeric
        numeric_data = pd.to_numeric(column_data, errors='coerce')
        
        results[col] = {
            'exists': True,
            'is_numeric': column_data.dtype in ['int64', 'float64'],
            'has_nulls': column_data.isna().any(),
            'null_count': column_data.isna().sum(),
            'negative_values': (numeric_data < 0).sum(),
            'zero_values': (numeric_data == 0).sum(),
            'conversion_errors': numeric_data.isna().sum() - column_data.isna().sum()
        }
    
    return results


def validate_engagement_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate engagement data for anomalies.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with engagement data.
    
    Returns
    -------
    Dict[str, Any]
        Validation results and warnings.
    """
    warnings = []
    
    # Check if views < likes (impossible)
    if 'views_count' in df.columns and 'likes_count' in df.columns:
        invalid_views = df[df['views_count'] < df['likes_count']]
        if len(invalid_views) > 0:
            warnings.append(
                f"{len(invalid_views)} posts have more likes than views"
            )
    
    # Check for suspiciously high engagement rates
    if 'engagement_count' in df.columns and 'views_count' in df.columns:
        df['temp_engagement_rate'] = df['engagement_count'] / df['views_count'].clip(lower=1)
        suspicious = df[df['temp_engagement_rate'] > 0.5]  # >50% engagement
        if len(suspicious) > 0:
            warnings.append(
                f"{len(suspicious)} posts have suspiciously high engagement rates (>50%)"
            )
        df.drop('temp_engagement_rate', axis=1, inplace=True)
    
    # Check for missing timestamps
    if 'timestamp' in df.columns:
        missing_timestamps = df['timestamp'].isna().sum()
        if missing_timestamps > 0:
            warnings.append(f"{missing_timestamps} posts have missing timestamps")
    
    # Check for duplicate posts
    if 'post_id' in df.columns:
        duplicates = df['post_id'].duplicated().sum()
        if duplicates > 0:
            warnings.append(f"{duplicates} duplicate post IDs found")
    
    return {
        'is_valid': len(warnings) == 0,
        'warnings': warnings,
        'total_posts': len(df),
        'date_range': {
            'start': df['timestamp'].min() if 'timestamp' in df.columns else None,
            'end': df['timestamp'].max() if 'timestamp' in df.columns else None
        }
    }


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe file system usage.
    
    Parameters
    ----------
    filename : str
        Original filename.
    
    Returns
    -------
    str
        Sanitized filename.
    """
    # Remove invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing dots and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 200:
        sanitized = sanitized[:200]
    
    # Ensure not empty
    if not sanitized:
        sanitized = 'unnamed'
    
    return sanitized


def validate_api_response(
    response_data: List[Dict[str, Any]],
    platform: str
) -> Dict[str, Any]:
    """
    Validate API response data structure.
    
    Parameters
    ----------
    response_data : List[Dict[str, Any]]
        API response data.
    platform : str
        Platform name.
    
    Returns
    -------
    Dict[str, Any]
        Validation results.
    """
    if not isinstance(response_data, list):
        return {
            'is_valid': False,
            'error': 'Response data must be a list',
            'valid_count': 0
        }
    
    valid_count = 0
    errors = []
    
    # Platform-specific required fields
    required_fields = {
        'instagram': ['id', 'url', 'timestamp'],
        'tiktok': ['createTimeISO', 'webVideoUrl']
    }
    
    fields = required_fields.get(platform.lower(), [])
    
    for idx, item in enumerate(response_data):
        if not isinstance(item, dict):
            errors.append(f"Item {idx} is not a dictionary")
            continue
        
        missing_fields = [f for f in fields if f not in item]
        if missing_fields:
            errors.append(f"Item {idx} missing fields: {missing_fields}")
        else:
            valid_count += 1
    
    return {
        'is_valid': valid_count == len(response_data),
        'valid_count': valid_count,
        'total_count': len(response_data),
        'errors': errors[:10]  # Limit error messages
    }