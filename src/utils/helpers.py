"""
Helper utilities for social media analytics.

This module provides general utility functions used across the toolkit.
"""

import hashlib
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Parameters
    ----------
    config_path : Union[str, Path]
        Path to configuration file.
    
    Returns
    -------
    Dict[str, Any]
        Configuration dictionary.
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        logger.warning(f"Config file not found: {config_path}")
        return {}
    
    try:
        with open(config_path, 'r') as f:
            if config_path.suffix in ['.yml', '.yaml']:
                return yaml.safe_load(f)
            elif config_path.suffix == '.json':
                return json.load(f)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        return {}


def format_number(num: Union[int, float], precision: int = 0) -> str:
    """
    Format number with K/M/B suffixes.
    
    Parameters
    ----------
    num : Union[int, float]
        Number to format.
    precision : int
        Decimal precision.
    
    Returns
    -------
    str
        Formatted number string.
    """
    if pd.isna(num):
        return "N/A"
    
    num = float(num)
    
    if num >= 1_000_000_000:
        return f"{num/1_000_000_000:.{precision}f}B"
    elif num >= 1_000_000:
        return f"{num/1_000_000:.{precision}f}M"
    elif num >= 1_000:
        return f"{num/1_000:.{precision}f}K"
    else:
        return f"{num:.{precision}f}"


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Parameters
    ----------
    seconds : float
        Duration in seconds.
    
    Returns
    -------
    str
        Formatted duration string.
    """
    if pd.isna(seconds) or seconds < 0:
        return "N/A"
    
    if seconds < 60:
        return f"{int(seconds)}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


def calculate_date_range(
    df: pd.DataFrame,
    date_column: str = 'timestamp'
) -> Dict[str, Any]:
    """
    Calculate date range statistics from DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with date column.
    date_column : str
        Name of the date column.
    
    Returns
    -------
    Dict[str, Any]
        Date range statistics.
    """
    if date_column not in df.columns or len(df) == 0:
        return {
            'start': None,
            'end': None,
            'days': 0,
            'posts_per_day': 0
        }
    
    dates = pd.to_datetime(df[date_column])
    start = dates.min()
    end = dates.max()
    days = (end - start).days + 1
    
    return {
        'start': start.isoformat(),
        'end': end.isoformat(),
        'days': days,
        'posts_per_day': len(df) / days if days > 0 else 0,
        'date_coverage': _calculate_date_coverage(dates)
    }


def _calculate_date_coverage(dates: pd.Series) -> float:
    """Calculate percentage of days with posts."""
    if len(dates) == 0:
        return 0.0
    
    unique_days = dates.dt.date.nunique()
    total_days = (dates.max() - dates.min()).days + 1
    
    return (unique_days / total_days) * 100 if total_days > 0 else 0.0


def generate_cache_key(
    data: Union[str, Dict[str, Any]],
    prefix: str = ''
) -> str:
    """
    Generate cache key from data.
    
    Parameters
    ----------
    data : Union[str, Dict[str, Any]]
        Data to generate key from.
    prefix : str
        Optional prefix for the key.
    
    Returns
    -------
    str
        Cache key.
    """
    if isinstance(data, dict):
        data = json.dumps(data, sort_keys=True)
    
    hash_obj = hashlib.md5(data.encode())
    key = hash_obj.hexdigest()
    
    return f"{prefix}_{key}" if prefix else key


def ensure_directory(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if not.
    
    Parameters
    ----------
    path : Union[str, Path]
        Directory path.
    
    Returns
    -------
    Path
        Path object.
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_time_ago(timestamp: datetime) -> str:
    """
    Get human-readable time ago string.
    
    Parameters
    ----------
    timestamp : datetime
        Past timestamp.
    
    Returns
    -------
    str
        Time ago string (e.g., "2 hours ago").
    """
    if pd.isna(timestamp):
        return "Unknown"
    
    now = datetime.now()
    if timestamp.tzinfo:
        now = now.replace(tzinfo=timestamp.tzinfo)
    
    delta = now - timestamp
    
    if delta.days > 365:
        years = delta.days // 365
        return f"{years} year{'s' if years > 1 else ''} ago"
    elif delta.days > 30:
        months = delta.days // 30
        return f"{months} month{'s' if months > 1 else ''} ago"
    elif delta.days > 0:
        return f"{delta.days} day{'s' if delta.days > 1 else ''} ago"
    elif delta.seconds > 3600:
        hours = delta.seconds // 3600
        return f"{hours} hour{'s' if hours > 1 else ''} ago"
    elif delta.seconds > 60:
        minutes = delta.seconds // 60
        return f"{minutes} minute{'s' if minutes > 1 else ''} ago"
    else:
        return "Just now"


def batch_process(
    items: List[Any],
    batch_size: int = 100
) -> List[List[Any]]:
    """
    Split items into batches for processing.
    
    Parameters
    ----------
    items : List[Any]
        Items to batch.
    batch_size : int
        Size of each batch.
    
    Returns
    -------
    List[List[Any]]
        List of batches.
    """
    return [
        items[i:i + batch_size]
        for i in range(0, len(items), batch_size)
    ]


def merge_dataframes(
    dfs: List[pd.DataFrame],
    on: Optional[Union[str, List[str]]] = None,
    how: str = 'outer'
) -> pd.DataFrame:
    """
    Merge multiple DataFrames with conflict resolution.
    
    Parameters
    ----------
    dfs : List[pd.DataFrame]
        List of DataFrames to merge.
    on : Optional[Union[str, List[str]]]
        Column(s) to merge on.
    how : str
        Merge method.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame.
    """
    if not dfs:
        return pd.DataFrame()
    
    if len(dfs) == 1:
        return dfs[0]
    
    # Start with first DataFrame
    result = dfs[0]
    
    for df in dfs[1:]:
        if on:
            result = pd.merge(result, df, on=on, how=how, suffixes=('', '_dup'))
            # Remove duplicate columns
            dup_cols = [col for col in result.columns if col.endswith('_dup')]
            result.drop(columns=dup_cols, inplace=True)
        else:
            result = pd.concat([result, df], ignore_index=True)
    
    return result


def export_to_excel(
    data: Dict[str, pd.DataFrame],
    filepath: Union[str, Path],
    include_summary: bool = True
) -> None:
    """
    Export multiple DataFrames to Excel with formatting.
    
    Parameters
    ----------
    data : Dict[str, pd.DataFrame]
        Dictionary of sheet names to DataFrames.
    filepath : Union[str, Path]
        Output file path.
    include_summary : bool
        Whether to include a summary sheet.
    """
    filepath = Path(filepath)
    
    with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
        # Write data sheets
        for sheet_name, df in data.items():
            df.to_excel(writer, sheet_name=sheet_name[:31], index=False)
        
        # Add summary sheet if requested
        if include_summary:
            summary_data = []
            for sheet_name, df in data.items():
                summary_data.append({
                    'Sheet': sheet_name,
                    'Rows': len(df),
                    'Columns': len(df.columns),
                    'Memory (KB)': df.memory_usage(deep=True).sum() / 1024
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
    
    logger.info(f"Exported data to {filepath}")


def clean_text(text: str, remove_urls: bool = True) -> str:
    """
    Clean text for analysis.
    
    Parameters
    ----------
    text : str
        Text to clean.
    remove_urls : bool
        Whether to remove URLs.
    
    Returns
    -------
    str
        Cleaned text.
    """
    if pd.isna(text):
        return ""
    
    text = str(text)
    
    # Remove URLs if requested
    if remove_urls:
        import re
        text = re.sub(r'https?://\S+|www\.\S+', '', text)
    
    # Remove extra whitespace
    text = ' '.join(text.split())
    
    return text.strip()


def calculate_growth_rate(
    values: List[float],
    periods: int = 1
) -> float:
    """
    Calculate growth rate over periods.
    
    Parameters
    ----------
    values : List[float]
        Time series values.
    periods : int
        Number of periods for growth calculation.
    
    Returns
    -------
    float
        Growth rate as percentage.
    """
    if len(values) < periods + 1:
        return 0.0
    
    start_val = values[-(periods + 1)]
    end_val = values[-1]
    
    if start_val == 0:
        return 0.0
    
    return ((end_val - start_val) / start_val) * 100


def get_platform_colors() -> Dict[str, str]:
    """
    Get platform-specific color schemes.
    
    Returns
    -------
    Dict[str, str]
        Platform color mappings.
    """
    return {
        'instagram': '#E4405F',
        'tiktok': '#000000',
        'facebook': '#1877F2',
        'twitter': '#1DA1F2',
        'youtube': '#FF0000',
        'linkedin': '#0A66C2'
    }