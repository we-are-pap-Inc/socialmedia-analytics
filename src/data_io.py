"""
Data I/O module for managing analytics runs and cached data.

This module provides functions to save and load analytics data
in a structured directory format with metadata tracking.
"""

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


def get_run_dir(timestamp: datetime) -> Path:
    """
    Get the directory path for a specific run based on timestamp.
    
    Parameters
    ----------
    timestamp : datetime
        The timestamp of the run.
    
    Returns
    -------
    Path
        Path to the run directory.
    """
    # Format: data/2025-07-02T20-15-33/
    timestamp_str = timestamp.strftime("%Y-%m-%dT%H-%M-%S")
    run_dir = Path("data") / timestamp_str
    return run_dir


def get_latest_run_dir() -> Optional[Path]:
    """
    Get the directory of the most recent run.
    
    Returns
    -------
    Optional[Path]
        Path to the latest run directory, or None if no runs exist.
    """
    data_dir = Path("data")
    
    if not data_dir.exists():
        logger.warning("Data directory does not exist")
        return None
    
    # Find all run directories (they follow timestamp pattern)
    run_dirs = []
    for dir_path in data_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.count('-') == 4:  # YYYY-MM-DDTHH-MM-SS
            try:
                # Try to parse the timestamp
                datetime.strptime(dir_path.name, "%Y-%m-%dT%H-%M-%S")
                run_dirs.append(dir_path)
            except ValueError:
                continue
    
    if not run_dirs:
        logger.warning("No valid run directories found")
        return None
    
    # Sort by name (which is timestamp) and get the latest
    latest_dir = sorted(run_dirs, key=lambda x: x.name)[-1]
    logger.info(f"Found latest run directory: {latest_dir}")
    return latest_dir


def save_run(
    df: pd.DataFrame,
    metrics: Dict[str, Any],
    dest: Path,
    platform: str,
    username: str
) -> None:
    """
    Save a run's data and metrics to the specified directory.
    
    Parameters
    ----------
    df : pd.DataFrame
        The analytics data DataFrame.
    metrics : Dict[str, Any]
        Calculated metrics dictionary.
    dest : Path
        Destination directory path.
    platform : str
        Social media platform name.
    username : str
        Account username.
    """
    # Ensure directory exists
    dest.mkdir(parents=True, exist_ok=True)
    
    # Create filename base
    filename_base = f"{platform}_{username}"
    
    # Save DataFrame as Parquet
    parquet_path = dest / f"{filename_base}_data.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)
    logger.info(f"Saved data to {parquet_path}")
    
    # Save metrics as JSON
    json_path = dest / f"{filename_base}_metrics.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2, default=str)
    logger.info(f"Saved metrics to {json_path}")
    
    # Also save as CSV for compatibility
    csv_path = dest / f"{filename_base}_data.csv"
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV to {csv_path}")


def load_run(path: Path, platform: str, username: str) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Load a run's data and metrics from the specified directory.
    
    Parameters
    ----------
    path : Path
        Directory path containing the run data.
    platform : str
        Social media platform name.
    username : str
        Account username.
    
    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Tuple of (DataFrame, metrics dictionary).
    
    Raises
    ------
    FileNotFoundError
        If required files are not found.
    """
    filename_base = f"{platform}_{username}"
    
    # Load DataFrame from Parquet (preferred) or CSV
    parquet_path = path / f"{filename_base}_data.parquet"
    csv_path = path / f"{filename_base}_data.csv"
    
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        logger.info(f"Loaded data from {parquet_path}")
    elif csv_path.exists():
        df = pd.read_csv(csv_path)
        logger.info(f"Loaded data from {csv_path}")
    else:
        raise FileNotFoundError(f"No data file found for {platform} @{username} in {path}")
    
    # Load metrics from JSON
    json_path = path / f"{filename_base}_metrics.json"
    if json_path.exists():
        with open(json_path, 'r', encoding='utf-8') as f:
            metrics = json.load(f)
        logger.info(f"Loaded metrics from {json_path}")
    else:
        logger.warning(f"No metrics file found at {json_path}, using empty dict")
        metrics = {}
    
    return df, metrics


def append_run_metadata(
    timestamp: datetime,
    accounts: List[Tuple[str, str]],  # List of (platform, username) tuples
    duration_seconds: float,
    status: str = "completed"
) -> None:
    """
    Append run metadata to the master runs.csv file.
    
    Parameters
    ----------
    timestamp : datetime
        Run timestamp.
    accounts : List[Tuple[str, str]]
        List of (platform, username) tuples that were analyzed.
    duration_seconds : float
        Duration of the run in seconds.
    status : str
        Status of the run (completed, failed, partial).
    """
    runs_file = Path("data/runs.csv")
    runs_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Check if file exists to determine if we need headers
    file_exists = runs_file.exists()
    
    # Prepare row data
    row_data = {
        'timestamp': timestamp.isoformat(),
        'run_dir': get_run_dir(timestamp).name,
        'accounts': json.dumps([f"{p}:@{u}" for p, u in accounts]),
        'account_count': len(accounts),
        'duration_seconds': round(duration_seconds, 2),
        'status': status
    }
    
    # Write to CSV
    with open(runs_file, 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)
    
    logger.info(f"Appended run metadata to {runs_file}")


def get_run_history() -> pd.DataFrame:
    """
    Get the history of all runs from runs.csv.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with run history, or empty DataFrame if no history.
    """
    runs_file = Path("data/runs.csv")
    
    if not runs_file.exists():
        logger.warning("No runs.csv file found")
        return pd.DataFrame()
    
    df = pd.read_csv(runs_file)
    
    # Parse timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Parse accounts JSON
    if 'accounts' in df.columns:
        df['accounts'] = df['accounts'].apply(json.loads)
    
    return df


def save_combined_report(
    all_results: List[Dict[str, Any]],
    dest: Path
) -> None:
    """
    Save a combined report for all accounts.
    
    Parameters
    ----------
    all_results : List[Dict[str, Any]]
        List of analysis results for all accounts.
    dest : Path
        Destination directory.
    """
    dest.mkdir(parents=True, exist_ok=True)
    
    # Save as JSON
    json_path = dest / "all_accounts_summary.json"
    summary_data = {
        'generated_at': datetime.now().isoformat(),
        'total_accounts': len(all_results),
        'accounts': all_results
    }
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    logger.info(f"Saved combined report to {json_path}")


def load_combined_report(path: Path) -> Optional[Dict[str, Any]]:
    """
    Load the combined report from a run directory.
    
    Parameters
    ----------
    path : Path
        Directory path containing the run data.
    
    Returns
    -------
    Optional[Dict[str, Any]]
        Combined report data or None if not found.
    """
    json_path = path / "all_accounts_summary.json"
    
    if not json_path.exists():
        logger.warning(f"No combined report found at {json_path}")
        return None
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    logger.info(f"Loaded combined report from {json_path}")
    return data
    
    # Create combined CSV
    csv_rows = []
    for result in all_results:
        if result and 'metrics' in result:
            row = {
                'platform': result['platform'],
                'username': result['username'],
                'total_posts': result['metrics'].get('total_posts', 0),
                'total_engagement': result['metrics'].get('total_engagement', 0),
                'average_views': result['metrics'].get('average_views', 0),
                'engagement_rate': result['metrics'].get('engagement_rate', 0),
                'viral_velocity_score': result['metrics'].get('viral_velocity_score', 0)
            }
            
            # Add profile data if available
            if result.get('profile_data'):
                row['followers_count'] = result['profile_data'].get('followersCount', 'N/A')
            
            csv_rows.append(row)
    
    if csv_rows:
        csv_df = pd.DataFrame(csv_rows)
        csv_path = dest / "all_accounts_summary.csv"
        csv_df.to_csv(csv_path, index=False)
        logger.info(f"Saved combined CSV to {csv_path}")


def cleanup_old_runs(keep_latest: int = 10) -> None:
    """
    Clean up old run directories, keeping only the latest N runs.
    
    Parameters
    ----------
    keep_latest : int
        Number of latest runs to keep.
    """
    data_dir = Path("data")
    
    if not data_dir.exists():
        return
    
    # Find all run directories
    run_dirs = []
    for dir_path in data_dir.iterdir():
        if dir_path.is_dir() and dir_path.name.count('-') == 4:
            try:
                datetime.strptime(dir_path.name, "%Y-%m-%dT%H-%M-%S")
                run_dirs.append(dir_path)
            except ValueError:
                continue
    
    if len(run_dirs) <= keep_latest:
        return
    
    # Sort by timestamp and get directories to remove
    sorted_dirs = sorted(run_dirs, key=lambda x: x.name)
    dirs_to_remove = sorted_dirs[:-keep_latest]
    
    for dir_path in dirs_to_remove:
        try:
            import shutil
            shutil.rmtree(dir_path)
            logger.info(f"Removed old run directory: {dir_path}")
        except Exception as e:
            logger.error(f"Failed to remove {dir_path}: {str(e)}")