"""
Base analytics class for social media account analysis.

This module provides the abstract base class for platform-specific
analytics implementations.
"""

import json
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
from pydantic import BaseModel

# Try to import pyarrow, but don't fail if it's not available
try:
    import pyarrow as pa
    import pyarrow.parquet as pq
    PYARROW_AVAILABLE = True
except ImportError:
    PYARROW_AVAILABLE = False

from ..metrics import calculate_all_metrics
from ..transform import create_unified_schema

logger = logging.getLogger(__name__)


class AccountInfo(BaseModel):
    """Account information model."""
    
    username: str
    user_id: Optional[str] = None
    full_name: Optional[str] = None
    followers_count: Optional[int] = None
    following_count: Optional[int] = None
    posts_count: Optional[int] = None
    bio: Optional[str] = None
    verified: bool = False
    platform: str
    scraped_at: datetime = datetime.now()
    additional_info: Dict[str, Any] = {}


class AnalyticsReport(BaseModel):
    """Analytics report model."""
    
    account_info: AccountInfo
    metrics: Dict[str, Any]
    time_period: Dict[str, str]
    generated_at: datetime = datetime.now()
    platform_specific_metrics: Dict[str, Any] = {}
    insights: List[str] = []
    warnings: List[str] = []


class AccountAnalytics(ABC):
    """
    Abstract base class for social media account analytics.
    
    Provides common functionality for analyzing social media accounts
    across different platforms.
    """
    
    def __init__(
        self,
        data: pd.DataFrame,
        account_info: Union[Dict[str, Any], AccountInfo]
    ) -> None:
        """
        Initialize AccountAnalytics.
        
        Parameters
        ----------
        data : pd.DataFrame
            Normalized data for the account.
        account_info : Union[Dict[str, Any], AccountInfo]
            Account information.
        """
        self.data = data
        
        if isinstance(account_info, dict):
            self.account_info = AccountInfo(**account_info)
        else:
            self.account_info = account_info
        
        self._metrics = None
        self._report = None
        
        logger.info(
            f"Initialized {self.__class__.__name__} for @{self.account_info.username}"
        )
    
    @abstractmethod
    def calculate_platform_specific_metrics(self) -> Dict[str, Any]:
        """
        Calculate platform-specific metrics.
        
        Returns
        -------
        Dict[str, Any]
            Platform-specific metrics.
        """
        pass
    
    @abstractmethod
    def generate_insights(self) -> List[str]:
        """
        Generate platform-specific insights.
        
        Returns
        -------
        List[str]
            List of insights about the account.
        """
        pass
    
    def calculate_metrics(self, force: bool = False) -> Dict[str, Any]:
        """
        Calculate all metrics for the account.
        
        Parameters
        ----------
        force : bool
            Force recalculation even if metrics are cached.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary of calculated metrics.
        """
        if self._metrics is not None and not force:
            return self._metrics
        
        # Calculate standard metrics
        standard_metrics = calculate_all_metrics(
            self.data,
            self.account_info.followers_count
        )
        
        # Calculate platform-specific metrics
        platform_metrics = self.calculate_platform_specific_metrics()
        
        # Combine metrics
        self._metrics = {
            **standard_metrics,
            'platform_specific': platform_metrics
        }
        
        return self._metrics
    
    def generate_report(self) -> AnalyticsReport:
        """
        Generate comprehensive analytics report.
        
        Returns
        -------
        AnalyticsReport
            Complete analytics report.
        """
        if self._report is not None:
            return self._report
        
        # Calculate metrics
        metrics = self.calculate_metrics()
        
        # Determine time period
        if len(self.data) > 0:
            time_period = {
                'start': self.data['timestamp'].min().isoformat(),
                'end': self.data['timestamp'].max().isoformat(),
                'days': str((self.data['timestamp'].max() - self.data['timestamp'].min()).days + 1)
            }
        else:
            time_period = {
                'start': None,
                'end': None,
                'days': '0'
            }
        
        # Generate insights
        insights = self.generate_insights()
        
        # Check for warnings
        warnings = self._check_data_quality()
        
        # Create report
        self._report = AnalyticsReport(
            account_info=self.account_info,
            metrics=metrics,
            time_period=time_period,
            platform_specific_metrics=metrics.get('platform_specific', {}),
            insights=insights,
            warnings=warnings
        )
        
        return self._report
    
    def _check_data_quality(self) -> List[str]:
        """
        Check data quality and generate warnings.
        
        Returns
        -------
        List[str]
            List of data quality warnings.
        """
        warnings = []
        
        if len(self.data) == 0:
            warnings.append("No data available for analysis")
            return warnings
        
        # Check for missing values
        missing_cols = []
        for col in ['likes_count', 'views_count', 'comments_count']:
            if col in self.data.columns:
                missing_pct = (self.data[col].isna().sum() / len(self.data)) * 100
                if missing_pct > 10:
                    missing_cols.append(f"{col} ({missing_pct:.1f}% missing)")
        
        if missing_cols:
            warnings.append(f"Significant missing data in: {', '.join(missing_cols)}")
        
        # Check for data recency
        latest_post = self.data['timestamp'].max()
        # Convert to timezone-naive for comparison
        if latest_post.tz is not None:
            latest_post = latest_post.tz_localize(None)
        days_since_latest = (datetime.now() - latest_post).days
        
        if days_since_latest > 30:
            warnings.append(f"Latest post is {days_since_latest} days old")
        
        # Check for sufficient data
        if len(self.data) < 10:
            warnings.append(f"Limited data available ({len(self.data)} posts)")
        
        return warnings
    
    def export_metrics(
        self,
        filepath: Union[str, Path],
        format: str = 'json'
    ) -> None:
        """
        Export metrics to file.
        
        Parameters
        ----------
        filepath : Union[str, Path]
            Output file path.
        format : str
            Export format ('json', 'csv', 'parquet').
        """
        filepath = Path(filepath)
        report = self.generate_report()
        
        if format == 'json':
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(report.model_dump(), f, indent=2, default=str)
            logger.info(f"Exported metrics to {filepath}")
            
        elif format == 'csv':
            # Flatten metrics for CSV export
            flat_metrics = self._flatten_dict(report.metrics)
            metrics_df = pd.DataFrame([flat_metrics])
            metrics_df.to_csv(filepath, index=False)
            logger.info(f"Exported metrics to {filepath}")
            
        elif format == 'parquet':
            if PYARROW_AVAILABLE:
                # Export full data with metrics as metadata
                table = pa.Table.from_pandas(self.data)
                metadata = {
                    b'metrics': json.dumps(report.metrics, default=str).encode('utf-8'),
                    b'account_info': json.dumps(
                        report.account_info.model_dump(), 
                        default=str
                    ).encode('utf-8')
                }
                table = table.replace_schema_metadata(metadata)
                pq.write_table(table, filepath)
                logger.info(f"Exported data and metrics to {filepath}")
            else:
                logger.warning("PyArrow not available. Falling back to CSV export.")
                # Fall back to CSV
                self.export_metrics(filepath.with_suffix('.csv'), format='csv')
            
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _flatten_dict(
        self,
        d: Dict[str, Any],
        parent_key: str = '',
        sep: str = '.'
    ) -> Dict[str, Any]:
        """
        Flatten nested dictionary.
        
        Parameters
        ----------
        d : Dict[str, Any]
            Dictionary to flatten.
        parent_key : str
            Parent key for recursion.
        sep : str
            Separator for nested keys.
        
        Returns
        -------
        Dict[str, Any]
            Flattened dictionary.
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(
                    self._flatten_dict(v, new_key, sep=sep).items()
                )
            else:
                items.append((new_key, v))
        return dict(items)
    
    def get_top_posts(
        self,
        n: int = 10,
        by: str = 'engagement_count'
    ) -> pd.DataFrame:
        """
        Get top performing posts.
        
        Parameters
        ----------
        n : int
            Number of posts to return.
        by : str
            Metric to sort by.
        
        Returns
        -------
        pd.DataFrame
            Top posts sorted by specified metric.
        """
        if by not in self.data.columns:
            raise ValueError(f"Column '{by}' not found in data")
        
        return self.data.nlargest(n, by)
    
    def get_posting_patterns(self) -> Dict[str, Any]:
        """
        Analyze posting patterns.
        
        Returns
        -------
        Dict[str, Any]
            Dictionary with posting pattern analysis.
        """
        if len(self.data) == 0:
            return {}
        
        # Hour distribution
        hour_dist = self.data['hour'].value_counts().sort_index()
        
        # Day of week distribution
        dow_dist = self.data['day_name'].value_counts()
        
        # Weekend vs weekday
        weekend_pct = (self.data['is_weekend'].sum() / len(self.data)) * 100
        
        # Best performing times
        avg_by_hour = self.data.groupby('hour')['engagement_count'].mean()
        best_hour = avg_by_hour.idxmax()
        
        avg_by_dow = self.data.groupby('day_name')['engagement_count'].mean()
        best_day = avg_by_dow.idxmax()
        
        return {
            'hour_distribution': hour_dist.to_dict(),
            'day_distribution': dow_dist.to_dict(),
            'weekend_percentage': float(weekend_pct),
            'best_hour': int(best_hour) if pd.notna(best_hour) else None,
            'best_day': str(best_day) if pd.notna(best_day) else None,
            'avg_engagement_by_hour': avg_by_hour.to_dict(),
            'avg_engagement_by_day': avg_by_dow.to_dict()
        }
    
    def get_hashtag_analysis(self) -> Dict[str, Any]:
        """
        Analyze hashtag usage and performance.
        
        Returns
        -------
        Dict[str, Any]
            Hashtag analysis results.
        """
        # Count hashtag usage
        all_hashtags = []
        for hashtags in self.data['hashtags']:
            if isinstance(hashtags, list):
                all_hashtags.extend(hashtags)
        
        if not all_hashtags:
            return {
                'total_unique_hashtags': 0,
                'avg_hashtags_per_post': 0,
                'top_hashtags': {},
                'hashtag_effectiveness': {}
            }
        
        hashtag_counts = pd.Series(all_hashtags).value_counts()
        
        # Calculate effectiveness
        from ..metrics import hashtag_effectiveness_score
        effectiveness = hashtag_effectiveness_score(self.data)
        
        return {
            'total_unique_hashtags': len(hashtag_counts),
            'avg_hashtags_per_post': float(self.data['hashtag_count'].mean()),
            'top_hashtags': dict(hashtag_counts.head(20)),
            'hashtag_effectiveness': dict(list(effectiveness.items())[:20])
        }
    
    def compare_periods(
        self,
        period_days: int = 30
    ) -> Dict[str, Any]:
        """
        Compare metrics between time periods.
        
        Parameters
        ----------
        period_days : int
            Number of days for each period.
        
        Returns
        -------
        Dict[str, Any]
            Comparison of metrics between periods.
        """
        if len(self.data) < 2:
            return {}
        
        # Sort by timestamp
        sorted_data = self.data.sort_values('timestamp')
        
        # Find midpoint
        midpoint = sorted_data['timestamp'].median()
        
        # Split data
        first_half = sorted_data[sorted_data['timestamp'] < midpoint]
        second_half = sorted_data[sorted_data['timestamp'] >= midpoint]
        
        if len(first_half) == 0 or len(second_half) == 0:
            return {}
        
        # Calculate metrics for each half
        metrics_first = {
            'avg_engagement': float(first_half['engagement_count'].mean()),
            'avg_views': float(first_half['views_count'].mean()),
            'post_count': len(first_half)
        }
        
        metrics_second = {
            'avg_engagement': float(second_half['engagement_count'].mean()),
            'avg_views': float(second_half['views_count'].mean()),
            'post_count': len(second_half)
        }
        
        # Calculate growth
        growth = {}
        for metric in ['avg_engagement', 'avg_views']:
            if metrics_first[metric] > 0:
                growth_pct = (
                    (metrics_second[metric] - metrics_first[metric]) / 
                    metrics_first[metric]
                ) * 100
                growth[f'{metric}_growth'] = float(growth_pct)
        
        return {
            'first_period': metrics_first,
            'second_period': metrics_second,
            'growth': growth
        }