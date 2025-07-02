"""
Metrics calculation module for social media analytics.

This module provides functions to calculate standard and creative metrics
for analyzing social media performance across platforms.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# Standard Metrics

def total_engagement(df: pd.DataFrame) -> int:
    """
    Calculate total engagement across all posts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    int
        Total engagement count (likes + comments + shares).
    """
    return int(df['engagement_count'].sum())


def average_views(df: pd.DataFrame) -> float:
    """
    Calculate average views per post.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    float
        Average views per post.
    """
    views = df['views_count']
    return float(views.mean()) if len(views) > 0 else 0.0


def average_likes(df: pd.DataFrame) -> float:
    """
    Calculate average likes per post.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    float
        Average likes per post.
    """
    return float(df['likes_count'].mean()) if len(df) > 0 else 0.0


def average_comments(df: pd.DataFrame) -> float:
    """
    Calculate average comments per post.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    float
        Average comments per post.
    """
    return float(df['comments_count'].mean()) if len(df) > 0 else 0.0


def total_likes(df: pd.DataFrame) -> int:
    """
    Calculate total likes across all posts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    int
        Total likes count.
    """
    return int(df['likes_count'].sum())


def total_views(df: pd.DataFrame) -> int:
    """
    Calculate total views across all posts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    int
        Total views count.
    """
    return int(df['views_count'].sum())


def total_comments(df: pd.DataFrame) -> int:
    """
    Calculate total comments across all posts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    int
        Total comments count.
    """
    return int(df['comments_count'].sum())


def max_likes(df: pd.DataFrame) -> int:
    """
    Get maximum likes on a single post.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    int
        Maximum likes count.
    """
    return int(df['likes_count'].max()) if len(df) > 0 else 0


def max_views(df: pd.DataFrame) -> int:
    """
    Get maximum views on a single post.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    int
        Maximum views count.
    """
    return int(df['views_count'].max()) if len(df) > 0 else 0


def get_post_with_max_views(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Get post with maximum views.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    Tuple[int, str]
        Tuple of (views_count, post_url).
    """
    if len(df) == 0:
        return (0, "")
    
    max_idx = df['views_count'].idxmax()
    post = df.loc[max_idx]
    return (int(post['views_count']), str(post.get('url', '')))


def get_post_with_min_views(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Get post with minimum views.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    Tuple[int, str]
        Tuple of (views_count, post_url).
    """
    if len(df) == 0:
        return (0, "")
    
    min_idx = df['views_count'].idxmin()
    post = df.loc[min_idx]
    return (int(post['views_count']), str(post.get('url', '')))


def get_post_with_max_likes(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Get post with maximum likes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    Tuple[int, str]
        Tuple of (likes_count, post_url).
    """
    if len(df) == 0:
        return (0, "")
    
    max_idx = df['likes_count'].idxmax()
    post = df.loc[max_idx]
    return (int(post['likes_count']), str(post.get('url', '')))


def get_post_with_min_likes(df: pd.DataFrame) -> Tuple[int, str]:
    """
    Get post with minimum likes.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    Tuple[int, str]
        Tuple of (likes_count, post_url).
    """
    if len(df) == 0:
        return (0, "")
    
    min_idx = df['likes_count'].idxmin()
    post = df.loc[min_idx]
    return (int(post['likes_count']), str(post.get('url', '')))


def average_views_videos(df: pd.DataFrame) -> float:
    """
    Calculate average views for video posts only.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    float
        Average views per video post.
    """
    video_df = df[df['is_video'] == True] if 'is_video' in df.columns else pd.DataFrame()
    return float(video_df['views_count'].mean()) if len(video_df) > 0 else 0.0


def average_views_non_videos(df: pd.DataFrame) -> float:
    """
    Calculate average views for non-video posts.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    float
        Average views per non-video post.
    """
    non_video_df = df[df['is_video'] == False] if 'is_video' in df.columns else df
    return float(non_video_df['views_count'].mean()) if len(non_video_df) > 0 else 0.0


def percentile_views(df: pd.DataFrame, percentile: int = 50) -> float:
    """
    Calculate percentile of views.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    percentile : int
        Percentile to calculate (0-100).
    
    Returns
    -------
    float
        Views at specified percentile.
    """
    if len(df) == 0:
        return 0.0
    
    return float(np.percentile(df['views_count'], percentile))


def engagement_rate(
    df: pd.DataFrame, 
    followers_count: Optional[int] = None
) -> float:
    """
    Calculate engagement rate.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    followers_count : Optional[int]
        Number of followers. If not provided, uses views as denominator.
    
    Returns
    -------
    float
        Engagement rate as percentage.
    """
    if len(df) == 0:
        return 0.0
    
    total_engagements = df['engagement_count'].sum()
    
    if followers_count and followers_count > 0:
        # Traditional engagement rate: (engagements / followers) * 100
        rate = (total_engagements / (len(df) * followers_count)) * 100
    else:
        # Alternative: engagement / views ratio
        total_views = df['views_count'].sum()
        if total_views > 0:
            rate = (total_engagements / total_views) * 100
        else:
            rate = 0.0
    
    return float(rate)


def posting_frequency(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate posting frequency metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    Dict[str, float]
        Dictionary with frequency metrics.
    """
    if len(df) == 0:
        return {
            'posts_per_day': 0.0,
            'posts_per_week': 0.0,
            'posts_per_month': 0.0,
            'avg_hours_between_posts': 0.0
        }
    
    # Sort by timestamp
    sorted_df = df.sort_values('timestamp')
    
    # Calculate date range
    date_range = (sorted_df['timestamp'].max() - sorted_df['timestamp'].min()).days + 1
    
    # Calculate frequencies
    posts_per_day = len(df) / date_range if date_range > 0 else 0
    posts_per_week = posts_per_day * 7
    posts_per_month = posts_per_day * 30
    
    # Calculate average time between posts
    if len(sorted_df) > 1:
        time_diffs = sorted_df['timestamp'].diff().dropna()
        avg_hours = time_diffs.mean().total_seconds() / 3600
    else:
        avg_hours = 0.0
    
    return {
        'posts_per_day': float(posts_per_day),
        'posts_per_week': float(posts_per_week),
        'posts_per_month': float(posts_per_month),
        'avg_hours_between_posts': float(avg_hours)
    }


# Creative Metrics

def viral_velocity_score(df: pd.DataFrame, hours: int = 24) -> float:
    """
    Calculate viral velocity score based on engagement growth rate.
    
    Measures how quickly content gains traction in the first N hours.
    Score = (Early Engagement Rate) * (Growth Factor) * 100
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    hours : int
        Number of hours to consider for early engagement.
    
    Returns
    -------
    float
        Viral velocity score (0-100+).
    """
    if len(df) == 0:
        return 0.0
    
    # For this example, we'll use a simplified approach
    # In reality, you'd need timestamp data for each engagement
    
    # Calculate engagement rate variance as proxy for velocity
    engagement_rates = []
    for _, post in df.iterrows():
        if post['views_count'] > 0:
            rate = (post['engagement_count'] / post['views_count']) * 100
            engagement_rates.append(rate)
    
    if not engagement_rates:
        return 0.0
    
    # High variance in engagement rates suggests some posts go viral
    variance = np.var(engagement_rates)
    mean_rate = np.mean(engagement_rates)
    
    # Score based on variance and mean
    # Higher variance and higher mean = higher viral potential
    score = (variance ** 0.5) * (mean_rate ** 0.5)
    
    return float(min(score, 100))  # Cap at 100


def content_consistency_index(df: pd.DataFrame) -> float:
    """
    Calculate content consistency index (0-1).
    
    Measures how regularly content is posted.
    1 = perfectly consistent, 0 = highly irregular
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    float
        Consistency index between 0 and 1.
    """
    if len(df) < 2:
        return 0.0
    
    # Sort by timestamp
    sorted_df = df.sort_values('timestamp')
    
    # Calculate time differences between posts
    time_diffs = sorted_df['timestamp'].diff().dropna()
    
    if len(time_diffs) == 0:
        return 0.0
    
    # Convert to hours
    hours_between = time_diffs.dt.total_seconds() / 3600
    
    # Calculate coefficient of variation (CV)
    # Lower CV = more consistent
    mean_hours = hours_between.mean()
    std_hours = hours_between.std()
    
    if mean_hours == 0:
        return 0.0
    
    cv = std_hours / mean_hours
    
    # Convert to 0-1 scale (1 = consistent)
    # Using exponential decay function
    consistency = np.exp(-cv)
    
    return float(consistency)


def peak_performance_ratio(df: pd.DataFrame, top_percent: float = 0.1) -> float:
    """
    Calculate ratio of top performing posts to average.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    top_percent : float
        Percentage of posts to consider as "top" (default: 10%).
    
    Returns
    -------
    float
        Ratio of top performers to average.
    """
    if len(df) == 0:
        return 0.0
    
    # Sort by engagement
    sorted_df = df.sort_values('engagement_count', ascending=False)
    
    # Get top N% posts
    n_top = max(1, int(len(df) * top_percent))
    top_posts = sorted_df.head(n_top)
    
    # Calculate averages
    top_avg = top_posts['engagement_count'].mean()
    overall_avg = df['engagement_count'].mean()
    
    if overall_avg == 0:
        return 0.0
    
    ratio = top_avg / overall_avg
    
    return float(ratio)


def audience_retention_rate(df: pd.DataFrame, window_days: int = 30) -> float:
    """
    Calculate audience retention rate based on engagement trends.
    
    Measures if engagement is maintained over time.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    window_days : int
        Number of days for rolling window.
    
    Returns
    -------
    float
        Retention rate (0-100%).
    """
    if len(df) < 2:
        return 0.0
    
    # Sort by timestamp
    sorted_df = df.sort_values('timestamp').copy()
    
    # Set timestamp as index
    sorted_df.set_index('timestamp', inplace=True)
    
    # Calculate rolling average engagement
    rolling_engagement = sorted_df['engagement_count'].rolling(
        window=f'{window_days}D',
        min_periods=1
    ).mean()
    
    if len(rolling_engagement) < 2:
        return 0.0
    
    # Calculate trend using linear regression
    x = np.arange(len(rolling_engagement))
    y = rolling_engagement.values
    
    # Remove NaN values
    mask = ~np.isnan(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return 0.0
    
    # Fit linear trend
    slope, intercept = np.polyfit(x, y, 1)
    
    # Positive slope = growing engagement = good retention
    # Normalize to 0-100 scale
    if intercept > 0:
        retention_score = 50 + (slope / intercept) * 50
    else:
        retention_score = 50
    
    # Bound between 0 and 100
    retention_score = max(0, min(100, retention_score))
    
    return float(retention_score)


def hashtag_effectiveness_score(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate effectiveness score for each hashtag.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    
    Returns
    -------
    Dict[str, float]
        Dictionary mapping hashtags to effectiveness scores.
    """
    hashtag_performance = {}
    
    # Flatten hashtags and associate with engagement
    for _, post in df.iterrows():
        hashtags = post.get('hashtags', [])
        if isinstance(hashtags, list):
            engagement = post['engagement_count']
            views = post['views_count']
            
            for hashtag in hashtags:
                if hashtag not in hashtag_performance:
                    hashtag_performance[hashtag] = {
                        'total_engagement': 0,
                        'total_views': 0,
                        'post_count': 0
                    }
                
                hashtag_performance[hashtag]['total_engagement'] += engagement
                hashtag_performance[hashtag]['total_views'] += views
                hashtag_performance[hashtag]['post_count'] += 1
    
    # Calculate effectiveness scores
    hashtag_scores = {}
    
    for hashtag, stats in hashtag_performance.items():
        if stats['total_views'] > 0:
            engagement_rate = (stats['total_engagement'] / stats['total_views']) * 100
        else:
            engagement_rate = 0
        
        # Weight by post count (popular hashtags get slight boost)
        popularity_factor = np.log1p(stats['post_count'])
        
        score = engagement_rate * popularity_factor
        hashtag_scores[hashtag] = float(score)
    
    # Sort by score
    sorted_scores = dict(sorted(
        hashtag_scores.items(), 
        key=lambda x: x[1], 
        reverse=True
    ))
    
    return sorted_scores


def growth_velocity(df: pd.DataFrame, metric: str = 'engagement_count') -> float:
    """
    Calculate growth velocity for a specific metric.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    metric : str
        Metric to calculate growth for.
    
    Returns
    -------
    float
        Growth rate per day.
    """
    if len(df) < 2 or metric not in df.columns:
        return 0.0
    
    # Sort by timestamp
    sorted_df = df.sort_values('timestamp').copy()
    
    # Calculate daily aggregates
    sorted_df['date'] = sorted_df['timestamp'].dt.date
    daily_metrics = sorted_df.groupby('date')[metric].sum()
    
    if len(daily_metrics) < 2:
        return 0.0
    
    # Calculate growth rate
    first_value = daily_metrics.iloc[0]
    last_value = daily_metrics.iloc[-1]
    days = (daily_metrics.index[-1] - daily_metrics.index[0]).days
    
    if days == 0 or first_value == 0:
        return 0.0
    
    # Compound growth rate
    growth_rate = ((last_value / first_value) ** (1 / days) - 1) * 100
    
    return float(growth_rate)


def calculate_all_metrics(
    df: pd.DataFrame,
    followers_count: Optional[int] = None
) -> Dict[str, Union[float, int, Dict]]:
    """
    Calculate all available metrics.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with social media posts.
    followers_count : Optional[int]
        Number of followers.
    
    Returns
    -------
    Dict[str, Union[float, int, Dict]]
        Dictionary with all calculated metrics.
    """
    # Get posts with min/max metrics
    max_views_data = get_post_with_max_views(df)
    min_views_data = get_post_with_min_views(df)
    max_likes_data = get_post_with_max_likes(df)
    min_likes_data = get_post_with_min_likes(df)
    
    metrics = {
        # Standard metrics
        'total_posts': len(df),
        'total_engagement': total_engagement(df),
        'average_views': average_views(df),
        'average_likes': average_likes(df),
        'average_comments': average_comments(df),
        'total_likes': total_likes(df),
        'total_views': total_views(df),
        'total_comments': total_comments(df),
        'average_views_videos': average_views_videos(df),
        'average_views_non_videos': average_views_non_videos(df),
        'max_likes': max_likes(df),
        'max_views': max_views(df),
        'max_views_count': max_views_data[0],
        'max_views_url': max_views_data[1],
        'min_views_count': min_views_data[0],
        'min_views_url': min_views_data[1],
        'max_likes_count': max_likes_data[0],
        'max_likes_url': max_likes_data[1],
        'min_likes_count': min_likes_data[0],
        'min_likes_url': min_likes_data[1],
        'median_views': percentile_views(df, 50),
        'percentile_90_views': percentile_views(df, 90),
        'engagement_rate': engagement_rate(df, followers_count),
        'posting_frequency': posting_frequency(df),
        
        # Creative metrics
        'viral_velocity_score': viral_velocity_score(df),
        'content_consistency_index': content_consistency_index(df),
        'peak_performance_ratio': peak_performance_ratio(df),
        'audience_retention_rate': audience_retention_rate(df),
        'growth_velocity': growth_velocity(df),
        'top_hashtags': dict(list(hashtag_effectiveness_score(df).items())[:10])
    }
    
    return metrics