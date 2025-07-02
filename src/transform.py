"""
Data transformation module for normalizing social media data into unified schemas.

This module provides functions to transform platform-specific data into 
standardized pandas DataFrames for consistent analysis.
"""

import logging
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from dateutil import parser as date_parser

logger = logging.getLogger(__name__)


def extract_hashtags_from_text(text: str) -> List[str]:
    """
    Extract hashtags from text content.
    
    Parameters
    ----------
    text : str
        Text content to extract hashtags from.
    
    Returns
    -------
    List[str]
        List of hashtags (without # prefix).
    """
    if not text:
        return []
    
    hashtags = re.findall(r'#(\w+)', text)
    return list(set(hashtags))


def extract_mentions_from_text(text: str) -> List[str]:
    """
    Extract mentions from text content.
    
    Parameters
    ----------
    text : str
        Text content to extract mentions from.
    
    Returns
    -------
    List[str]
        List of mentions (without @ prefix).
    """
    if not text:
        return []
    
    mentions = re.findall(r'@(\w+)', text)
    return list(set(mentions))


def parse_timestamp(timestamp: str) -> datetime:
    """
    Parse various timestamp formats into datetime objects.
    
    Parameters
    ----------
    timestamp : str
        Timestamp string in various formats.
    
    Returns
    -------
    datetime
        Parsed datetime object.
    """
    try:
        return date_parser.parse(timestamp)
    except Exception as e:
        logger.error(f"Failed to parse timestamp {timestamp}: {str(e)}")
        raise


def normalize_instagram_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize Instagram data into a standardized DataFrame.
    
    Parameters
    ----------
    raw_data : List[Dict[str, Any]]
        List of raw Instagram post data.
    
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with standardized columns.
    """
    logger.info(f"Normalizing {len(raw_data)} Instagram posts")
    
    normalized_data = []
    
    for post in raw_data:
        try:
            # Extract hashtags and mentions
            hashtags = post.get('hashtags', [])
            mentions = post.get('mentions', [])
            
            # If hashtags/mentions are not provided, extract from caption
            caption = post.get('caption', '')
            if not hashtags and caption:
                hashtags = extract_hashtags_from_text(caption)
            if not mentions and caption:
                mentions = extract_mentions_from_text(caption)
            
            # Parse timestamp
            timestamp = parse_timestamp(post.get('timestamp', ''))
            
            normalized_post = {
                'post_id': post.get('id'),
                'platform': 'instagram',
                'post_type': post.get('type', 'Unknown').lower(),
                'short_code': post.get('shortCode'),
                'caption': caption,
                'hashtags': hashtags,
                'mentions': mentions,
                'url': post.get('url'),
                'likes_count': post.get('likesCount', 0),
                'comments_count': post.get('commentsCount', 0),
                'views_count': post.get('videoViewCount') or post.get('videoPlayCount', 0),
                'shares_count': 0,  # Instagram doesn't provide share counts
                'timestamp': timestamp,
                'timestamp_str': post.get('timestamp'),
                'owner_username': post.get('ownerUsername'),
                'owner_id': post.get('ownerId'),
                'owner_fullname': post.get('ownerFullName'),
                'video_duration': post.get('videoDuration'),
                'is_video': post.get('type', '').lower() == 'video',
                'is_sponsored': post.get('isSponsored', False),
                'music_info': post.get('musicInfo', {}),
                'raw_data': post
            }
            
            normalized_data.append(normalized_post)
            
        except Exception as e:
            logger.warning(f"Failed to normalize Instagram post {post.get('id', 'unknown')}: {str(e)}")
            continue
    
    df = pd.DataFrame(normalized_data)
    logger.info(f"Successfully normalized {len(df)} Instagram posts")
    
    return df


def normalize_tiktok_data(raw_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Normalize TikTok data into a standardized DataFrame.
    
    Parameters
    ----------
    raw_data : List[Dict[str, Any]]
        List of raw TikTok post data.
    
    Returns
    -------
    pd.DataFrame
        Normalized DataFrame with standardized columns.
    """
    logger.info(f"Normalizing {len(raw_data)} TikTok posts")
    
    normalized_data = []
    
    for post in raw_data:
        try:
            # Extract hashtags and mentions from text
            text = post.get('text', '')
            hashtags = extract_hashtags_from_text(text)
            mentions = extract_mentions_from_text(text)
            
            # Parse timestamp
            timestamp = parse_timestamp(post.get('createTimeISO', ''))
            
            # Extract author info
            author_name = post.get('authorMeta.name', '')
            author_avatar = post.get('authorMeta.avatar', '')
            
            # Extract music info
            music_name = post.get('musicMeta.musicName', '')
            music_author = post.get('musicMeta.musicAuthor', '')
            music_original = post.get('musicMeta.musicOriginal', False)
            
            # Extract video duration
            video_duration = post.get('videoMeta.duration', 0)
            
            normalized_post = {
                'post_id': post.get('id') or post.get('webVideoUrl', '').split('/')[-1],
                'platform': 'tiktok',
                'post_type': 'video',  # TikTok is primarily video
                'short_code': None,  # TikTok doesn't use short codes
                'caption': text,
                'hashtags': hashtags,
                'mentions': mentions,
                'url': post.get('webVideoUrl'),
                'likes_count': post.get('diggCount', 0),
                'comments_count': post.get('commentCount', 0),
                'views_count': post.get('playCount', 0),
                'shares_count': post.get('shareCount', 0),
                'timestamp': timestamp,
                'timestamp_str': post.get('createTimeISO'),
                'owner_username': author_name,
                'owner_id': None,  # Not provided in sample data
                'owner_fullname': author_name,
                'owner_avatar': author_avatar,
                'video_duration': video_duration,
                'is_video': True,
                'is_sponsored': False,  # Would need additional logic to detect
                'music_info': {
                    'name': music_name,
                    'author': music_author,
                    'is_original': music_original
                },
                'raw_data': post
            }
            
            normalized_data.append(normalized_post)
            
        except Exception as e:
            logger.warning(f"Failed to normalize TikTok post: {str(e)}")
            continue
    
    df = pd.DataFrame(normalized_data)
    logger.info(f"Successfully normalized {len(df)} TikTok posts")
    
    return df


def create_unified_schema(df: pd.DataFrame, platform: str) -> pd.DataFrame:
    """
    Ensure DataFrame conforms to unified schema with all required columns.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to unify.
    platform : str
        Platform name for validation.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with unified schema.
    """
    required_columns = [
        'post_id', 'platform', 'post_type', 'caption', 'hashtags', 'mentions',
        'url', 'likes_count', 'comments_count', 'views_count', 'shares_count',
        'timestamp', 'owner_username', 'video_duration', 'is_video'
    ]
    
    # Add missing columns with default values
    for col in required_columns:
        if col not in df.columns:
            if col in ['likes_count', 'comments_count', 'views_count', 'shares_count', 'video_duration']:
                df[col] = 0
            elif col in ['hashtags', 'mentions']:
                df[col] = [[] for _ in range(len(df))]
            elif col == 'is_video':
                df[col] = False
            else:
                df[col] = None
    
    # Ensure numeric columns are numeric
    numeric_columns = ['likes_count', 'comments_count', 'views_count', 'shares_count', 'video_duration']
    for col in numeric_columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    
    # Add calculated columns
    df['engagement_count'] = df['likes_count'] + df['comments_count'] + df['shares_count']
    df['hashtag_count'] = df['hashtags'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['mention_count'] = df['mentions'].apply(lambda x: len(x) if isinstance(x, list) else 0)
    df['caption_length'] = df['caption'].fillna('').apply(len)
    
    # Handle music_info column if it exists
    if 'music_info' in df.columns:
        df['has_music'] = df['music_info'].apply(lambda x: bool(x) if isinstance(x, dict) else False)
    else:
        df['has_music'] = False
    
    # Add time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['day_name'] = df['timestamp'].dt.day_name()
    df['is_weekend'] = df['day_of_week'].isin([5, 6])
    
    logger.info(f"Created unified schema for {len(df)} {platform} posts")
    
    return df


def merge_platform_data(
    instagram_df: Optional[pd.DataFrame] = None,
    tiktok_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Merge data from multiple platforms into a single DataFrame.
    
    Parameters
    ----------
    instagram_df : Optional[pd.DataFrame]
        Instagram data DataFrame.
    tiktok_df : Optional[pd.DataFrame]
        TikTok data DataFrame.
    
    Returns
    -------
    pd.DataFrame
        Merged DataFrame with data from all platforms.
    """
    dfs_to_merge = []
    
    if instagram_df is not None and not instagram_df.empty:
        dfs_to_merge.append(instagram_df)
    
    if tiktok_df is not None and not tiktok_df.empty:
        dfs_to_merge.append(tiktok_df)
    
    if not dfs_to_merge:
        logger.warning("No data to merge")
        return pd.DataFrame()
    
    merged_df = pd.concat(dfs_to_merge, ignore_index=True, sort=False)
    
    # Sort by timestamp
    merged_df = merged_df.sort_values('timestamp', ascending=False)
    
    logger.info(f"Merged {len(merged_df)} posts from {merged_df['platform'].nunique()} platforms")
    
    return merged_df


def transform_to_analytics_ready(
    raw_data: List[Dict[str, Any]],
    platform: str
) -> pd.DataFrame:
    """
    Complete transformation pipeline from raw data to analytics-ready DataFrame.
    
    Parameters
    ----------
    raw_data : List[Dict[str, Any]]
        Raw data from platform.
    platform : str
        Platform name ('instagram' or 'tiktok').
    
    Returns
    -------
    pd.DataFrame
        Analytics-ready DataFrame.
    """
    platform = platform.lower()
    
    if platform == 'instagram':
        df = normalize_instagram_data(raw_data)
    elif platform == 'tiktok':
        df = normalize_tiktok_data(raw_data)
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    
    df = create_unified_schema(df, platform)
    
    return df