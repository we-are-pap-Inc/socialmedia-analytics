"""
Data ingestion module for loading social media data from Apify and local files.

This module provides functions to load Instagram and TikTok data from various sources
and validate the data schema.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple

from apify_client import ApifyClient
from pydantic import BaseModel, ValidationError, field_validator
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Actor IDs
INSTAGRAM_ACTOR_ID = "shu8hvrXbJbY3Eb9W"  # For posts
INSTAGRAM_PROFILE_ACTOR_ID = "dSCLg0C3YEZ83HzYX"  # For profile metadata
TIKTOK_ACTOR_ID = "OtzYfK1ndEGdwWFKQ"


class InstagramPostSchema(BaseModel):
    """Schema validation for Instagram post data."""
    
    id: str
    type: str
    shortCode: str
    caption: Optional[str] = ""
    hashtags: List[str] = []
    mentions: List[str] = []
    url: str
    commentsCount: int = 0
    likesCount: int = 0
    timestamp: str
    ownerUsername: str
    ownerId: str
    videoViewCount: Optional[int] = None
    videoPlayCount: Optional[int] = None
    videoDuration: Optional[float] = None
    
    @field_validator('timestamp')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp format."""
        if not v:
            raise ValueError('Timestamp cannot be empty')
        return v


class TikTokPostSchema(BaseModel):
    """Schema validation for TikTok post data."""
    
    text: str
    diggCount: int = 0
    shareCount: int = 0
    playCount: int = 0
    commentCount: int = 0
    createTimeISO: str
    webVideoUrl: str
    
    class Config:
        extra = 'allow'
    
    @field_validator('createTimeISO')
    @classmethod
    def validate_timestamp(cls, v: str) -> str:
        """Validate timestamp format."""
        if not v:
            raise ValueError('Timestamp cannot be empty')
        return v


def load_apify_data(
    dataset_id: str, 
    client: ApifyClient,
    limit: Optional[int] = None
) -> List[Dict]:
    """
    Load data from Apify dataset.
    
    Parameters
    ----------
    dataset_id : str
        The ID of the Apify dataset to load.
    client : ApifyClient
        Configured Apify client instance.
    limit : Optional[int]
        Maximum number of items to load. If None, loads all items.
    
    Returns
    -------
    List[Dict]
        List of data items from the dataset.
    
    Raises
    ------
    Exception
        If data loading fails.
    """
    try:
        logger.info(f"Loading data from Apify dataset: {dataset_id}")
        
        dataset_client = client.dataset(dataset_id)
        items = []
        
        for item in dataset_client.iterate_items():
            items.append(item)
            if limit and len(items) >= limit:
                break
        
        logger.info(f"Successfully loaded {len(items)} items from Apify dataset")
        return items
        
    except Exception as e:
        logger.error(f"Failed to load Apify data: {str(e)}")
        raise


def load_from_json(filepath: Union[str, Path]) -> List[Dict]:
    """
    Load data from a JSON file.
    
    Parameters
    ----------
    filepath : Union[str, Path]
        Path to the JSON file containing social media data.
    
    Returns
    -------
    List[Dict]
        List of data items from the JSON file.
    
    Raises
    ------
    FileNotFoundError
        If the file doesn't exist.
    JSONDecodeError
        If the file contains invalid JSON.
    """
    filepath = Path(filepath)
    
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    try:
        logger.info(f"Loading data from file: {filepath}")
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if not isinstance(data, list):
            data = [data]
        
        logger.info(f"Successfully loaded {len(data)} items from file")
        return data
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in file {filepath}: {str(e)}")
        raise


def validate_data_schema(
    data: List[Dict], 
    platform: str,
    strict: bool = False
) -> List[Dict]:
    """
    Validate data against platform-specific schema.
    
    Parameters
    ----------
    data : List[Dict]
        List of data items to validate.
    platform : str
        Platform name ('instagram' or 'tiktok').
    strict : bool
        If True, raises exception on validation error. 
        If False, logs errors and returns valid items only.
    
    Returns
    -------
    List[Dict]
        List of validated data items.
    
    Raises
    ------
    ValueError
        If platform is not supported or strict mode is enabled and validation fails.
    """
    platform = platform.lower()
    
    if platform == 'instagram':
        schema_class = InstagramPostSchema
    elif platform == 'tiktok':
        schema_class = TikTokPostSchema
    else:
        raise ValueError(f"Unsupported platform: {platform}")
    
    validated_items = []
    errors = []
    
    for idx, item in enumerate(data):
        try:
            validated_item = schema_class(**item)
            validated_items.append(validated_item.model_dump())
        except ValidationError as e:
            error_msg = f"Validation error for item {idx}: {str(e)}"
            errors.append(error_msg)
            logger.warning(error_msg)
            
            if strict:
                raise ValueError(error_msg) from e
    
    if errors:
        logger.warning(f"Validation completed with {len(errors)} errors out of {len(data)} items")
    else:
        logger.info(f"All {len(data)} items validated successfully")
    
    return validated_items


def load_instagram_data(
    source: Union[str, ApifyClient],
    dataset_id: Optional[str] = None,
    validate: bool = True,
    strict: bool = False
) -> List[Dict]:
    """
    Convenience function to load Instagram data.
    
    Parameters
    ----------
    source : Union[str, ApifyClient]
        Either a file path (str) or ApifyClient instance.
    dataset_id : Optional[str]
        Dataset ID if source is ApifyClient.
    validate : bool
        Whether to validate data schema.
    strict : bool
        Whether to use strict validation.
    
    Returns
    -------
    List[Dict]
        List of Instagram post data.
    """
    if isinstance(source, str) or isinstance(source, Path):
        data = load_from_json(source)
    elif isinstance(source, ApifyClient):
        if not dataset_id:
            raise ValueError("dataset_id required when using ApifyClient")
        data = load_apify_data(dataset_id, source)
    else:
        raise TypeError("source must be file path or ApifyClient instance")
    
    if validate:
        data = validate_data_schema(data, 'instagram', strict=strict)
    
    return data


def load_tiktok_data(
    source: Union[str, ApifyClient],
    dataset_id: Optional[str] = None,
    validate: bool = True,
    strict: bool = False
) -> List[Dict]:
    """
    Convenience function to load TikTok data.
    
    Parameters
    ----------
    source : Union[str, ApifyClient]
        Either a file path (str) or ApifyClient instance.
    dataset_id : Optional[str]
        Dataset ID if source is ApifyClient.
    validate : bool
        Whether to validate data schema.
    strict : bool
        Whether to use strict validation.
    
    Returns
    -------
    List[Dict]
        List of TikTok post data.
    """
    if isinstance(source, str) or isinstance(source, Path):
        data = load_from_json(source)
    elif isinstance(source, ApifyClient):
        if not dataset_id:
            raise ValueError("dataset_id required when using ApifyClient")
        data = load_apify_data(dataset_id, source)
    else:
        raise TypeError("source must be file path or ApifyClient instance")
    
    if validate:
        data = validate_data_schema(data, 'tiktok', strict=strict)
    
    return data


def scrape_instagram(
    username: str,
    limit: int = 200,
    client: Optional[ApifyClient] = None
) -> List[Dict]:
    """
    Scrape Instagram posts for a given username.
    
    Parameters
    ----------
    username : str
        Instagram username (without @).
    limit : int
        Maximum number of posts to scrape.
    client : Optional[ApifyClient]
        Apify client instance. If None, creates one from environment.
    
    Returns
    -------
    List[Dict]
        List of Instagram post data.
    
    Raises
    ------
    Exception
        If scraping fails.
    """
    if client is None:
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            # Try Streamlit secrets as fallback
            try:
                import streamlit as st
                api_token = st.secrets.get("APIFY_TOKEN")
            except:
                pass
        
        if not api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment or secrets")
        
        client = ApifyClient(api_token.strip('"').strip("'"))
    
    # Create actor input
    run_input = {
        "directUrls": [f"https://www.instagram.com/{username}/"],
        "resultsType": "posts",
        "resultsLimit": limit,
        "searchType": "user",
        "searchLimit": 1,
        "addParentData": True
    }
    
    logger.info(f"Starting Instagram scrape for @{username} (limit: {limit})")
    
    try:
        # Run the actor
        run = client.actor(INSTAGRAM_ACTOR_ID).call(run_input=run_input, wait_secs=600)
        
        if run['status'] != 'SUCCEEDED':
            raise Exception(f"Actor run failed with status: {run['status']}")
        
        # Get the dataset
        dataset_id = run['defaultDatasetId']
        items = list(client.dataset(dataset_id).iterate_items())
        
        logger.info(f"Successfully scraped {len(items)} posts for @{username}")
        return items
        
    except Exception as e:
        logger.error(f"Failed to scrape Instagram @{username}: {str(e)}")
        raise


def scrape_instagram_profile(
    username: str,
    client: Optional[ApifyClient] = None
) -> Optional[Dict]:
    """
    Scrape Instagram profile metadata.
    
    Parameters
    ----------
    username : str
        Instagram username (without @).
    client : Optional[ApifyClient]
        Apify client instance. If None, creates one from environment.
    
    Returns
    -------
    Optional[Dict]
        Profile metadata or None if failed.
    """
    if client is None:
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            # Try Streamlit secrets as fallback
            try:
                import streamlit as st
                api_token = st.secrets.get("APIFY_TOKEN")
            except:
                pass
        
        if not api_token:
            logger.warning("APIFY_API_TOKEN not found, skipping profile fetch")
            return None
        
        client = ApifyClient(api_token.strip('"').strip("'"))
    
    try:
        run_input = {"usernames": [username]}
        run = client.actor(INSTAGRAM_PROFILE_ACTOR_ID).call(run_input=run_input, wait_secs=300)
        
        if run['status'] == 'SUCCEEDED':
            dataset_id = run['defaultDatasetId']
            items = list(client.dataset(dataset_id).iterate_items())
            if items:
                logger.info(f"Successfully fetched profile data for @{username}")
                return items[0]
        
        logger.warning(f"Could not fetch profile data for @{username}")
        return None
        
    except Exception as e:
        logger.warning(f"Failed to fetch Instagram profile @{username}: {str(e)}")
        return None


def scrape_tiktok(
    username: str,
    limit: int = 100,
    date_limit: Optional[str] = None,
    client: Optional[ApifyClient] = None
) -> List[Dict]:
    """
    Scrape TikTok videos for a given username.
    
    Parameters
    ----------
    username : str
        TikTok username (without @).
    limit : int
        Maximum number of videos to scrape.
    date_limit : Optional[str]
        Optional date limit (e.g., "7 days", "2024-01-01").
    client : Optional[ApifyClient]
        Apify client instance. If None, creates one from environment.
    
    Returns
    -------
    List[Dict]
        List of TikTok video data.
    
    Raises
    ------
    Exception
        If scraping fails.
    """
    if client is None:
        api_token = os.getenv("APIFY_API_TOKEN")
        if not api_token:
            # Try Streamlit secrets as fallback
            try:
                import streamlit as st
                api_token = st.secrets.get("APIFY_TOKEN")
            except:
                pass
        
        if not api_token:
            raise ValueError("APIFY_API_TOKEN not found in environment or secrets")
        
        client = ApifyClient(api_token.strip('"').strip("'"))
    
    # Remove @ if present
    username = username.lstrip('@')
    
    # Create actor input
    run_input = {
        "profiles": [username],
        "resultsPerPage": limit,
        "profileScrapeSections": ["videos"],
        "profileSorting": "latest",
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadSlideshowImages": False
    }
    
    # Add date filter if specified
    if date_limit:
        run_input["oldestPostDateUnified"] = date_limit
    
    logger.info(f"Starting TikTok scrape for @{username} (limit: {limit})")
    
    try:
        # Run the actor
        run = client.actor(TIKTOK_ACTOR_ID).call(run_input=run_input, wait_secs=600)
        
        if run['status'] != 'SUCCEEDED':
            raise Exception(f"Actor run failed with status: {run['status']}")
        
        # Get the dataset
        dataset_id = run['defaultDatasetId']
        items = list(client.dataset(dataset_id).iterate_items())
        
        logger.info(f"Successfully scraped {len(items)} videos for @{username}")
        return items
        
    except Exception as e:
        logger.error(f"Failed to scrape TikTok @{username}: {str(e)}")
        raise