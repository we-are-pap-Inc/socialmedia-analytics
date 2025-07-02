"""
Data ingestion module for loading social media data from Apify and local files.

This module provides functions to load Instagram and TikTok data from various sources
and validate the data schema.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from apify_client import ApifyClient
from pydantic import BaseModel, ValidationError, field_validator

logger = logging.getLogger(__name__)


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