"""
Configuration management for the Social Media Analytics application.

This module handles environment-specific settings and configuration validation.
"""

import os
from typing import Optional
from pydantic import BaseSettings, validator


class Settings(BaseSettings):
    """Application settings with environment variable support."""
    
    # Environment
    environment: str = "development"
    debug: bool = False
    
    # API Configuration
    apify_api_token: Optional[str] = None
    api_timeout: int = 600
    max_retries: int = 3
    
    # Scraping Limits
    instagram_posts_limit: int = 200
    tiktok_videos_limit: int = 100
    max_concurrent_scrapes: int = 5
    
    # Cache Configuration
    cache_ttl: int = 3600  # 1 hour
    cache_backend: str = "memory"  # memory, redis, file
    redis_url: Optional[str] = None
    
    # Security
    enable_auth: bool = False
    allowed_origins: list = ["*"]
    max_request_size: int = 10 * 1024 * 1024  # 10MB
    
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json, text
    enable_request_logging: bool = True
    
    # Performance
    enable_profiling: bool = False
    memory_limit_mb: int = 1024
    
    # Monitoring
    enable_monitoring: bool = False
    sentry_dsn: Optional[str] = None
    metrics_endpoint: Optional[str] = None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
    
    @validator("environment")
    def validate_environment(cls, v):
        allowed = ["development", "staging", "production"]
        if v not in allowed:
            raise ValueError(f"environment must be one of {allowed}")
        return v
    
    @validator("debug")
    def debug_only_in_development(cls, v, values):
        if v and values.get("environment") == "production":
            raise ValueError("debug cannot be True in production")
        return v
    
    @validator("apify_api_token")
    def validate_api_token(cls, v):
        if not v and os.getenv("APIFY_TOKEN"):
            return os.getenv("APIFY_TOKEN")
        return v


# Singleton instance
settings = Settings()


def get_settings() -> Settings:
    """Get application settings."""
    return settings