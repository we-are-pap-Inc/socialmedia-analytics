"""
Error handling utilities for production deployment.

This module provides decorators and utilities for proper error handling,
including retry logic, error boundaries, and user-friendly error messages.
"""

import functools
import logging
import time
from typing import Any, Callable, Optional, Type, Union

import streamlit as st
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

logger = logging.getLogger(__name__)


class AppError(Exception):
    """Base exception for application errors."""
    def __init__(self, message: str, details: Optional[dict] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)


class APIError(AppError):
    """Exception for API-related errors."""
    pass


class DataError(AppError):
    """Exception for data processing errors."""
    pass


class ValidationError(AppError):
    """Exception for validation errors."""
    pass


def safe_execute(
    func: Callable,
    error_message: str = "An error occurred",
    show_details: bool = False,
    raise_on_error: bool = False
) -> Callable:
    """
    Decorator for safe execution with error handling.
    
    Parameters
    ----------
    func : Callable
        Function to wrap.
    error_message : str
        User-friendly error message.
    show_details : bool
        Whether to show error details to user.
    raise_on_error : bool
        Whether to re-raise the exception.
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in {func.__name__}: {str(e)}", exc_info=True)
            
            if show_details:
                st.error(f"{error_message}: {str(e)}")
            else:
                st.error(error_message)
                
            if raise_on_error:
                raise
            
            return None
    
    return wrapper


def retry_on_failure(
    max_attempts: int = 3,
    wait_multiplier: int = 1,
    min_wait: int = 4,
    max_wait: int = 10,
    exceptions: tuple = (APIError,)
) -> Callable:
    """
    Decorator for retrying failed operations.
    
    Parameters
    ----------
    max_attempts : int
        Maximum number of retry attempts.
    wait_multiplier : int
        Multiplier for exponential backoff.
    min_wait : int
        Minimum wait time between retries.
    max_wait : int
        Maximum wait time between retries.
    exceptions : tuple
        Exceptions to retry on.
    """
    return retry(
        stop=stop_after_attempt(max_attempts),
        wait=wait_exponential(multiplier=wait_multiplier, min=min_wait, max=max_wait),
        retry=retry_if_exception_type(exceptions),
        before_sleep=lambda retry_state: logger.warning(
            f"Retrying {retry_state.fn.__name__} "
            f"(attempt {retry_state.attempt_number}/{max_attempts})"
        )
    )


def validate_input(
    validation_func: Callable[[Any], bool],
    error_message: str = "Invalid input"
) -> Callable:
    """
    Decorator for input validation.
    
    Parameters
    ----------
    validation_func : Callable
        Function to validate input.
    error_message : str
        Error message for invalid input.
    """
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Validate first positional argument
            if args and not validation_func(args[0]):
                raise ValidationError(error_message)
            return func(*args, **kwargs)
        return wrapper
    return decorator


class ErrorBoundary:
    """
    Context manager for error boundaries in Streamlit apps.
    
    Usage
    -----
    with ErrorBoundary("Loading data"):
        # Code that might fail
        data = load_data()
    """
    
    def __init__(
        self,
        operation: str,
        show_spinner: bool = True,
        fallback_value: Any = None
    ):
        self.operation = operation
        self.show_spinner = show_spinner
        self.fallback_value = fallback_value
        self.spinner = None
        
    def __enter__(self):
        if self.show_spinner:
            self.spinner = st.spinner(f"{self.operation}...")
            self.spinner.__enter__()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.spinner:
            self.spinner.__exit__(exc_type, exc_val, exc_tb)
            
        if exc_type is not None:
            logger.error(
                f"Error during {self.operation}: {exc_val}",
                exc_info=(exc_type, exc_val, exc_tb)
            )
            st.error(f"Failed to {self.operation.lower()}")
            return True  # Suppress exception
        
        return False


def handle_api_error(error: Exception) -> dict:
    """
    Convert API errors to user-friendly messages.
    
    Parameters
    ----------
    error : Exception
        The exception to handle.
        
    Returns
    -------
    dict
        Error information dictionary.
    """
    error_info = {
        "error": True,
        "message": "An unexpected error occurred",
        "type": type(error).__name__
    }
    
    if isinstance(error, APIError):
        error_info["message"] = "API service is temporarily unavailable"
    elif isinstance(error, ValidationError):
        error_info["message"] = str(error)
    elif isinstance(error, DataError):
        error_info["message"] = "Error processing data"
    elif "rate limit" in str(error).lower():
        error_info["message"] = "Rate limit exceeded. Please try again later"
    elif "timeout" in str(error).lower():
        error_info["message"] = "Request timed out. Please try again"
    elif "network" in str(error).lower():
        error_info["message"] = "Network error. Please check your connection"
    
    return error_info


def create_error_response(
    platform: str,
    username: str,
    error: Union[str, Exception]
) -> dict:
    """
    Create a standardized error response.
    
    Parameters
    ----------
    platform : str
        Social media platform.
    username : str
        Account username.
    error : Union[str, Exception]
        Error message or exception.
        
    Returns
    -------
    dict
        Standardized error response.
    """
    return {
        "error": True,
        "platform": platform,
        "username": username,
        "message": str(error) if isinstance(error, str) else handle_api_error(error)["message"],
        "timestamp": time.time()
    }