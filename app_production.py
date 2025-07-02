"""
Production-ready Social Media Analytics Dashboard

Enhanced version with proper error handling, security, and performance optimizations.
"""

import json
import logging
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import local modules with error handling
try:
    from src.analytics.instagram import InstagramAnalytics
    from src.analytics.tiktok import TikTokAnalytics
    from src.data_io import (
        get_latest_run_dir, get_run_dir, save_run, load_run,
        append_run_metadata, save_combined_report, load_combined_report
    )
    from src.ingest import scrape_instagram, scrape_instagram_profile, scrape_tiktok
    from src.transform import transform_to_analytics_ready
    from src.utils.helpers import get_platform_colors
    from src.utils.validators import (
        validate_username, validate_platform, validate_limit,
        validate_config, validate_api_token
    )
    from src.utils.error_handler import (
        ErrorBoundary, safe_execute, retry_on_failure,
        create_error_response, APIError
    )
except ImportError as e:
    logger.error(f"Failed to import required modules: {e}")
    st.error("Application initialization failed. Please check the logs.")
    st.stop()


# Page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/social-media-analytics',
        'Report a bug': 'https://github.com/your-repo/social-media-analytics/issues',
        'About': '# Social Media Analytics\\nAnalyze Instagram & TikTok accounts with ease!'
    }
)

# Hide sensitive error details in production
if st.secrets.get("environment", "development") == "production":
    st.set_option('client.showErrorDetails', False)


# Cache configuration with TTL for production
CACHE_TTL = st.secrets.get("cache_ttl", 3600)  # 1 hour default
MAX_RETRIES = st.secrets.get("max_retries", 3)


# Load accounts configuration with validation
def load_accounts_config() -> Dict[str, Any]:
    """Load and validate accounts configuration."""
    try:
        # First, try to load from Streamlit secrets (for cloud deployment)
        if hasattr(st, 'secrets') and 'accounts' in st.secrets:
            config = {}
            
            # Safely extract configuration
            accounts_section = st.secrets.get("accounts", {})
            
            config['instagram'] = list(accounts_section.get("instagram", []))
            config['tiktok'] = list(accounts_section.get("tiktok", []))
            config['settings'] = dict(accounts_section.get("settings", {}))
            
            # Validate configuration
            is_valid, error_msg = validate_config(config)
            if not is_valid:
                logger.error(f"Invalid configuration: {error_msg}")
                st.error(f"Configuration error: {error_msg}")
                return {"instagram": [], "tiktok": [], "settings": {}}
            
            return config
            
    except Exception as e:
        logger.warning(f"Error loading from secrets: {e}")
    
    # Fall back to accounts.json file
    accounts_path = Path("accounts.json")
    if not accounts_path.exists():
        st.warning("No accounts configured. Please add accounts in the sidebar.")
        return {"instagram": [], "tiktok": [], "settings": {}}
    
    try:
        with open(accounts_path, 'r') as f:
            config = json.load(f)
            
        # Validate configuration
        is_valid, error_msg = validate_config(config)
        if not is_valid:
            logger.error(f"Invalid configuration: {error_msg}")
            st.error(f"Configuration error: {error_msg}")
            return {"instagram": [], "tiktok": [], "settings": {}}
            
        return config
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in accounts.json: {e}")
        st.error("Invalid configuration file format")
        return {"instagram": [], "tiktok": [], "settings": {}}
    except Exception as e:
        logger.error(f"Error loading configuration: {e}")
        st.error("Failed to load configuration")
        return {"instagram": [], "tiktok": [], "settings": {}}


# Enhanced analytics function with retry logic
@st.cache_data(ttl=CACHE_TTL, show_spinner=True)
@retry_on_failure(max_attempts=MAX_RETRIES, exceptions=(APIError,))
def load_account_data(
    platform: str,
    username: str,
    post_limit: int = 200,
    date_limit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and analyze data for a single account with error handling.
    """
    # Validate inputs
    if not validate_platform(platform):
        return create_error_response(platform, username, "Invalid platform")
    
    if not validate_username(username):
        return create_error_response(platform, username, "Invalid username format")
    
    if not validate_limit(post_limit):
        return create_error_response(platform, username, "Invalid post limit")
    
    try:
        with ErrorBoundary(f"Scraping {platform} @{username}"):
            # Validate API token
            api_token = st.secrets.get("APIFY_TOKEN") or os.getenv("APIFY_API_TOKEN")
            if not validate_api_token(api_token):
                raise APIError("Invalid or missing API token")
            
            # Scrape data
            if platform == "instagram":
                profile_data = scrape_instagram_profile(username)
                raw_data = scrape_instagram(username, limit=post_limit)
            else:
                raw_data = scrape_tiktok(username, limit=post_limit, date_limit=date_limit)
                profile_data = None
                
                # Extract profile from first post for TikTok
                if raw_data and len(raw_data) > 0:
                    first_post = raw_data[0]
                    if 'authorMeta' in first_post:
                        author_meta = first_post['authorMeta']
                        profile_data = {
                            'followersCount': author_meta.get('fans', 0),
                            'fullName': author_meta.get('nickName', ''),
                            'biography': author_meta.get('signature', ''),
                            'postsCount': author_meta.get('video', 0),
                            'verified': author_meta.get('verified', False)
                        }
        
        # Check for empty data
        if not raw_data or (len(raw_data) == 1 and 'error' in raw_data[0]):
            return create_error_response(
                platform, username,
                "No data found. Account may be private or doesn't exist."
            )
        
        # Transform data
        with ErrorBoundary("Processing data"):
            df = transform_to_analytics_ready(raw_data, platform)
            
            # Create account info
            account_info = {
                'username': username,
                'platform': platform,
                'followers_count': profile_data.get('followersCount') if profile_data else None,
                'full_name': profile_data.get('fullName') if profile_data else None,
                'bio': profile_data.get('biography') if profile_data else None,
                'posts_count': profile_data.get('postsCount') if profile_data else None
            }
            
            # Create analytics instance
            if platform == "instagram":
                analytics = InstagramAnalytics(df, account_info)
            else:
                analytics = TikTokAnalytics(df, account_info)
            
            # Generate report and metrics
            report = analytics.generate_report()
            metrics = analytics.calculate_metrics()
            
            return {
                'error': False,
                'username': username,
                'platform': platform,
                'df': df,
                'metrics': metrics,
                'report': report.model_dump() if hasattr(report, 'model_dump') else report,
                'profile_data': profile_data,
                'analytics': analytics
            }
        
    except APIError as e:
        logger.error(f"API error for {platform} @{username}: {str(e)}")
        return create_error_response(platform, username, str(e))
    except Exception as e:
        logger.error(f"Unexpected error for {platform} @{username}: {str(e)}", exc_info=True)
        return create_error_response(platform, username, "An unexpected error occurred")


# [Include all the display functions from the original app.py here]
# For brevity, I'm showing the main function with enhanced error handling


def main():
    """Main application function with comprehensive error handling."""
    try:
        st.title("📊 Social Media Analytics Dashboard")
        
        # Display environment indicator
        env = st.secrets.get("environment", "development")
        if env != "production":
            st.caption(f"Environment: {env}")
        
        # Display last update timestamp
        with ErrorBoundary("Loading update timestamp"):
            latest_run = get_latest_run_dir()
            if latest_run:
                timestamp_str = latest_run.name
                try:
                    timestamp = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
                    st.caption(f"Last update: {timestamp.strftime('%B %d, %Y at %I:%M %p')}")
                except ValueError:
                    st.caption(f"Last update: {timestamp_str}")
        
        # Initialize session state for config if not exists
        if 'config' not in st.session_state:
            st.session_state.config = load_accounts_config()
        
        # Use config from session state
        config = st.session_state.config
        all_accounts = []
        
        # Combine all accounts with validation
        for platform in ['instagram', 'tiktok']:
            for username in config.get(platform, []):
                if validate_username(username):
                    all_accounts.append((platform, username))
                else:
                    st.warning(f"Invalid username format: {username}")
        
        # Sidebar configuration
        with st.sidebar:
            st.header("Configuration")
            
            # Account Management Section with error handling
            with st.expander("📝 Manage Accounts", expanded=False):
                st.caption("Add or remove accounts")
                
                # Add new account
                col1, col2 = st.columns([1, 2])
                with col1:
                    new_platform = st.selectbox("Platform", ["instagram", "tiktok"], key="new_platform")
                with col2:
                    new_username = st.text_input("Username (without @)", key="new_username", max_chars=30)
                
                if st.button("➕ Add Account", type="secondary"):
                    if new_username:
                        if not validate_username(new_username):
                            st.error("Invalid username format. Only letters, numbers, underscore, and dot are allowed.")
                        else:
                            # Update config in session state
                            if new_platform not in st.session_state.config:
                                st.session_state.config[new_platform] = []
                            if new_username not in st.session_state.config[new_platform]:
                                st.session_state.config[new_platform].append(new_username)
                                # Save to file for persistence (local only)
                                try:
                                    with open("accounts.json", "w") as f:
                                        json.dump(st.session_state.config, f, indent=2)
                                    st.success(f"Added @{new_username} to {new_platform}!")
                                    st.rerun()
                                except Exception as e:
                                    st.warning("Could not save to file (expected on cloud deployment)")
                                    logger.warning(f"Failed to save config: {e}")
                            else:
                                st.warning("Account already exists!")
                
                # Remove account functionality here...
        
            # Account selection
            selected_accounts = st.multiselect(
                "Select Accounts to Analyze",
                options=all_accounts,
                default=all_accounts[:10],  # Limit default selection
                format_func=lambda x: f"{x[0].title()} @{x[1]}"
            )
            
            # Post limits with validation
            st.subheader("Scraping Limits")
            instagram_limit = st.number_input(
                "Instagram posts limit",
                min_value=10,
                max_value=500,
                value=min(config.get('settings', {}).get('instagram_posts_limit', 200), 500)
            )
            
            tiktok_limit = st.number_input(
                "TikTok videos limit",
                min_value=10,
                max_value=500,
                value=min(config.get('settings', {}).get('tiktok_videos_limit', 100), 500)
            )
            
            # Refresh button
            st.divider()
            refresh_button = st.button(
                "🔄 Refresh Data",
                type="primary",
                use_container_width=True,
                help="Fetch fresh data from social media platforms"
            )
            
            if refresh_button:
                # Clear all cached data
                st.cache_data.clear()
                st.success("Cache cleared! Fetching fresh data...")
                time.sleep(0.5)  # Brief pause for user feedback
                st.rerun()
        
        # Main content area
        if not selected_accounts:
            st.info("Please select at least one account from the sidebar.")
            return
        
        # Limit number of accounts for performance
        if len(selected_accounts) > 20:
            st.warning(f"Selected {len(selected_accounts)} accounts. Processing first 20 for performance.")
            selected_accounts = selected_accounts[:20]
        
        # Load data with progress tracking
        all_results = []
        
        with st.container():
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for idx, (platform, username) in enumerate(selected_accounts):
                progress = (idx + 1) / len(selected_accounts)
                progress_bar.progress(progress)
                status_text.text(f"Loading {platform} @{username}...")
                
                # Set limits based on platform
                limit = instagram_limit if platform == "instagram" else tiktok_limit
                
                # Load data with error handling
                try:
                    result = load_account_data(platform, username, limit)
                    all_results.append(result)
                except Exception as e:
                    logger.error(f"Failed to load {platform} @{username}: {e}")
                    all_results.append(create_error_response(platform, username, "Failed to load data"))
            
            progress_bar.empty()
            status_text.empty()
        
        # Display results with error handling
        valid_results = [r for r in all_results if not r.get('error', False)]
        
        if not valid_results:
            st.error("No data could be loaded. Please check your configuration and try again.")
            return
        
        if len(valid_results) < len(all_results):
            st.warning(f"Loaded {len(valid_results)} of {len(all_results)} accounts successfully.")
        
        # [Include the rest of the display logic here with error handling]
        
    except Exception as e:
        logger.error(f"Critical error in main: {e}", exc_info=True)
        st.error("An unexpected error occurred. Please refresh the page or contact support.")


if __name__ == "__main__":
    main()