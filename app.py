"""
Social Media Analytics Dashboard

A Streamlit dashboard for analyzing Instagram and TikTok accounts
with cached data and manual refresh capabilities.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pytz

logger = logging.getLogger(__name__)

from src.analytics.instagram import InstagramAnalytics
from src.analytics.tiktok import TikTokAnalytics
from src.data_io import (
    get_latest_run_dir, get_run_dir, save_run, load_run,
    append_run_metadata, save_combined_report, load_combined_report
)
from src.ingest import scrape_instagram, scrape_instagram_profile, scrape_tiktok
from src.transform import transform_to_analytics_ready
from src.utils.helpers import get_platform_colors


# Page configuration
st.set_page_config(
    page_title="Social Media Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/your-repo/social-media-analytics',
        'Report a bug': 'https://github.com/your-repo/social-media-analytics/issues',
        'About': '# Social Media Analytics\nAnalyze Instagram & TikTok accounts with ease!'
    }
)


# Load accounts configuration
def load_accounts_config() -> Dict[str, Any]:
    """Load accounts configuration from accounts.json or Streamlit secrets."""
    # First, try to load from Streamlit secrets (for cloud deployment)
    try:
        if hasattr(st, 'secrets') and 'accounts' in st.secrets:
            # Convert streamlit secrets to proper dict format
            config = {}
            if 'instagram' in st.secrets.accounts:
                config['instagram'] = list(st.secrets.accounts.instagram)
            else:
                config['instagram'] = []
            
            if 'tiktok' in st.secrets.accounts:
                config['tiktok'] = list(st.secrets.accounts.tiktok)
            else:
                config['tiktok'] = []
            
            if 'settings' in st.secrets.accounts:
                config['settings'] = dict(st.secrets.accounts.settings)
            else:
                config['settings'] = {}
            
            return config
    except Exception as e:
        logger.warning(f"Error loading from secrets: {e}")
    
    # Fall back to accounts.json file
    accounts_path = Path("accounts.json")
    if not accounts_path.exists():
        st.error("accounts.json not found! Please create it with your account list.")
        return {"instagram": [], "tiktok": [], "settings": {}}
    
    with open(accounts_path, 'r') as f:
        return json.load(f)


# Main analytics function with caching
@st.cache_data(show_spinner=True)
def load_account_data(
    platform: str,
    username: str,
    post_limit: int = 200,
    date_limit: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load and analyze data for a single account.
    
    This function is cached - data persists until manually cleared.
    """
    try:
        # Scrape data
        if platform == "instagram":
            # Get profile data first
            profile_data = scrape_instagram_profile(username)
            
            # Get posts
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
            return {
                'error': True,
                'message': f"No data found for {platform} @{username}. Account may be private or doesn't exist.",
                'platform': platform,
                'username': username
            }
        
        # Transform data
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
        
    except Exception as e:
        return {
            'error': True,
            'message': f"Error analyzing {platform} @{username}: {str(e)}",
            'platform': platform,
            'username': username
        }


def display_kpi_metrics(metrics: Dict[str, Any], col):
    """Display KPI metrics in a column."""
    col.metric(
        "Total Interactions",
        f"{metrics.get('total_engagement', 0):,}",
        delta=None,
        help="Total likes + comments + shares"
    )
    
    col.metric(
        "Average Views",
        f"{metrics.get('average_views', 0):,.1f}",
        delta=None
    )
    
    col.metric(
        "Interaction Rate",
        f"{metrics.get('engagement_rate', 0):.2f}%",
        delta=None,
        help="(Likes + Comments + Shares) / Views × 100"
    )
    
    col.metric(
        "Viral Score",
        f"{metrics.get('viral_velocity_score', 0):.1f}",
        delta=None,
        help="Score indicating viral potential (0-100)"
    )


def create_time_series_chart(df: pd.DataFrame, platform: str) -> go.Figure:
    """Create time series chart for views and likes."""
    # Sort by timestamp
    df_sorted = df.sort_values('timestamp')
    
    # Get platform color
    colors = get_platform_colors()
    platform_color = colors.get(platform, '#1f77b4')
    
    # Create figure with secondary y-axis
    fig = go.Figure()
    
    # Add views trace
    fig.add_trace(
        go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['views_count'],
            name='Views',
            line=dict(color=platform_color, width=2),
            mode='lines+markers'
        )
    )
    
    # Add likes trace
    fig.add_trace(
        go.Scatter(
            x=df_sorted['timestamp'],
            y=df_sorted['likes_count'],
            name='Likes',
            line=dict(color='#ff7f0e', width=2),
            mode='lines+markers',
            yaxis='y2'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=f"{platform.title()} Performance Over Time",
        xaxis_title="Date",
        yaxis=dict(title="Views", side="left"),
        yaxis2=dict(title="Likes", overlaying="y", side="right"),
        hovermode='x unified',
        template='plotly_white'
    )
    
    return fig


def create_comprehensive_metrics_chart_with_subplots(all_results: List[Dict[str, Any]]) -> go.Figure:
    """Create comprehensive bar chart showing followers, likes, and views using subplots."""
    from plotly.subplots import make_subplots
    
    # Prepare data
    accounts = []
    followers_data = []
    likes_data = []
    views_data = []
    platforms = []
    
    for result in all_results:
        if not result['error']:
            # Get account name
            account_name = f"@{result['username']}"
            accounts.append(account_name)
            platforms.append(result['platform'])
            
            # Get followers
            followers = 0
            if result.get('profile_data') and result['profile_data'].get('followersCount'):
                followers = result['profile_data'].get('followersCount', 0)
            elif result.get('metrics') and 'followers_count' in result['metrics']:
                followers = result['metrics'].get('followers_count', 0)
            followers_data.append(followers)
            
            # Get metrics
            total_likes = result.get('metrics', {}).get('total_likes', 0)
            total_views = result.get('metrics', {}).get('total_views', 0)
            likes_data.append(total_likes)
            views_data.append(total_views)
    
    if not accounts:
        return go.Figure()
    
    # Get platform colors
    colors = get_platform_colors()
    
    # Create subplots
    fig = make_subplots(
        rows=3, cols=1,
        row_heights=[0.33, 0.33, 0.34],
        subplot_titles=('Followers', 'Total Likes', 'Total Views'),
        vertical_spacing=0.1
    )
    
    # Create color list based on platforms
    bar_colors = [colors.get(p, '#cccccc') for p in platforms]
    
    # Add followers subplot
    fig.add_trace(
        go.Bar(
            x=accounts,
            y=followers_data,
            marker_color=bar_colors,
            text=followers_data,
            textposition='outside',
            hovertemplate='%{x}<br>Followers: %{y:,}<extra></extra>',
            showlegend=False
        ),
        row=1, col=1
    )
    
    # Add likes subplot
    fig.add_trace(
        go.Bar(
            x=accounts,
            y=likes_data,
            marker_color=bar_colors,
            text=[f'{l:,}' for l in likes_data],
            textposition='outside',
            hovertemplate='%{x}<br>Total Likes: %{y:,}<extra></extra>',
            showlegend=False
        ),
        row=2, col=1
    )
    
    # Add views subplot
    fig.add_trace(
        go.Bar(
            x=accounts,
            y=views_data,
            marker_color=bar_colors,
            text=[f'{v:,}' for v in views_data],
            textposition='outside',
            hovertemplate='%{x}<br>Total Views: %{y:,}<extra></extra>',
            showlegend=False
        ),
        row=3, col=1
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Account Metrics Comparison',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        template='plotly_white',
        height=700,
        showlegend=False
    )
    
    # Update all x-axes
    fig.update_xaxes(tickangle=-45)
    
    # Update y-axes titles
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=3, col=1)
    
    # Add platform legend
    unique_platforms = list(set(platforms))
    for i, platform in enumerate(unique_platforms):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode='markers',
                marker=dict(size=10, color=colors.get(platform, '#cccccc')),
                legendgroup=platform,
                showlegend=True,
                name=platform.title()
            )
        )
    
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="top",
            y=1.05,
            xanchor="center",
            x=0.5
        )
    )
    
    return fig


def create_comprehensive_metrics_chart(all_results: List[Dict[str, Any]]) -> go.Figure:
    """Create comprehensive bar chart showing followers, likes, and views for all accounts."""
    # Prepare data for long format
    data_records = []
    
    for result in all_results:
        if not result['error']:
            account_name = f"@{result['username']}"
            platform = result['platform']
            
            # Get followers
            followers = 0
            if result.get('profile_data') and result['profile_data'].get('followersCount'):
                followers = result['profile_data'].get('followersCount', 0)
            elif result.get('metrics') and 'followers_count' in result['metrics']:
                followers = result['metrics'].get('followers_count', 0)
            
            # Get metrics
            total_likes = result.get('metrics', {}).get('total_likes', 0)
            total_views = result.get('metrics', {}).get('total_views', 0)
            
            # Add records for each metric
            data_records.append({
                'Account': account_name,
                'Platform': platform,
                'Metric': 'Followers',
                'Value': followers,
                'Display': str(followers)
            })
            data_records.append({
                'Account': account_name,
                'Platform': platform,
                'Metric': 'Likes',
                'Value': total_likes,
                'Display': f'{total_likes:,}'
            })
            data_records.append({
                'Account': account_name,
                'Platform': platform,
                'Metric': 'Views',
                'Value': total_views,
                'Display': f'{total_views:,}'
            })
    
    if not data_records:
        return go.Figure()
    
    # Create DataFrame
    df = pd.DataFrame(data_records)
    
    # Get unique accounts and metrics
    accounts = df['Account'].unique()
    metrics = ['Followers', 'Likes', 'Views']
    
    # Define colors for metrics
    metric_colors = {
        'Followers': '#3498db',  # Blue
        'Likes': '#e74c3c',      # Red
        'Views': '#2ecc71'       # Green
    }
    
    # Get platform colors
    platform_colors = get_platform_colors()
    
    # Create figure
    fig = go.Figure()
    
    # Calculate positions
    n_accounts = len(accounts)
    n_metrics = len(metrics)
    bar_width = 0.8 / n_metrics  # Total width divided by number of metrics
    
    # Add bars for each metric
    for i, metric in enumerate(metrics):
        metric_data = df[df['Metric'] == metric]
        
        # Create position offset for grouped bars
        offset = (i - (n_metrics - 1) / 2) * bar_width
        
        # Get values and hover text
        values = []
        hover_texts = []
        colors = []
        
        for account in accounts:
            account_data = metric_data[metric_data['Account'] == account]
            if not account_data.empty:
                row = account_data.iloc[0]
                values.append(row['Value'])
                hover_texts.append(f"{account}<br>{metric}: {row['Display']}")
                # Use platform color with transparency for visual grouping
                platform = row['Platform']
                base_color = platform_colors.get(platform, '#cccccc')
                colors.append(base_color)
            else:
                values.append(0)
                hover_texts.append(f"{account}<br>{metric}: 0")
                colors.append('#cccccc')
        
        # Add trace
        fig.add_trace(go.Bar(
            name=metric,
            x=[i + offset for i in range(n_accounts)],
            y=values,
            width=bar_width,
            marker_color=metric_colors[metric],
            text=[v if v > 0 else '' for v in values],
            textposition='outside',
            texttemplate='%{text}',
            hovertext=hover_texts,
            hoverinfo='text'
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': 'Account Metrics Comparison',
            'font': {'size': 20},
            'x': 0.5,
            'xanchor': 'center'
        },
        xaxis={
            'tickmode': 'array',
            'tickvals': list(range(n_accounts)),
            'ticktext': accounts,
            'tickangle': -45,
            'title': ''
        },
        yaxis={
            'title': 'Count',
            'gridcolor': 'rgba(0,0,0,0.1)'
        },
        barmode='group',
        template='plotly_white',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor='rgba(255,255,255,0.8)'
        ),
        height=500,
        margin=dict(t=100, b=120, l=80, r=80),
        plot_bgcolor='white'
    )
    
    # Add subtle platform indicators
    for i, account in enumerate(accounts):
        # Get platform for this account
        account_data = df[df['Account'] == account].iloc[0]
        platform = account_data['Platform']
        platform_color = platform_colors.get(platform, '#cccccc')
        
        # Add platform text below bars
        fig.add_annotation(
            text=f"[{platform}]",
            x=i,
            y=-0.08,
            xref="x",
            yref="paper",
            showarrow=False,
            font=dict(size=9, color=platform_color),
            xanchor="center"
        )
    
    # Add value labels that are cut off
    fig.update_traces(
        textfont_size=10,
        textangle=0,
        cliponaxis=False
    )
    
    return fig


def display_account_analysis(result: Dict[str, Any]):
    """Display detailed analysis for a single account."""
    if result['error']:
        st.error(result['message'])
        return
    
    # Header with profile info
    col1, col2 = st.columns([1, 3])
    
    with col1:
        st.subheader(f"@{result['username']}")
        if result.get('profile_data'):
            profile = result['profile_data']
            if profile.get('followersCount'):
                st.metric("Followers", f"{profile['followersCount']:,}")
            if profile.get('fullName'):
                st.caption(profile['fullName'])
    
    with col2:
        # KPI Metrics - First row
        metrics_cols = st.columns(4)
        metrics = result['metrics']
        
        metrics_cols[0].metric(
            "Total Posts",
            metrics.get('total_posts', 0)
        )
        
        metrics_cols[1].metric(
            "Total Likes",
            f"{metrics.get('total_likes', 0):,}"
        )
        
        metrics_cols[2].metric(
            "Total Comments",
            f"{metrics.get('total_comments', 0):,}"
        )
        
        metrics_cols[3].metric(
            "Total Views",
            f"{metrics.get('total_views', 0):,}"
        )
        
        # Second row of metrics
        metrics_cols2 = st.columns(4)
        
        metrics_cols2[0].metric(
            "Avg Likes",
            f"{metrics.get('average_likes', 0):,.1f}"
        )
        
        metrics_cols2[1].metric(
            "Avg Comments",
            f"{metrics.get('average_comments', 0):,.1f}"
        )
        
        metrics_cols2[2].metric(
            "Avg Views",
            f"{metrics.get('average_views', 0):,.1f}"
        )
        
        metrics_cols2[3].metric(
            "Interaction Rate",
            f"{metrics.get('engagement_rate', 0):.2f}%"
        )
    
    # Time series chart
    st.plotly_chart(
        create_time_series_chart(result['df'], result['platform']),
        use_container_width=True
    )
    
    # Platform-specific insights
    if 'report' in result and result['report'].get('insights'):
        st.subheader("Insights")
        for insight in result['report']['insights'][:5]:
            st.info(f"💡 {insight}")
    
    # Top performing content
    if 'analytics' in result:
        st.subheader("Top Performing Posts")
        top_posts = result['analytics'].get_top_posts(5, by='engagement_count')
        
        if len(top_posts) > 0:
            display_df = top_posts[['caption', 'likes_count', 'views_count', 'url']].copy()
            display_df['caption'] = display_df['caption'].str[:50] + '...'
            st.dataframe(display_df, use_container_width=True)


def main():
    """Main application function."""
    st.title("📊 Social Media Analytics Dashboard")
    
    # Display last update timestamp
    latest_run = get_latest_run_dir()
    if latest_run:
        # Extract timestamp from directory name (format: YYYY-MM-DDTHH-MM-SS)
        timestamp_str = latest_run.name
        try:
            # Parse the timestamp (assumes UTC)
            timestamp_utc = datetime.strptime(timestamp_str, "%Y-%m-%dT%H-%M-%S")
            
            # Convert to configured timezone (default to PT)
            utc_tz = pytz.timezone('UTC')
            
            # Get timezone from settings or default to Pacific
            display_timezone = 'US/Pacific'
            timezone_abbr = 'PT'
            
            if 'config' in st.session_state and 'settings' in st.session_state.config:
                tz_setting = st.session_state.config['settings'].get('timezone', 'US/Pacific')
                if tz_setting:
                    display_timezone = tz_setting
                    # Get timezone abbreviation
                    tz_obj = pytz.timezone(display_timezone)
                    timezone_abbr = datetime.now(tz_obj).strftime('%Z')
            
            target_tz = pytz.timezone(display_timezone)
            timestamp_utc = utc_tz.localize(timestamp_utc)
            timestamp_local = timestamp_utc.astimezone(target_tz)
            
            # Format with timezone
            st.caption(f"Last update: {timestamp_local.strftime('%B %d, %Y at %I:%M %p')} {timezone_abbr}")
        except Exception as e:
            logger.warning(f"Error parsing timestamp: {e}")
            st.caption(f"Last update: {timestamp_str}")
    
    # Initialize session state for config if not exists
    if 'config' not in st.session_state:
        st.session_state.config = load_accounts_config()
    
    # Use config from session state
    config = st.session_state.config
    all_accounts = []
    
    # Combine all accounts
    for platform in ['instagram', 'tiktok']:
        for username in config.get(platform, []):
            all_accounts.append((platform, username))
    
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Account Management Section
        with st.expander("📝 Manage Accounts", expanded=False):
            st.caption("Add or remove accounts")
            
            # Add new account
            col1, col2 = st.columns([1, 2])
            with col1:
                new_platform = st.selectbox("Platform", ["instagram", "tiktok"], key="new_platform")
            with col2:
                new_username = st.text_input("Username (without @)", key="new_username")
            
            if st.button("➕ Add Account", type="secondary"):
                if new_username:
                    # Update config in session state
                    if new_platform not in st.session_state.config:
                        st.session_state.config[new_platform] = []
                    if new_username not in st.session_state.config[new_platform]:
                        st.session_state.config[new_platform].append(new_username)
                        # Save to file for persistence (local only)
                        try:
                            with open("accounts.json", "w") as f:
                                json.dump(st.session_state.config, f, indent=2)
                        except Exception as e:
                            st.warning(f"Could not save to file: {e}")
                        st.success(f"Added @{new_username} to {new_platform}!")
                        st.rerun()
                    else:
                        st.warning("Account already exists!")
            
            # Remove account
            if all_accounts:
                st.divider()
                account_to_remove = st.selectbox(
                    "Remove account",
                    options=all_accounts,
                    format_func=lambda x: f"{x[0].title()} @{x[1]}",
                    key="remove_account"
                )
                if st.button("🗑️ Remove Account", type="secondary"):
                    platform, username = account_to_remove
                    st.session_state.config[platform].remove(username)
                    # Save to file for persistence (local only)
                    try:
                        with open("accounts.json", "w") as f:
                            json.dump(st.session_state.config, f, indent=2)
                    except Exception as e:
                        st.warning(f"Could not save to file: {e}")
                    st.success(f"Removed @{username} from {platform}!")
                    st.rerun()
        
        st.divider()
        
        # Account selection
        selected_accounts = st.multiselect(
            "Select Accounts to Analyze",
            options=all_accounts,
            default=all_accounts,
            format_func=lambda x: f"{x[0].title()} @{x[1]}"
        )
        
        # Post limits
        st.subheader("Scraping Limits")
        instagram_limit = st.number_input(
            "Instagram posts limit",
            min_value=10,
            max_value=500,
            value=config.get('settings', {}).get('instagram_posts_limit', 200)
        )
        
        tiktok_limit = st.number_input(
            "TikTok videos limit",
            min_value=10,
            max_value=500,
            value=config.get('settings', {}).get('tiktok_videos_limit', 100)
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
            st.rerun()
    
    # Main content area
    if not selected_accounts:
        st.info("Please select at least one account from the sidebar.")
        return
    
    # Load data for selected accounts
    all_results = []
    
    # Use columns for progress
    progress_col, status_col = st.columns([3, 1])
    
    with progress_col:
        progress_bar = st.progress(0)
        status_text = st.empty()
    
    # Check if we should load from cache or latest run
    latest_run = get_latest_run_dir()
    load_from_cache = True
    combined_report = None
    
    if latest_run and not refresh_button:
        # Try to load from latest run first
        status_text.text("Loading from latest run...")
        load_from_cache = False
        # Load combined report for profile data
        combined_report = load_combined_report(latest_run)
    
    # Process each account
    for idx, (platform, username) in enumerate(selected_accounts):
        progress = (idx + 1) / len(selected_accounts)
        progress_bar.progress(progress)
        status_text.text(f"Loading {platform} @{username}...")
        
        # Set limits based on platform
        if platform == "instagram":
            limit = instagram_limit
        else:
            limit = tiktok_limit
        
        # Load data
        if load_from_cache:
            result = load_account_data(platform, username, limit)
        else:
            # Try to load from disk first
            try:
                df, metrics = load_run(latest_run, platform, username)
                
                # Get profile data from combined report if available
                profile_data = None
                if combined_report and 'accounts' in combined_report:
                    # Find this account in the combined report
                    for account in combined_report['accounts']:
                        if account.get('username') == username and account.get('platform') == platform:
                            profile_data = account.get('profile_data')
                            break
                
                # If no profile data from combined report, try to reconstruct from metrics
                if not profile_data and 'followers_count' in metrics:
                    profile_data = {'followersCount': metrics.get('followers_count', 0)}
                
                # Reconstruct result format
                result = {
                    'error': False,
                    'username': username,
                    'platform': platform,
                    'df': df,
                    'metrics': metrics,
                    'profile_data': profile_data
                }
            except:
                # Fall back to fetching fresh data
                result = load_account_data(platform, username, limit)
        
        all_results.append(result)
    
    progress_bar.empty()
    status_text.empty()
    
    # Save run if we fetched fresh data
    if refresh_button or load_from_cache:
        timestamp = datetime.now()
        run_dir = get_run_dir(timestamp)
        
        # Save individual results
        for result in all_results:
            if not result['error']:
                save_run(
                    result['df'],
                    result['metrics'],
                    run_dir,
                    result['platform'],
                    result['username']
                )
        
        # Save combined report
        save_combined_report(all_results, run_dir)
        
        # Append to runs.csv
        append_run_metadata(
            timestamp,
            [(r['platform'], r['username']) for r in all_results if not r['error']],
            0.0,  # Duration would need to be tracked
            "completed"
        )
    
    # Display overview metrics
    st.header("Overview Statistics")
    
    # Calculate comprehensive totals
    total_posts = sum(r['metrics'].get('total_posts', 0) for r in all_results if not r.get('error'))
    total_likes = sum(r['metrics'].get('total_likes', 0) for r in all_results if not r.get('error'))
    total_views = sum(r['metrics'].get('total_views', 0) for r in all_results if not r.get('error'))
    total_comments = sum(r['metrics'].get('total_comments', 0) for r in all_results if not r.get('error'))
    
    # Calculate total followers
    total_followers = 0
    for result in all_results:
        if not result.get('error'):
            # Try profile data first
            if result.get('profile_data') and result['profile_data'].get('followersCount'):
                followers = result['profile_data'].get('followersCount', 0)
                if isinstance(followers, int):
                    total_followers += followers
            # If no profile data, check if it's in the metrics
            elif result.get('metrics') and 'followers_count' in result['metrics']:
                followers = result['metrics'].get('followers_count', 0)
                if isinstance(followers, int) and followers > 0:
                    total_followers += followers
    
    # Calculate averages
    avg_likes_per_post = total_likes / max(total_posts, 1)
    avg_views_per_post = total_views / max(total_posts, 1)
    avg_comments_per_post = total_comments / max(total_posts, 1)
    
    # Calculate account averages
    num_accounts = len([r for r in all_results if not r.get('error')])
    avg_followers_per_account = total_followers / max(num_accounts, 1)
    avg_posts_per_account = total_posts / max(num_accounts, 1)
    
    # Display metrics in two rows
    # First row - Totals
    st.subheader("Cumulative Totals")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Total Followers", f"{total_followers:,}")
    col2.metric("Total Posts", f"{total_posts:,}")
    col3.metric("Total Likes", f"{total_likes:,}")
    col4.metric("Total Views", f"{total_views:,}")
    col5.metric("Total Comments", f"{total_comments:,}")
    
    # Second row - Averages
    st.subheader("Average Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    
    col1.metric("Avg Followers per Account", f"{avg_followers_per_account:,.1f}")
    col2.metric("Avg Posts per Account", f"{avg_posts_per_account:,.1f}")
    col3.metric("Avg Likes per Post", f"{avg_likes_per_post:,.1f}")
    col4.metric("Avg Views per Post", f"{avg_views_per_post:,.1f}")
    col5.metric("Avg Comments per Post", f"{avg_comments_per_post:,.1f}")
    
    # Display charts
    st.header("Account Comparisons")
    
    # Add chart type selector
    col1, col2, col3 = st.columns([2, 2, 6])
    with col1:
        chart_type = st.radio(
            "Chart Type",
            ["Separate Metrics", "Grouped Bars"],
            horizontal=False,
            help="Choose how to display the metrics"
        )
    
    # Display selected chart
    if chart_type == "Separate Metrics":
        st.plotly_chart(
            create_comprehensive_metrics_chart_with_subplots(all_results),
            use_container_width=True
        )
    else:
        st.plotly_chart(
            create_comprehensive_metrics_chart(all_results),
            use_container_width=True
        )
    
    # Account tabs
    st.header("Account Analysis")
    
    # Create tabs for each account
    tab_names = []
    for r in all_results:
        if 'platform' in r and 'username' in r:
            tab_names.append(f"{r['platform'].title()} @{r['username']}")
        else:
            # Fallback for malformed results
            tab_names.append("Unknown Account")
    
    if tab_names:
        tabs = st.tabs(tab_names)
        
        for tab, result in zip(tabs, all_results):
            with tab:
                display_account_analysis(result)
    else:
        st.warning("No accounts to display")
    
    # Download section
    st.header("Export Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Download Combined CSV"):
            # Create combined CSV
            csv_data = []
            for result in all_results:
                if not result['error']:
                    csv_data.append({
                        'platform': result['platform'],
                        'username': result['username'],
                        'total_posts': result['metrics'].get('total_posts', 0),
                        'total_engagement': result['metrics'].get('total_engagement', 0),
                        'average_views': result['metrics'].get('average_views', 0),
                        'engagement_rate': result['metrics'].get('engagement_rate', 0)
                    })
            
            if csv_data:
                # Add more detailed metrics to CSV
                enhanced_csv_data = []
                for result in all_results:
                    if not result['error']:
                        row = {
                            'platform': result['platform'],
                            'username': result['username'],
                            'followers': result.get('profile_data', {}).get('followersCount', 'N/A'),
                            'total_posts': result['metrics'].get('total_posts', 0),
                            'total_likes': result['metrics'].get('total_likes', 0),
                            'total_views': result['metrics'].get('total_views', 0),
                            'total_comments': result['metrics'].get('total_comments', 0),
                            'avg_likes_per_post': result['metrics'].get('average_likes', 0),
                            'avg_views_per_post': result['metrics'].get('average_views', 0),
                            'avg_comments_per_post': result['metrics'].get('average_comments', 0),
                            'interaction_rate': result['metrics'].get('engagement_rate', 0),
                            'viral_score': result['metrics'].get('viral_velocity_score', 0)
                        }
                        enhanced_csv_data.append(row)
                
                df = pd.DataFrame(enhanced_csv_data)
                csv = df.to_csv(index=False)
                st.download_button(
                    "Download CSV",
                    data=csv,
                    file_name=f"social_media_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
    
    with col2:
        # Placeholder for PDF download
        st.info("PDF export coming soon!")


if __name__ == "__main__":
    main()