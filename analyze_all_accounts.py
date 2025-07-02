#!/usr/bin/env python3
"""
Streamlined script to analyze all social media accounts and generate comprehensive reports.

This script:
1. Reads accounts from accounts.json
2. Fetches data for each account
3. Analyzes all accounts
4. Generates individual and combined reports
"""

import json
import os
import sys
import time
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

from apify_client import ApifyClient
from dotenv import load_dotenv
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.table import Table

# Import our modules
from src.analytics.instagram import InstagramAnalytics
from src.analytics.tiktok import TikTokAnalytics
from src.ingest import load_instagram_data, load_tiktok_data
from src.transform import transform_to_analytics_ready
from src.report_generator import SocialMediaReportGenerator, generate_multi_account_report

# Load environment variables
load_dotenv()

console = Console()

# Actor IDs
INSTAGRAM_ACTOR_ID = "shu8hvrXbJbY3Eb9W"  # For posts
INSTAGRAM_PROFILE_ACTOR_ID = "dSCLg0C3YEZ83HzYX"  # For profile metadata
TIKTOK_ACTOR_ID = "OtzYfK1ndEGdwWFKQ"


def load_accounts() -> Dict[str, any]:
    """Load accounts and settings from accounts.json."""
    if not Path("accounts.json").exists():
        console.print("[red]Error:[/red] accounts.json not found!")
        console.print("Please create accounts.json with your account usernames.")
        sys.exit(1)
    
    with open("accounts.json", "r") as f:
        data = json.load(f)
    
    # Ensure settings exist with defaults
    if 'settings' not in data:
        data['settings'] = {}
    
    # Set default values
    data['settings'].setdefault('instagram_posts_limit', 200)
    data['settings'].setdefault('tiktok_videos_limit', 100)
    data['settings'].setdefault('tiktok_date_limit', None)
    
    return data


def create_instagram_input(username: str, posts_limit: int = 200) -> dict:
    """Create Instagram scraper input."""
    return {
        "directUrls": [f"https://www.instagram.com/{username}/"],
        "resultsType": "posts",
        "resultsLimit": posts_limit,
        "searchType": "user",
        "searchLimit": 1,
        "addParentData": True
    }


def create_tiktok_input(username: str, videos_limit: int = 100, date_limit: str = None) -> dict:
    """Create TikTok scraper input.
    
    Args:
        username: TikTok username (without @)
        videos_limit: Maximum number of videos to scrape
        date_limit: Optional date limit (e.g., "7 days", "2024-01-01")
    """
    # Remove @ if present (the scraper expects just the username)
    username = username.lstrip('@')
    
    input_data = {
        "profiles": [username],  # Just the username, not the full URL
        "resultsPerPage": videos_limit,
        "profileScrapeSections": ["videos"],  # Scrape the videos section
        "profileSorting": "latest",  # Get latest videos first
        "shouldDownloadVideos": False,
        "shouldDownloadCovers": False,
        "shouldDownloadSubtitles": False,
        "shouldDownloadSlideshowImages": False
    }
    
    # Add date filter if specified
    if date_limit:
        input_data["oldestPostDateUnified"] = date_limit
    
    return input_data


def fetch_instagram_profile(
    client: ApifyClient,
    username: str
) -> Optional[Dict]:
    """Fetch Instagram profile metadata."""
    try:
        run_input = {"usernames": [username]}
        run = client.actor(INSTAGRAM_PROFILE_ACTOR_ID).call(run_input=run_input, wait_secs=300)
        
        if run['status'] == 'SUCCEEDED':
            dataset_id = run['defaultDatasetId']
            items = list(client.dataset(dataset_id).iterate_items())
            if items:
                return items[0]  # Return first (and only) profile
        return None
    except Exception as e:
        console.print(f"[yellow]Warning: Could not fetch profile data for @{username}:[/yellow] {str(e)}")
        return None


def fetch_account_data(
    client: ApifyClient, 
    platform: str, 
    username: str,
    progress: Progress,
    task_id,
    settings: Dict = None
) -> Tuple[str, bool, Optional[Dict]]:
    """Fetch data for a single account."""
    progress.update(task_id, description=f"Fetching {platform} @{username}...")
    profile_data = None
    
    try:
        if platform == "instagram":
            # First fetch profile metadata
            profile_data = fetch_instagram_profile(client, username)
            
            # Then fetch posts
            actor_id = INSTAGRAM_ACTOR_ID
            posts_limit = settings.get('instagram_posts_limit', 200) if settings else 200
            run_input = create_instagram_input(username, posts_limit)
        else:
            actor_id = TIKTOK_ACTOR_ID
            videos_limit = settings.get('tiktok_videos_limit', 100) if settings else 100
            date_limit = settings.get('tiktok_date_limit', None) if settings else None
            run_input = create_tiktok_input(username, videos_limit, date_limit)
        
        # Start the actor
        run = client.actor(actor_id).call(run_input=run_input, wait_secs=600)  # 10 min timeout
        
        if run['status'] == 'SUCCEEDED':
            dataset_id = run['defaultDatasetId']
            progress.update(task_id, description=f"✓ {platform} @{username}")
            return dataset_id, True, profile_data
        else:
            progress.update(task_id, description=f"✗ {platform} @{username} - Failed")
            return None, False, profile_data
            
    except Exception as e:
        progress.update(task_id, description=f"✗ {platform} @{username} - Error")
        console.print(f"[red]Error fetching {platform} @{username}:[/red] {str(e)}")
        return None, False, profile_data


def analyze_account(
    client: ApifyClient,
    platform: str,
    username: str,
    dataset_id: str,
    output_dir: Path,
    profile_data: Optional[Dict] = None
) -> Dict:
    """Analyze a single account and save metrics."""
    try:
        # Load data
        if platform == "instagram":
            raw_data = load_instagram_data(client, dataset_id)
        else:
            raw_data = load_tiktok_data(client, dataset_id)
        
        # Check for empty or error data
        if not raw_data or (len(raw_data) == 1 and 'error' in raw_data[0]):
            console.print(f"[yellow]Warning:[/yellow] No data for {platform} @{username}")
            return None
        
        # Transform data
        df = transform_to_analytics_ready(raw_data, platform)
        
        # For TikTok, extract profile data from the first post's authorMeta
        if platform == "tiktok" and raw_data and len(raw_data) > 0:
            first_post = raw_data[0]
            if 'authorMeta' in first_post:
                author_meta = first_post['authorMeta']
                profile_data = {
                    'followersCount': author_meta.get('fans', 0),
                    'fullName': author_meta.get('nickName', ''),
                    'biography': author_meta.get('signature', ''),
                    'postsCount': author_meta.get('video', 0),
                    'following': author_meta.get('following', 0),
                    'hearts': author_meta.get('heart', 0),
                    'verified': author_meta.get('verified', False)
                }
        
        # Create analytics with profile data if available
        account_info = {
            'username': username,
            'platform': platform,
            'followers_count': profile_data.get('followersCount') if profile_data else None,
            'full_name': profile_data.get('fullName') if profile_data else None,
            'bio': profile_data.get('biography') if profile_data else None,  # Note: AccountInfo uses 'bio' not 'biography'
            'posts_count': profile_data.get('postsCount') if profile_data else None
        }
        
        if platform == "instagram":
            analytics = InstagramAnalytics(df, account_info)
        else:
            analytics = TikTokAnalytics(df, account_info)
        
        # Generate report
        report = analytics.generate_report()
        metrics = analytics.calculate_metrics()
        
        return {
            'username': username,
            'platform': platform,
            'metrics': metrics,
            'report': report.model_dump() if hasattr(report, 'model_dump') else report,
            'profile_data': profile_data
        }
        
    except Exception as e:
        console.print(f"[red]Error analyzing {platform} @{username}:[/red] {str(e)}")
        return None


def generate_summary_table(all_results: List[Dict]) -> Table:
    """Generate a summary table of all accounts."""
    table = Table(title="📊 All Accounts Summary", show_header=True, header_style="bold magenta")
    table.add_column("Platform", style="cyan")
    table.add_column("Account", style="green")
    table.add_column("Followers", justify="right")
    table.add_column("Posts", justify="right")
    table.add_column("Total Engagement", justify="right")
    table.add_column("Avg Views", justify="right")
    table.add_column("Engagement Rate", justify="right")
    table.add_column("Viral Score", justify="right")
    
    for result in all_results:
        if result and 'metrics' in result:
            metrics = result['metrics']
            # Use 'or {}' to handle None values from TikTok (no profile actor)
            profile_data = result.get('profile_data') or {}
            followers = profile_data.get('followersCount', 'N/A')
            followers_str = f"{followers:,}" if isinstance(followers, int) else followers
            
            # Use actual posts count from profile if available, otherwise use analyzed count
            posts_count = profile_data.get('postsCount', metrics.get('total_posts', 0))
            
            table.add_row(
                result['platform'].title(),
                f"@{result['username']}",
                followers_str,
                str(posts_count),
                f"{metrics.get('total_engagement', 0):,}",
                f"{metrics.get('average_views', 0):,.0f}",
                f"{metrics.get('engagement_rate', 0):.1f}%",
                f"{metrics.get('viral_velocity_score', 0):.1f}"
            )
    
    return table


def generate_detailed_csv_reports(all_results: List[Dict], output_dir: Path):
    """Generate detailed CSV reports for each account and a combined summary."""
    # Get date in the requested format
    now = datetime.now()
    date_str = now.strftime("%B_%d_%Y").lower()  # e.g., "december_7_2024"
    
    # Process each account individually
    for result in all_results:
        if result and 'metrics' in result:
            platform = result['platform']
            username = result['username']
            metrics = result['metrics']
            profile_data = result.get('profile_data', {})
            
            # Create filename with new format: username_platform_account.csv
            filename = f"{username}_{platform}_account.csv"
            filepath = output_dir / filename
            
            # Prepare data for CSV
            with open(filepath, 'w', newline='', encoding='utf-8') as f:
                # Write header comments
                f.write(f"# Social Media Analytics Report\n")
                f.write(f"# Generated on: {now.strftime('%B %d, %Y')}\n")
                f.write(f"# Account: @{username}\n")
                f.write(f"# Platform: {platform.title()}\n")
                f.write(f"# Total Posts Analyzed: {metrics.get('total_posts', 0)}\n")
                f.write(f"#\n")
                
                # Write CSV data
                writer = csv.writer(f)
                
                # Write headers
                headers = [
                    'account_name', 'platform', 'bio', 'full_name',
                    'total_posts', 'followers_count',
                    'avg_likes', 'avg_comments', 'avg_views',
                    'avg_views_videos', 'avg_views_non_videos',
                    'highest_views_count', 'highest_views_url',
                    'lowest_views_count', 'lowest_views_url',
                    'highest_likes_count', 'highest_likes_url',
                    'lowest_likes_count', 'lowest_likes_url',
                    'total_likes', 'total_views', 'total_comments'
                ]
                writer.writerow(headers)
                
                # Write data row
                # Handle potential None profile_data for platforms without profile actors
                safe_profile = profile_data or {}
                data_row = [
                    username,
                    platform,
                    safe_profile.get('biography', safe_profile.get('bio', '')),
                    safe_profile.get('fullName', safe_profile.get('full_name', '')),
                    metrics.get('total_posts', 0),
                    safe_profile.get('followersCount', safe_profile.get('followers_count', 'N/A')),
                    f"{metrics.get('average_likes', 0):.2f}",
                    f"{metrics.get('average_comments', 0):.2f}",
                    f"{metrics.get('average_views', 0):.2f}",
                    f"{metrics.get('average_views_videos', 0):.2f}",
                    f"{metrics.get('average_views_non_videos', 0):.2f}",
                    metrics.get('max_views_count', 0),
                    metrics.get('max_views_url', ''),
                    metrics.get('min_views_count', 0),
                    metrics.get('min_views_url', ''),
                    metrics.get('max_likes_count', 0),
                    metrics.get('max_likes_url', ''),
                    metrics.get('min_likes_count', 0),
                    metrics.get('min_likes_url', ''),
                    metrics.get('total_likes', 0),
                    metrics.get('total_views', 0),
                    metrics.get('total_comments', 0)
                ]
                writer.writerow(data_row)
            
            console.print(f"[green]✓[/green] Created CSV: {filename}")
    
    # Create combined CSV
    combined_filename = "all_accounts_combined.csv"
    combined_filepath = output_dir / combined_filename
    
    with open(combined_filepath, 'w', newline='', encoding='utf-8') as f:
        # Write header comments
        f.write(f"# Combined Social Media Analytics Report\n")
        f.write(f"# Generated on: {now.strftime('%B %d, %Y')}\n")
        f.write(f"# Total Accounts: {len(all_results)}\n")
        f.write(f"#\n")
        
        writer = csv.writer(f)
        
        # Same headers as individual files
        writer.writerow(headers)
        
        # Write data for all accounts
        for result in all_results:
            if result and 'metrics' in result:
                metrics = result['metrics']
                # Use 'or {}' to handle None values from platforms without profile actors
                profile_data = result.get('profile_data') or {}
                
                data_row = [
                    result['username'],
                    result['platform'],
                    profile_data.get('biography', profile_data.get('bio', '')),
                    profile_data.get('fullName', profile_data.get('full_name', '')),
                    metrics.get('total_posts', 0),
                    profile_data.get('followersCount', profile_data.get('followers_count', 'N/A')),
                    f"{metrics.get('average_likes', 0):.2f}",
                    f"{metrics.get('average_comments', 0):.2f}",
                    f"{metrics.get('average_views', 0):.2f}",
                    f"{metrics.get('average_views_videos', 0):.2f}",
                    f"{metrics.get('average_views_non_videos', 0):.2f}",
                    metrics.get('max_views_count', 0),
                    metrics.get('max_views_url', ''),
                    metrics.get('min_views_count', 0),
                    metrics.get('min_views_url', ''),
                    metrics.get('max_likes_count', 0),
                    metrics.get('max_likes_url', ''),
                    metrics.get('min_likes_count', 0),
                    metrics.get('min_likes_url', ''),
                    metrics.get('total_likes', 0),
                    metrics.get('total_views', 0),
                    metrics.get('total_comments', 0)
                ]
                writer.writerow(data_row)
        
        # Add summary row with enhanced totals
        writer.writerow([])  # Empty row
        
        # Calculate cumulative totals and find highest/lowest metrics
        total_posts_all = sum(r['metrics'].get('total_posts', 0) for r in all_results if r)
        
        # Calculate total followers (handle 'N/A' values)
        total_followers = 0
        for r in all_results:
            if r and r.get('profile_data'):
                followers = r['profile_data'].get('followersCount', r['profile_data'].get('followers_count', 0))
                if isinstance(followers, int):
                    total_followers += followers
        
        # Find highest and lowest metrics across all accounts
        all_max_views = [r['metrics'].get('max_views_count', 0) for r in all_results if r]
        all_min_views = [r['metrics'].get('min_views_count', float('inf')) for r in all_results if r and r['metrics'].get('min_views_count', 0) > 0]
        all_max_likes = [r['metrics'].get('max_likes_count', 0) for r in all_results if r]
        all_min_likes = [r['metrics'].get('min_likes_count', float('inf')) for r in all_results if r and r['metrics'].get('min_likes_count', 0) > 0]
        
        highest_views = max(all_max_views) if all_max_views else 0
        lowest_views = min(all_min_views) if all_min_views else 0
        highest_likes = max(all_max_likes) if all_max_likes else 0
        lowest_likes = min(all_min_likes) if all_min_likes else 0
        
        # Find URLs for highest/lowest metrics
        highest_views_url = ''
        lowest_views_url = ''
        highest_likes_url = ''
        lowest_likes_url = ''
        
        for r in all_results:
            if r and 'metrics' in r:
                if r['metrics'].get('max_views_count', 0) == highest_views:
                    highest_views_url = r['metrics'].get('max_views_url', '')
                if r['metrics'].get('min_views_count', 0) == lowest_views and lowest_views != float('inf'):
                    lowest_views_url = r['metrics'].get('min_views_url', '')
                if r['metrics'].get('max_likes_count', 0) == highest_likes:
                    highest_likes_url = r['metrics'].get('max_likes_url', '')
                if r['metrics'].get('min_likes_count', 0) == lowest_likes and lowest_likes != float('inf'):
                    lowest_likes_url = r['metrics'].get('min_likes_url', '')
        
        # Handle infinity values for display
        if lowest_views == float('inf'):
            lowest_views = 0
        if lowest_likes == float('inf'):
            lowest_likes = 0
        
        writer.writerow(['TOTALS', '', '', '', 
                        total_posts_all,  # Total posts cumulative
                        f"{total_followers:,}" if total_followers > 0 else 'N/A',  # Total followers cumulative
                        f"{sum(r['metrics'].get('average_likes', 0) for r in all_results if r):.2f}",
                        f"{sum(r['metrics'].get('average_comments', 0) for r in all_results if r):.2f}",
                        f"{sum(r['metrics'].get('average_views', 0) for r in all_results if r):.2f}",
                        '', '',  # Skip average views for videos/non-videos in totals
                        highest_views,  # Highest views from all accounts
                        highest_views_url,
                        lowest_views,  # Lowest views from all accounts
                        lowest_views_url,
                        highest_likes,  # Highest likes from all accounts
                        highest_likes_url,
                        lowest_likes,  # Lowest likes from all accounts
                        lowest_likes_url,
                        sum(r['metrics'].get('total_likes', 0) for r in all_results if r),
                        sum(r['metrics'].get('total_views', 0) for r in all_results if r),
                        sum(r['metrics'].get('total_comments', 0) for r in all_results if r)])
    
    console.print(f"[green]✓[/green] Created combined CSV: {combined_filename}")


def generate_combined_report(all_results: List[Dict], output_dir: Path):
    """Generate a combined PDF report for all accounts."""
    # Prepare data for report generator
    instagram_data = {}
    tiktok_data = {}
    
    for result in all_results:
        if result and 'metrics' in result:
            account_key = f"{result['username']}"
            if result['platform'] == 'instagram':
                instagram_data[account_key] = result['metrics']
            else:
                tiktok_data[account_key] = result['metrics']
    
    # Create multi-account report
    generator = SocialMediaReportGenerator()
    
    # Generate individual reports for each account
    for result in all_results:
        if result and 'metrics' in result:
            output_path = output_dir / f"{result['platform']}_{result['username']}_report.pdf"
            
            # Prepare data with profile info
            report_data = result['metrics'].copy()
            if result.get('profile_data'):
                report_data['profile_info'] = result['profile_data']
            
            if result['platform'] == 'instagram':
                generator.generate_report(
                    instagram_data=report_data,
                    output_path=str(output_path)
                )
            else:
                generator.generate_report(
                    tiktok_data=report_data,
                    output_path=str(output_path)
                )
    
    # Save combined summary
    summary_data = {
        'generated_at': datetime.now().isoformat(),
        'total_accounts': len(all_results),
        'accounts': all_results
    }
    
    with open(output_dir / 'all_accounts_summary.json', 'w') as f:
        json.dump(summary_data, f, indent=2, default=str)
    
    # Generate multi-account PDF report
    if all_results:
        multi_report_path = output_dir / 'all_accounts_report.pdf'
        generate_multi_account_report(all_results, str(multi_report_path))


def main():
    """Main function to analyze all accounts."""
    console.print("\n[bold cyan]🚀 Social Media Multi-Account Analyzer[/bold cyan]")
    console.print("=" * 50)
    
    # Check API token
    api_token = os.getenv("APIFY_API_TOKEN")
    if not api_token:
        console.print("[red]Error:[/red] APIFY_API_TOKEN not found in .env file!")
        sys.exit(1)
    
    # Remove quotes if present
    api_token = api_token.strip('"').strip("'")
    client = ApifyClient(api_token)
    
    # Load accounts and settings
    data = load_accounts()
    accounts = data
    settings = data.get('settings', {})
    total_accounts = len(accounts.get('instagram', [])) + len(accounts.get('tiktok', []))
    
    console.print(f"\n📋 Found {total_accounts} accounts to analyze:")
    console.print(f"   Instagram: {len(accounts.get('instagram', []))} accounts")
    console.print(f"   TikTok: {len(accounts.get('tiktok', []))} accounts")
    
    # Display settings
    console.print(f"\n⚙️  Settings:")
    console.print(f"   Instagram posts limit: {settings.get('instagram_posts_limit', 200)}")
    console.print(f"   TikTok videos limit: {settings.get('tiktok_videos_limit', 100)}")
    if settings.get('tiktok_date_limit'):
        console.print(f"   TikTok date limit: {settings['tiktok_date_limit']}")
    
    # Create output directory with date and timestamp format inside reports folder
    now = datetime.now()
    date_str = now.strftime("%B_%d_%Y").lower()  # e.g., "december_7_2024"
    time_str = now.strftime("%H%M%S")  # e.g., "143052" for 14:30:52
    
    # Create reports folder if it doesn't exist
    reports_dir = Path("reports")
    reports_dir.mkdir(exist_ok=True)
    
    # Create specific output directory inside reports folder
    output_dir = reports_dir / f"social_media_reports_{date_str}_{time_str}"
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    
    # Process all accounts
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        
        # Fetch data for all accounts
        console.print("\n[bold]📥 Step 1: Fetching data from social media platforms[/bold]")
        
        fetch_results = {}
        profile_results = {}
        
        # Instagram accounts
        for username in accounts.get('instagram', []):
            task = progress.add_task(f"Fetching Instagram @{username}...", total=1)
            dataset_id, success, profile_data = fetch_account_data(client, 'instagram', username, progress, task, settings)
            fetch_results[('instagram', username)] = dataset_id
            profile_results[('instagram', username)] = profile_data
            progress.update(task, completed=1)
            time.sleep(2)  # Be nice to the API
        
        # TikTok accounts  
        for username in accounts.get('tiktok', []):
            task = progress.add_task(f"Fetching TikTok @{username}...", total=1)
            dataset_id, success, profile_data = fetch_account_data(client, 'tiktok', username, progress, task, settings)
            fetch_results[('tiktok', username)] = dataset_id
            profile_results[('tiktok', username)] = None  # TikTok doesn't have separate profile actor yet
            progress.update(task, completed=1)
            time.sleep(2)  # Be nice to the API
    
    # Analyze all accounts
    console.print("\n[bold]📊 Step 2: Analyzing account data[/bold]")
    
    for (platform, username), dataset_id in fetch_results.items():
        if dataset_id:
            console.print(f"\nAnalyzing {platform} @{username}...")
            profile_data = profile_results.get((platform, username))
            result = analyze_account(client, platform, username, dataset_id, output_dir, profile_data)
            if result:
                all_results.append(result)
                console.print(f"[green]✓[/green] Completed {platform} @{username}")
                if result.get('profile_data'):
                    profile_data = result['profile_data']
                    followers = profile_data.get('followersCount', 'N/A')
                    if isinstance(followers, int):
                        console.print(f"  → Followers: {followers:,}")
                    else:
                        console.print(f"  → Followers: {followers}")
                    
                    if platform == 'instagram':
                        console.print(f"  → Total Posts on Profile: {profile_data.get('postsCount', 'N/A')}")
                    elif platform == 'tiktok':
                        console.print(f"  → Total Videos: {profile_data.get('postsCount', 'N/A')}")
                        console.print(f"  → Total Hearts: {profile_data.get('hearts', 'N/A'):,}")
                    
                    console.print(f"  → Posts Analyzed: {result['metrics'].get('total_posts', 0)}")
            else:
                console.print(f"[yellow]⚠[/yellow] No data for {platform} @{username}")
    
    # Generate summary
    console.print("\n[bold]📄 Step 3: Generating reports[/bold]")
    
    if all_results:
        # Display summary table
        summary_table = generate_summary_table(all_results)
        console.print("\n")
        console.print(summary_table)
        
        # Generate CSV reports instead of PDFs
        generate_detailed_csv_reports(all_results, output_dir)
        
        # Still save the summary JSON for reference
        summary_data = {
            'generated_at': datetime.now().isoformat(),
            'total_accounts': len(all_results),
            'accounts': all_results
        }
        
        with open(output_dir / 'all_accounts_summary.json', 'w') as f:
            json.dump(summary_data, f, indent=2, default=str)
        
        console.print(f"\n[green]✅ All reports saved to:[/green] {output_dir}/")
        console.print("\nGenerated files:")
        console.print(f"  • Individual CSV files for each account")
        console.print(f"  • Combined CSV with all accounts")
        console.print(f"  • all_accounts_summary.json - Complete data")
    else:
        console.print("\n[yellow]No data was successfully analyzed.[/yellow]")
    
    console.print("\n[bold green]✨ Analysis complete![/bold green]")


if __name__ == "__main__":
    main()