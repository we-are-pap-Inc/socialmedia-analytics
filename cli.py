#!/usr/bin/env python3
"""
Command-line interface for Social Media Analytics toolkit.

This module provides a Click-based CLI for analyzing Instagram and TikTok data.
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

import click
import pandas as pd
from apify_client import ApifyClient
from dotenv import load_dotenv
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.analytics.instagram import InstagramAnalytics
from src.analytics.tiktok import TikTokAnalytics
from src.ingest import load_instagram_data, load_tiktok_data
from src.metrics import calculate_all_metrics
from src.transform import transform_to_analytics_ready
from src.report_generator import generate_comprehensive_report

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    handlers=[RichHandler(rich_tracebacks=True)]
)

logger = logging.getLogger(__name__)
console = Console()


@click.group()
@click.option('--debug', is_flag=True, help='Enable debug logging')
def cli(debug: bool) -> None:
    """Social Media Analytics CLI - Analyze Instagram & TikTok data."""
    if debug:
        logging.getLogger().setLevel(logging.DEBUG)


@cli.command()
@click.option(
    '--platform',
    type=click.Choice(['instagram', 'tiktok'], case_sensitive=False),
    required=True,
    help='Social media platform'
)
@click.option(
    '--source',
    type=click.Path(exists=True),
    help='Path to JSON file with data'
)
@click.option(
    '--dataset-id',
    help='Apify dataset ID'
)
@click.option(
    '--api-token',
    envvar='APIFY_API_TOKEN',
    help='Apify API token (or set APIFY_API_TOKEN env var)'
)
@click.option(
    '--username',
    required=True,
    help='Account username'
)
@click.option(
    '--followers',
    type=int,
    help='Number of followers (for engagement rate calculation)'
)
@click.option(
    '--output',
    type=click.Path(),
    help='Output file path for metrics'
)
@click.option(
    '--format',
    type=click.Choice(['json', 'csv', 'parquet']),
    default='json',
    help='Output format'
)
def analyze(
    platform: str,
    source: Optional[str],
    dataset_id: Optional[str],
    api_token: Optional[str],
    username: str,
    followers: Optional[int],
    output: Optional[str],
    format: str
) -> None:
    """Analyze social media account data."""
    platform = platform.lower()
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:
        # Load data
        task = progress.add_task("Loading data...", total=None)
        
        try:
            if source:
                # Load from file
                if platform == 'instagram':
                    raw_data = load_instagram_data(source)
                else:
                    raw_data = load_tiktok_data(source)
            elif dataset_id and api_token:
                # Load from Apify
                client = ApifyClient(api_token)
                if platform == 'instagram':
                    raw_data = load_instagram_data(client, dataset_id)
                else:
                    raw_data = load_tiktok_data(client, dataset_id)
            else:
                console.print(
                    "[red]Error:[/red] Must provide either --source or --dataset-id with --api-token"
                )
                sys.exit(1)
                
            progress.update(task, description=f"Loaded {len(raw_data)} posts")
            
            # Check if data is empty or contains errors
            if len(raw_data) == 0:
                console.print("[red]Error:[/red] No data found. The account might be private or doesn't exist.")
                sys.exit(1)
            
            # Check for Apify error responses
            if len(raw_data) == 1 and isinstance(raw_data[0], dict) and 'error' in raw_data[0]:
                error_msg = raw_data[0].get('errorDescription', 'Unknown error')
                console.print(f"[red]Error from Apify:[/red] {error_msg}")
                console.print("\n[yellow]Tip:[/yellow] Make sure the account exists and is public.")
                sys.exit(1)
                
        except Exception as e:
            console.print(f"[red]Error loading data:[/red] {str(e)}")
            sys.exit(1)
        
        # Transform data
        task = progress.add_task("Transforming data...", total=None)
        df = transform_to_analytics_ready(raw_data, platform)
        progress.update(task, description="Data transformed")
        
        # Create analytics instance
        task = progress.add_task("Calculating metrics...", total=None)
        
        account_info = {
            'username': username,
            'platform': platform,
            'followers_count': followers
        }
        
        if platform == 'instagram':
            analytics = InstagramAnalytics(df, account_info)
        else:
            analytics = TikTokAnalytics(df, account_info)
        
        report = analytics.generate_report()
        progress.update(task, description="Metrics calculated")
    
    # Display results
    display_report(report.model_dump(), platform)
    
    # Save output if requested
    if output:
        analytics.export_metrics(output, format)
        console.print(f"\n[green]✓[/green] Metrics saved to {output}")


@cli.command()
@click.option(
    '--actor-id',
    required=True,
    help='Apify Actor ID'
)
@click.option(
    '--api-token',
    envvar='APIFY_API_TOKEN',
    help='Apify API token (or set APIFY_API_TOKEN env var)'
)
@click.option(
    '--input-file',
    type=click.Path(exists=True),
    help='JSON file with actor input'
)
@click.option(
    '--platform',
    type=click.Choice(['instagram', 'tiktok'], case_sensitive=False),
    required=True,
    help='Social media platform'
)
@click.option(
    '--wait',
    is_flag=True,
    help='Wait for actor to finish'
)
def fetch(
    actor_id: str,
    api_token: str,
    input_file: Optional[str],
    platform: str,
    wait: bool
) -> None:
    """Fetch data from Apify Actor."""
    if not api_token:
        console.print("[red]Error:[/red] API token required")
        sys.exit(1)
    
    client = ApifyClient(api_token)
    
    # Load input
    if input_file:
        with open(input_file, 'r') as f:
            run_input = json.load(f)
    else:
        # Default inputs
        if platform.lower() == 'instagram':
            run_input = {
                "resultsType": "posts",
                "resultsLimit": 200,
                "searchType": "user",
                "searchLimit": 1
            }
        else:
            run_input = {
                "resultsPerPage": 100,
                "profileScrapeSections": ["videos"]
            }
    
    with console.status("Starting Apify Actor...") as status:
        try:
            # Use 1 hour timeout (3600 seconds) for wait mode
            run = client.actor(actor_id).call(run_input=run_input, wait_secs=3600 if wait else None)
            
            if wait:
                console.print(f"[green]✓[/green] Actor finished. Dataset ID: {run['defaultDatasetId']}")
            else:
                console.print(f"[yellow]⏳[/yellow] Actor started. Run ID: {run['id']}")
                console.print(f"Dataset ID: {run['defaultDatasetId']}")
                
        except Exception as e:
            console.print(f"[red]Error:[/red] {str(e)}")
            sys.exit(1)


@cli.command()
@click.option(
    '--instagram',
    type=click.Path(exists=True),
    help='Instagram data file'
)
@click.option(
    '--tiktok',
    type=click.Path(exists=True),
    help='TikTok data file'
)
@click.option(
    '--output',
    type=click.Path(),
    default='comparison_report.json',
    help='Output file for comparison report'
)
def compare(
    instagram: Optional[str],
    tiktok: Optional[str],
    output: str
) -> None:
    """Compare performance across platforms."""
    if not instagram and not tiktok:
        console.print("[red]Error:[/red] Provide at least one data file")
        sys.exit(1)
    
    comparison = {}
    
    # Load and analyze Instagram
    if instagram:
        console.print("\n[bold]Instagram Analysis[/bold]")
        raw_data = load_instagram_data(instagram)
        df = transform_to_analytics_ready(raw_data, 'instagram')
        metrics = calculate_all_metrics(df)
        comparison['instagram'] = metrics
        display_metrics_summary(metrics, 'Instagram')
    
    # Load and analyze TikTok
    if tiktok:
        console.print("\n[bold]TikTok Analysis[/bold]")
        raw_data = load_tiktok_data(tiktok)
        df = transform_to_analytics_ready(raw_data, 'tiktok')
        metrics = calculate_all_metrics(df)
        comparison['tiktok'] = metrics
        display_metrics_summary(metrics, 'TikTok')
    
    # Save comparison
    with open(output, 'w') as f:
        json.dump(comparison, f, indent=2, default=str)
    
    console.print(f"\n[green]✓[/green] Comparison saved to {output}")


def display_report(report: dict, platform: str) -> None:
    """Display analytics report in a formatted way."""
    console.print(f"\n[bold cyan]📊 {platform.title()} Analytics Report[/bold cyan]")
    console.print(f"Account: @{report['account_info']['username']}")
    console.print(f"Generated: {report['generated_at']}")
    
    # Time period
    if report['time_period']['start']:
        console.print(
            f"Period: {report['time_period']['start'][:10]} to "
            f"{report['time_period']['end'][:10]} "
            f"({report['time_period']['days']} days)"
        )
    
    # Key metrics table
    table = Table(title="Key Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    metrics = report['metrics']
    table.add_row("Total Posts", str(metrics['total_posts']))
    table.add_row("Total Engagement", f"{metrics['total_engagement']:,}")
    table.add_row("Average Views", f"{metrics['average_views']:,.0f}")
    table.add_row("Average Likes", f"{metrics['average_likes']:,.0f}")
    table.add_row("Engagement Rate", f"{metrics['engagement_rate']:.2f}%")
    table.add_row("Viral Velocity Score", f"{metrics['viral_velocity_score']:.1f}")
    table.add_row("Consistency Index", f"{metrics['content_consistency_index']:.2f}")
    table.add_row("Peak Performance Ratio", f"{metrics['peak_performance_ratio']:.1f}x")
    
    console.print(table)
    
    # Posting frequency
    freq = metrics['posting_frequency']
    console.print(f"\n[bold]Posting Frequency:[/bold]")
    console.print(f"  • {freq['posts_per_day']:.2f} posts/day")
    console.print(f"  • {freq['avg_hours_between_posts']:.1f} hours between posts")
    
    # Top hashtags
    if metrics.get('top_hashtags'):
        console.print(f"\n[bold]Top Hashtags:[/bold]")
        for tag, score in list(metrics['top_hashtags'].items())[:5]:
            console.print(f"  • #{tag}: {score:.1f}")
    
    # Insights
    if report.get('insights'):
        console.print(f"\n[bold]Insights:[/bold]")
        for insight in report['insights']:
            console.print(f"  💡 {insight}")
    
    # Warnings
    if report.get('warnings'):
        console.print(f"\n[bold yellow]Warnings:[/bold yellow]")
        for warning in report['warnings']:
            console.print(f"  ⚠️  {warning}")


def display_metrics_summary(metrics: dict, platform: str) -> None:
    """Display a summary of metrics."""
    table = Table(title=f"{platform} Metrics", show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", justify="right")
    
    table.add_row("Total Posts", str(metrics['total_posts']))
    table.add_row("Total Engagement", f"{metrics['total_engagement']:,}")
    table.add_row("Average Views", f"{metrics['average_views']:,.0f}")
    table.add_row("Engagement Rate", f"{metrics['engagement_rate']:.2f}%")
    table.add_row("Growth Velocity", f"{metrics['growth_velocity']:.1f}%")
    
    console.print(table)


@cli.command()
@click.option(
    '--source',
    type=click.Path(exists=True),
    required=True,
    help='Data file to validate'
)
@click.option(
    '--platform',
    type=click.Choice(['instagram', 'tiktok'], case_sensitive=False),
    required=True,
    help='Platform schema to validate against'
)
@click.option(
    '--strict',
    is_flag=True,
    help='Fail on any validation error'
)
def validate(source: str, platform: str, strict: bool) -> None:
    """Validate data file against platform schema."""
    from src.ingest import load_from_json, validate_data_schema
    
    console.print(f"Validating {source} against {platform} schema...")
    
    try:
        data = load_from_json(source)
        validated = validate_data_schema(data, platform, strict)
        
        console.print(f"[green]✓[/green] Validated {len(validated)}/{len(data)} items")
        
        if len(validated) < len(data):
            console.print(
                f"[yellow]⚠️[/yellow] {len(data) - len(validated)} items failed validation"
            )
            
    except Exception as e:
        console.print(f"[red]✗[/red] Validation failed: {str(e)}")
        sys.exit(1)


@cli.command()
@click.option(
    '--instagram-csv',
    type=click.Path(exists=True),
    help='Instagram metrics CSV file'
)
@click.option(
    '--tiktok-csv',
    type=click.Path(exists=True),
    help='TikTok metrics CSV file'
)
@click.option(
    '--comparison-json',
    type=click.Path(exists=True),
    help='Comparison JSON file'
)
@click.option(
    '--output',
    type=click.Path(),
    default='social_media_report.pdf',
    help='Output PDF file path'
)
def report(
    instagram_csv: Optional[str],
    tiktok_csv: Optional[str],
    comparison_json: Optional[str],
    output: str
) -> None:
    """Generate a comprehensive PDF report from analytics data."""
    if not any([instagram_csv, tiktok_csv, comparison_json]):
        console.print("[red]Error:[/red] Provide at least one data file")
        sys.exit(1)
    
    with console.status(f"Generating PDF report to {output}...") as status:
        try:
            output_path = generate_comprehensive_report(
                instagram_metrics_path=instagram_csv,
                tiktok_metrics_path=tiktok_csv,
                comparison_path=comparison_json,
                output_path=output
            )
            
            console.print(f"[green]✓[/green] Report generated: {output_path}")
            
        except Exception as e:
            console.print(f"[red]Error generating report:[/red] {str(e)}")
            sys.exit(1)


if __name__ == '__main__':
    cli()