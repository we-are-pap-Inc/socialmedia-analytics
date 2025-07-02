# Social Media Analytics Engine 🚀

A powerful, modular ETL pipeline for extracting, analyzing, and reporting on Instagram and TikTok account metrics. Built with clean architecture principles and designed for scalability.

## 🎯 What It Does

Transform your social media data into actionable insights with:
- **Multi-account analysis** across Instagram and TikTok
- **Comprehensive metrics** including engagement rates, viral scores, and content performance
- **Profile enrichment** with follower counts and bio information
- **Clean CSV exports** with all the data you need for further analysis
- **Automated data fetching** via Apify actors

## 🏗️ Architecture Overview

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Apify Cloud   │     │  Transform Layer │     │  Analytics Layer│
│                 │     │                 │     │                 │
│ Instagram Actor ├────►│ Unified Schema  ├────►│ Metrics Engine  │
│ TikTok Actor    │     │ Normalization   │     │ Platform Logic  │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                          │
                                                          ▼
                                                ┌─────────────────┐
                                                │   CSV Reports   │
                                                │ Individual Stats│
                                                │ Combined Summary│
                                                └─────────────────┘
```

### Data Flow
1. **Extract**: Apify actors fetch raw data from Instagram/TikTok APIs
2. **Transform**: Platform-specific data normalized into unified schema
3. **Analyze**: Metrics calculated with platform-specific enhancements
4. **Export**: Clean CSV files with comprehensive statistics

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Poetry (for dependency management)
- Apify account with API token

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd social-media-analytics

# Install dependencies
poetry install

# Set up environment
echo 'APIFY_API_TOKEN="your_api_token_here"' > .env
```

### Basic Usage

#### Option 1: Interactive Dashboard (NEW! 🚀)

1. **Configure your accounts** in `accounts.json`:
```json
{
  "instagram": ["account1", "account2"],
  "tiktok": ["account1"],
  "settings": {
    "instagram_posts_limit": 200,
    "tiktok_videos_limit": 100
  }
}
```

2. **Run the Streamlit dashboard**:
```bash
poetry run streamlit run app.py
```

3. **Use the dashboard**:
   - Select accounts from the sidebar
   - Adjust scraping limits
   - Click "🔄 Refresh Data" to fetch fresh data
   - View interactive charts and metrics
   - Download CSV reports

#### Option 2: Command Line Analysis

1. **Run the batch analysis**:
```bash
poetry run python analyze_all_accounts.py
```

2. **Find your reports** in `reports/social_media_reports_*`:
```
├── account1_instagram_account.csv
├── account1_tiktok_account.csv
└── all_accounts_combined.csv
```

## 📁 Project Structure

```
social-media-analytics/
├── src/
│   ├── ingest.py          # Data loading & validation (Pydantic schemas)
│   ├── transform.py       # Platform → Unified schema transformation
│   ├── metrics.py         # Core metrics calculations
│   ├── report_generator.py # PDF/CSV generation logic
│   ├── analytics/
│   │   ├── base.py        # Abstract analytics class
│   │   ├── instagram.py   # Instagram-specific metrics
│   │   └── tiktok.py      # TikTok-specific metrics
│   └── utils/
│       ├── helpers.py     # Utility functions
│       └── validators.py  # Data validation helpers
├── analyze_all_accounts.py # Main entry point
├── cli.py                 # CLI for advanced operations
├── accounts.json          # Account configuration
└── pyproject.toml         # Dependencies & project config
```

## 🔧 Core Components

### Data Ingestion (`src/ingest.py`)
- **Purpose**: Load and validate social media data
- **Key Features**:
  - Pydantic models for data validation
  - Support for Apify datasets and local JSON files
  - Platform-specific schema validation
- **Usage**:
```python
from src.ingest import load_instagram_data
data = load_instagram_data(apify_client, dataset_id)
```

### Data Transformation (`src/transform.py`)
- **Purpose**: Normalize platform-specific data into unified schema
- **Design Pattern**: Adapter pattern for cross-platform compatibility
- **Key Functions**:
  - `normalize_instagram_data()`: Instagram → Unified
  - `normalize_tiktok_data()`: TikTok → Unified
  - `create_unified_schema()`: Final transformation
- **Unified Schema Fields**:
  - Core: `post_id`, `timestamp`, `caption`, `url`
  - Engagement: `likes_count`, `comments_count`, `views_count`
  - Metadata: `hashtags`, `mentions`, `is_video`

### Metrics Engine (`src/metrics.py`)
- **Purpose**: Calculate comprehensive analytics metrics
- **Categories**:
  - **Basic**: totals, averages, min/max values
  - **Advanced**: viral velocity, consistency index, retention rate
  - **Comparison**: video vs non-video performance
- **Key Metrics**:
```python
# Examples of available metrics
average_likes()           # Average likes per post
total_engagement()        # Sum of all interactions
viral_velocity_score()    # Viral potential indicator
get_post_with_max_views() # Returns (count, url)
```

### Platform Analytics (`src/analytics/`)
- **Base Class**: `AccountAnalytics` - abstract interface
- **Instagram**: Reels analysis, music usage, sponsored content
- **TikTok**: Share metrics, sound trends, viral patterns
- **Design Pattern**: Template Method for extensibility

## 📊 CSV Output Format

Each CSV includes a descriptive header:
```csv
# Social Media Analytics Report
# Generated on: December 7, 2024
# Account: @username
# Platform: Instagram
# Total Posts Analyzed: 185
#
account_name,platform,bio,followers_count,avg_likes,avg_views,...
```

### Key Columns
- **Profile**: `account_name`, `bio`, `followers_count`
- **Averages**: `avg_likes`, `avg_comments`, `avg_views`
- **Totals**: `total_likes`, `total_views`, `total_comments`
- **Extremes**: `highest_views_url`, `lowest_likes_count`
- **Segmented**: `avg_views_videos`, `avg_views_non_videos`

## 🛠️ Configuration

### Environment Variables

#### Option 1: Using .env file
```bash
echo 'APIFY_API_TOKEN="your_api_token_here"' > .env
```

#### Option 2: Using Streamlit secrets (for dashboard)
```bash
# Create .streamlit/secrets.toml
mkdir -p .streamlit
echo 'APIFY_TOKEN = "your_api_token_here"' > .streamlit/secrets.toml
```

### Apify Actor IDs
```python
INSTAGRAM_ACTOR_ID = "shu8hvrXbJbY3Eb9W"        # Posts scraper
INSTAGRAM_PROFILE_ACTOR_ID = "dSCLg0C3YEZ83HzYX" # Profile data
TIKTOK_ACTOR_ID = "OtzYfK1ndEGdwWFKQ"           # Videos scraper
```

### Settings in `accounts.json`
- `instagram_posts_limit`: Max posts per account (default: 200)
- `tiktok_videos_limit`: Max videos per account (default: 100)
- `tiktok_date_limit`: Date filter (e.g., "7 days", "2024-01-01")

## 🔍 Advanced Usage

### CLI Commands
```bash
# Fetch data manually
poetry run python cli.py fetch \
  --actor-id shu8hvrXbJbY3Eb9W \
  --input-file instagram_input.json \
  --platform instagram \
  --wait

# Analyze specific dataset
poetry run python cli.py analyze \
  --platform instagram \
  --dataset-id DATASET_ID \
  --username account1
```

### Extending the Framework

1. **Add New Metrics**:
```python
# In src/metrics.py
def custom_metric(df: pd.DataFrame) -> float:
    """Your custom calculation."""
    return df['some_column'].apply(custom_logic).mean()
```

2. **Platform-Specific Analytics**:
```python
# In src/analytics/instagram.py
def _analyze_custom_feature(self) -> Dict[str, Any]:
    """Platform-specific analysis."""
    return {'custom_score': calculate_score(self.data)}
```

## 🧪 Development

### Code Style
- **Formatter**: Black (line length: 88)
- **Linter**: Ruff with comprehensive rule set
- **Type Checking**: MyPy with strict mode
- **Docstrings**: NumPy style

### Testing
```bash
# Run tests with coverage
poetry run pytest

# Type checking
poetry run mypy src/

# Linting
poetry run ruff check src/
```

### Design Patterns Used
- **Factory Pattern**: Analytics class instantiation
- **Strategy Pattern**: Platform-specific calculations
- **Template Method**: Base analytics workflow
- **Adapter Pattern**: Data normalization

## 📈 Metrics Glossary

- **Engagement Rate**: (Total engagements / followers) × 100
- **Viral Velocity Score**: Measures content momentum (0-100)
- **Consistency Index**: Posting regularity score (0-1)
- **Peak Performance Ratio**: Top 10% posts vs average
- **Retention Rate**: Audience engagement over time

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on [Apify](https://apify.com) actors for reliable data extraction
- Uses [Poetry](https://python-poetry.org) for modern dependency management
- Styled with [Rich](https://github.com/Textualize/rich) for beautiful CLI output