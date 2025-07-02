#!/usr/bin/env python3
"""
Test script to verify the dashboard is ready for deployment.
Run this before deploying to ensure everything works.
"""

import sys
import os
from pathlib import Path

def check_requirements():
    """Check if all requirements are met."""
    print("🔍 Checking requirements...\n")
    
    issues = []
    
    # Check Python version
    print("✓ Python version:", sys.version.split()[0])
    
    # Check required files
    required_files = [
        "app.py",
        "accounts.json",
        "pyproject.toml",
        "src/ingest.py",
        "src/transform.py",
        "src/metrics.py",
        "src/data_io.py",
        "src/analytics/base.py",
        "src/analytics/instagram.py",
        "src/analytics/tiktok.py",
    ]
    
    print("\n📁 Checking required files:")
    for file in required_files:
        if Path(file).exists():
            print(f"  ✓ {file}")
        else:
            print(f"  ✗ {file} - MISSING!")
            issues.append(f"Missing file: {file}")
    
    # Check imports
    print("\n📦 Checking imports:")
    try:
        import streamlit
        print("  ✓ streamlit", streamlit.__version__)
    except ImportError:
        print("  ✗ streamlit - NOT INSTALLED!")
        issues.append("Streamlit not installed")
    
    try:
        import plotly
        print("  ✓ plotly", plotly.__version__)
    except ImportError:
        print("  ✗ plotly - NOT INSTALLED!")
        issues.append("Plotly not installed")
    
    try:
        import pandas
        print("  ✓ pandas", pandas.__version__)
    except ImportError:
        print("  ✗ pandas - NOT INSTALLED!")
        issues.append("Pandas not installed")
    
    try:
        from apify_client import ApifyClient
        print("  ✓ apify_client")
    except ImportError:
        print("  ✗ apify_client - NOT INSTALLED!")
        issues.append("Apify client not installed")
    
    # Check API token configuration
    print("\n🔑 Checking API token configuration:")
    
    # Check .env file
    if Path(".env").exists():
        print("  ✓ .env file exists")
        with open(".env", "r") as f:
            if "APIFY_API_TOKEN" in f.read():
                print("  ✓ APIFY_API_TOKEN found in .env")
            else:
                print("  ⚠️  APIFY_API_TOKEN not found in .env")
                issues.append("APIFY_API_TOKEN not set in .env")
    else:
        print("  ⚠️  .env file not found")
    
    # Check Streamlit secrets
    secrets_path = Path(".streamlit/secrets.toml")
    if secrets_path.exists():
        print("  ✓ .streamlit/secrets.toml exists")
    else:
        print("  ⚠️  .streamlit/secrets.toml not found (needed for Streamlit Cloud)")
    
    # Check accounts.json
    print("\n👥 Checking accounts configuration:")
    try:
        import json
        with open("accounts.json", "r") as f:
            accounts = json.load(f)
        
        ig_count = len(accounts.get("instagram", []))
        tt_count = len(accounts.get("tiktok", []))
        
        print(f"  ✓ Instagram accounts: {ig_count}")
        print(f"  ✓ TikTok accounts: {tt_count}")
        
        if ig_count == 0 and tt_count == 0:
            issues.append("No accounts configured in accounts.json")
            
    except Exception as e:
        print(f"  ✗ Error reading accounts.json: {e}")
        issues.append("Cannot read accounts.json")
    
    # Check data directory
    print("\n📂 Checking data directory:")
    if Path("data").exists():
        print("  ✓ data/ directory exists")
    else:
        print("  ⚠️  data/ directory not found (will be created on first run)")
    
    # Summary
    print("\n" + "="*50)
    if issues:
        print("❌ Issues found:")
        for issue in issues:
            print(f"  - {issue}")
        print("\nPlease fix these issues before deployment.")
        return False
    else:
        print("✅ All checks passed! Your dashboard is ready for deployment.")
        print("\nNext steps:")
        print("1. Test locally: poetry run streamlit run app.py")
        print("2. Follow DEPLOYMENT_GUIDE.md for deployment options")
        return True


def test_imports():
    """Test if all modules can be imported successfully."""
    print("\n🧪 Testing module imports...")
    
    try:
        from src.ingest import scrape_instagram, scrape_tiktok
        from src.transform import transform_to_analytics_ready
        from src.metrics import calculate_all_metrics
        from src.data_io import get_latest_run_dir, save_run
        from src.analytics.instagram import InstagramAnalytics
        from src.analytics.tiktok import TikTokAnalytics
        print("✓ All modules imported successfully!")
        return True
    except Exception as e:
        print(f"✗ Import error: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Social Media Analytics Dashboard - Pre-deployment Check\n")
    
    # Run checks
    requirements_ok = check_requirements()
    
    if requirements_ok:
        imports_ok = test_imports()
        
        if imports_ok:
            print("\n🎉 Everything looks good!")
            print("\nTo run the dashboard locally:")
            print("  poetry run streamlit run app.py")
            print("\nTo deploy, see DEPLOYMENT_GUIDE.md")
        else:
            print("\n⚠️  Fix import issues before proceeding.")
    else:
        print("\n⚠️  Fix the issues above before proceeding.")