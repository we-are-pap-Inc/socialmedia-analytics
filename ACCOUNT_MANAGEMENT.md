# Account Management Guide

## Quick Start: How to Update Accounts

### Option 1: Local Deployment / Self-Hosted
✅ **Use the built-in UI manager** in the sidebar:
1. Open the dashboard
2. In the sidebar, click "📝 Manage Accounts"
3. Add or remove accounts as needed
4. Changes are saved automatically

### Option 2: Streamlit Cloud Deployment
⚠️ **Use Streamlit Secrets** (files don't persist on cloud):

1. **In Streamlit Cloud Dashboard:**
   - Go to your app
   - Click "Settings" → "Secrets"
   - Add this configuration:

```toml
APIFY_TOKEN = "your_actual_token"

[accounts]
instagram = ["username1", "username2", "username3"]
tiktok = ["tiktok_user1", "tiktok_user2"]

[accounts.settings]
instagram_posts_limit = 200
tiktok_videos_limit = 100
```

2. **To Update Accounts:**
   - Go back to Settings → Secrets
   - Edit the `instagram` or `tiktok` arrays
   - Click "Save"
   - The app will automatically restart with new accounts

### Option 3: For Technical Users - Environment-Based

Create different configurations for different teams/clients:

**Production (`secrets.toml`):**
```toml
[accounts]
instagram = ["brand_account", "ceo_account", "product_account"]
tiktok = ["brand_tiktok"]
```

**Development (`secrets.dev.toml`):**
```toml
[accounts]
instagram = ["test_account1", "test_account2"]
tiktok = ["test_tiktok"]
```

## Best Practices

### 1. Account Naming
- Use exact usernames (without @)
- Double-check spelling
- Ensure accounts are public

### 2. Limits Management
- Start with lower limits (50-100 posts) for testing
- Increase gradually based on needs
- Monitor Apify credit usage

### 3. For Teams
- Create separate deployments for different teams/clients
- Use descriptive account groupings
- Document which accounts belong to which projects

## Troubleshooting

### "Account not found"
- Verify the account is public
- Check exact spelling (case-sensitive)
- Try without special characters

### "Changes not saving" (Streamlit Cloud)
- Remember: File changes don't persist on Streamlit Cloud
- Always use the Secrets configuration
- Refresh the page after updating secrets

### "Too many accounts"
- Limit to 10-15 accounts per dashboard
- Create separate dashboards for different teams
- Use the multiselect to analyze subsets

## Quick Reference

### Add Account (Local)
1. Sidebar → "📝 Manage Accounts"
2. Select platform
3. Enter username
4. Click "➕ Add Account"

### Add Account (Cloud)
1. Settings → Secrets
2. Edit `instagram = [...]` or `tiktok = [...]`
3. Save

### Remove Account (Local)
1. Sidebar → "📝 Manage Accounts"
2. Select account from dropdown
3. Click "🗑️ Remove Account"

### Remove Account (Cloud)
1. Settings → Secrets
2. Remove username from array
3. Save