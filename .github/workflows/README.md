# GitHub Actions Workflows

## 📊 Update HDB Data (Delta)

**Workflow File:** `.github/workflows/update-hdb-data.yml`

### Overview

This workflow automatically updates your HDB resale data from the [data.gov.sg API](https://data.gov.sg/datasets/d_8b84c4ee58e3cfc0ece0d773c8ca6abc/view) and regenerates precomputed market segmentation data.

### When It Runs

1. **Scheduled**: Every Monday, Wednesday, Friday at 2 AM UTC (10 AM SGT)
   ```yaml
   schedule:
     - cron: '0 2 * * 1,3,5'
   ```

2. **Manual Trigger**: You can manually trigger it from the GitHub Actions tab
   - Go to: **Actions** → **Update HDB Data (Delta)** → **Run workflow**

3. **On Push**: Runs when you update the workflow file or the update script

### What It Does

1. ✅ Checks out your repository
2. ✅ Sets up Python 3.10
3. ✅ Installs dependencies from `requirements.txt`
4. ✅ Runs `scripts/update_hdb_data_delta.py` to fetch new data (delta/incremental sync)
5. ✅ Regenerates `5-4_precomputed_market.pkl` if data was updated
6. ✅ Commits and pushes changes back to the repository

### Files That Get Updated

- `raw_data_main.csv` - Main HDB resale dataset
- `5-4_precomputed_market.pkl` - Precomputed market segmentation data
- `.last_sync_date.txt` - Last sync timestamp (for delta updates)

### How It Works

1. **First Run**: Performs a **full sync** (fetches all data from API)
   - Creates `.last_sync_date.txt` with the latest month

2. **Subsequent Runs**: Performs **incremental sync** (only new data since last sync)
   - Reads `.last_sync_date.txt` to know where to start
   - Only fetches records with `month >= last_sync_month`
   - Much faster than full sync!

### Requirements

Your repository needs:
- ✅ `scripts/update_hdb_data_delta.py` - Delta update script
- ✅ `5-2_precompute_market.py` - Market segmentation precomputation script
- ✅ `requirements.txt` - Python dependencies
- ✅ `raw_data_main.csv` - Existing data file (will be created on first run)

**Optional** (for market segmentation):
- `hybrid_multi_metric_pipeline_50K.pkl` - Trained pipeline for clustering
  - If missing, the precomputation step will skip (but the workflow will still succeed)

### Setting Up

1. **Push this workflow file** to your GitHub repository:
   ```bash
   git add .github/workflows/update-hdb-data.yml
   git commit -m "Add GitHub Actions workflow for automatic data updates"
   git push
   ```

2. **Enable GitHub Actions** in your repository settings:
   - Go to: **Settings** → **Actions** → **General**
   - Ensure "Allow all actions and reusable workflows" is enabled

3. **Verify the workflow runs**:
   - Go to: **Actions** tab in your GitHub repository
   - You should see "Update HDB Data (Delta)" workflow
   - Click "Run workflow" to test it manually

### Viewing Workflow Runs

1. Go to the **Actions** tab in your GitHub repository
2. Click on **Update HDB Data (Delta)** to see all runs
3. Click on a specific run to see detailed logs

### Troubleshooting

**Problem**: Workflow fails with "Permission denied"
- **Solution**: Ensure GitHub Actions has write permissions:
  - Go to: **Settings** → **Actions** → **General** → **Workflow permissions**
  - Select "Read and write permissions"

**Problem**: Workflow runs but no data is updated
- **Solution**: Check the logs to see if there's actually new data available from the API
- The workflow will show "No new data to update" if there are no changes

**Problem**: Precomputation step fails
- **Solution**: Check if `hybrid_multi_metric_pipeline_50K.pkl` exists in your repository
- If missing, you may need to upload it or skip the precomputation step

### Manual Testing

You can test the update script locally:

```bash
# Full sync (first time)
python scripts/update_hdb_data_delta.py --full

# Incremental sync (subsequent runs)
python scripts/update_hdb_data_delta.py
```

### Notes

- ⚠️ The workflow uses `[skip ci]` in commit messages to prevent infinite loops
- ✅ The workflow will only commit changes if data was actually updated
- ✅ Delta sync is much faster than full sync (fetches only new records)
- 📅 Last sync date is stored in `.last_sync_date.txt` (tracked in Git)
