## Automated Daily Pipeline

A GitHub Actions workflow runs daily at 02:00 UTC to update the historical dataset with new Tennis Abstract and API-Tennis data, extract features, and cache results.

### Workflow File

`.github/workflows/daily_pipeline.yml` orchestrates:
1. Checkout code.
2. Install dependencies (`requirements.txt`, `requirements-dev.txt`).
3. Run a Python snippet to load or generate comprehensive historical data, integrate API-Tennis incremental updates, and save to cache.
4. Archive logs under `logs/`.

### Environment Variables

- `API_TENNIS_KEY`: Your API-Tennis key for fetching match data.
- `GITHUB_TOKEN`: (Optional) Used by GitHub Actions for authentication.

Set them before running locally or in CI:
```bash
export API_TENNIS_KEY="your_secret_key_here"
export GITHUB_TOKEN="${{ secrets.GITHUB_TOKEN }}"