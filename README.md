# Viral Product Hunter

A beginner-friendly FastAPI and Streamlit product research app that discovers trending products, sources suppliers from Alibaba and AliExpress, and ranks them by viral potential and supplier quality.

## Workflows

### Setup

```bash
uv venv viral-product-hunter
```

### Activation

- **macOS/Linux**: `source viral-product-hunter/bin/activate`
- **Windows**: `viral-product-hunter\Scripts\activate`

### Install dependencies

```bash
uv pip install -r requirements.txt
```

### Environment file

```bash
cp .env.example .env
```

Edit `.env` with your API keys:
- `DECODO_AUTH_TOKEN` (required): Your Decodo scraper API token
- `OPENAI_API_KEY` (optional): OpenAI API key for AI ranking
- `KIMI_API_KEY` / `MOONSHOT_API_KEY` (optional): Kimi or Moonshot API keys for AI ranking
- `AI_MODEL_API_KEY` / `AI_MODEL_API_URL` / `AI_MODEL_NAME` (optional): Generic AI provider
- `DECODO_SUPPLIER_MAX_WORKERS=10` (default): Concurrent supplier scraping workers
- `BACKEND_API_URL=http://127.0.0.1:8000` (default): Backend URL for Streamlit
- `BACKEND_READ_TIMEOUT=600` (default): Backend read timeout in seconds

### Run the backend

```bash
uv run api.py or python api.py
```

API URL: http://127.0.0.1:8000
API docs: http://127.0.0.1:8000/docs

### Run the frontend

```bash
uv run streamlit run streamlit_app.py
```

Streamlit URL: http://localhost:8501

## API Endpoints

### POST /hunt

Complete synchronous pipeline. Returns:
- `niche`: The product niche
- `message`: Completion message
- `initial_products`: List of products ranked by initial viral score (from discovery and AI ranking)
- `final_products`: List of products with final supplier-aware ranking
- `supplier_summary`: Summary of supplier sourcing results
- `supplier_data`: Detailed supplier data per product
- `discovery_summary`: Summary of discovery source results
- `discovery_data`: Compacted raw discovery data
- `ranking_warnings`: List of any warnings from ranking or sourcing

Each `initial_products` entry includes: `product`, `source`, `price`, `currency`, `url`, `image`, `viral_score`, `ai_viral_score`, `ai_reason`, `rating`, `reviews`, `sales`, `evidence`, `badges`, `asin`.

Each `final_products` entry includes all initial product fields plus: `final_score`, `selected_supplier_offer`, `supplier_price_reasoning`, `rejection_reason`, `ranking_method` ("ai" or "deterministic_fallback").

The `selected_supplier_offer` when present includes: `product_reference`, `supplier_platform`, `supplier_title`, `currency`, `unit_price` or `price_range`, `min_order_quantity`, `shipping_info`, `supplier_url`, `rating`, `orders`, `reviews`, `supplier_evidence`, `provenance`.

### POST /discover

Run discovery only: web scraping, compaction, summarization, product extraction, and initial AI ranking.

Request: `{"niche": "your niche"}`

Returns: `niche`, `initial_products`, `discovery_summary`, `discovery_data`, `ranking_warnings`, `message`.

### POST /source

Run supplier sourcing and final ranking for already-discovered products.

Request: `{"niche": "your niche", "initial_products": [...]}`

Returns: `niche`, `final_products`, `supplier_summary`, `supplier_data`, `ranking_warnings`, `message`.

## Staged Progress in Streamlit

The Streamlit frontend uses `/discover` followed by `/source` to show visible stage progress:

1. **Discovering products...** - POST to `/discover`, shows count of ranked candidates
2. **Sourcing Alibaba and AliExpress offers...** - POST to `/source`, shows supplier metrics
3. **Complete!** - Displays all three ranking sections

## Ranking

### Initial Viral Ranking

Products are first ranked by viral score (0-100) based on:
- Price between $5-$50: +12 points
- Rating: up to +15 points (rating * 3)
- Reviews/Sales volume: up to +18 points (log10(volume) * 5)
- Has image: +5 points
- Has URL: +5 points
- Best Seller / Amazon's Choice: +10 points

The top 10 candidates are then re-ranked by AI considering: problem solving, video potential, emotional appeal, shipping ease, competition, bundling, supplier evidence, and year-round demand.

### Final Supplier-Aware Ranking

The final ranking considers both viral potential and supplier quality. The AI evaluates:
- Viral potential (from initial ranking)
- Credible supplier cost (converted to USD)
- Evidence quality (badges, ratings, reviews)
- MOQ (lower is better)
- Shipping practicality
- Potential margin
- Availability across marketplaces

If AI final ranking is unavailable, a deterministic fallback is used with the following weights:
- Initial viral score: 30%
- Supplier cost: 18%
- Supplier evidence: 15%
- MOQ: 10%
- Shipping: 8%
- Estimated margin: 12%
- Cross-platform availability: 7%

Products without credible supplier offers remain in the output with zero supplier components and a rejection reason.

## Configuration Defaults

- `DECODO_API_URL`: https://scraper-api.decodo.com/v2/scrape
- `DECODO_REQUEST_TIMEOUT`: 120 seconds
- `DECODO_MAX_WORKERS`: 5 (discovery sources)
- `DECODO_SUPPLIER_MAX_WORKERS`: 10 (supplier marketplace requests)
- `DECODO_PROXY_POOL`: premium
- AI provider timeout: 240 seconds for Kimi/Moonshot, 60 seconds for others

## Testing

Run the complete test suite:

```bash
python -m unittest -v test_api.py
```

All tests are mocked and do not require live Decodo or AI API calls. Set `DECODO_AUTH_TOKEN` to any non-empty value before importing `api.py` in tests.

## Principal Project Files

- `api.py`: FastAPI backend with discovery, supplier sourcing, and ranking logic
- `streamlit_app.py`: Streamlit frontend with staged progress display
- `test_api.py`: Mocked unit tests for all functionality
- `requirements.txt`: Python dependencies
- `.env.example`: Environment variable template
- `README.md`: This documentation
