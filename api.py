"""
FastAPI backend for the Viral Product Hunter application.
"""
import copy
import html
import json
import math
import os
import re
import time
from html.parser import HTMLParser
from typing import Any
from urllib.parse import quote_plus, urljoin, urlparse, parse_qs

from concurrent.futures import ThreadPoolExecutor

import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
from dotenv import load_dotenv

load_dotenv()

BACKEND_PORT = 8000

DECODO_API_URL = os.getenv("DECODO_API_URL") or "https://scraper-api.decodo.com/v2/scrape"
DECODO_AUTH_TOKEN = os.getenv("DECODO_AUTH_TOKEN")
DECODO_REQUEST_TIMEOUT = int(os.getenv("DECODO_REQUEST_TIMEOUT") or "120")
DECODO_MAX_WORKERS = int(os.getenv("DECODO_MAX_WORKERS") or "5")
DECODO_PROXY_POOL = os.getenv("DECODO_PROXY_POOL") or "premium"
DECODO_SUPPLIER_MAX_WORKERS = int(os.getenv("DECODO_SUPPLIER_MAX_WORKERS") or "10")

if not DECODO_AUTH_TOKEN:
    raise ValueError("Missing DECODO_AUTH_TOKEN in .env")

# AI Model Configuration
DEFAULT_AI_MODEL_API_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_AI_MODEL_NAME = "gpt-4o-mini"
DEFAULT_KIMI_MODEL_API_URL = "https://api.moonshot.ai/v1/chat/completions"
DEFAULT_KIMI_MODEL_NAME = "kimi-k3"
DEFAULT_KIMI_CODE_API_URL = "https://api.kimi.com/coding/v1/chat/completions"
DEFAULT_KIMI_CODE_MODEL_NAME = "k3-256k"

KIMI_API_URL = os.getenv("KIMI_API_URL")
KIMI_API_KEY = os.getenv("KIMI_API_KEY")
MOONSHOT_API_KEY = os.getenv("MOONSHOT_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
AI_MODEL_API_KEY = os.getenv("AI_MODEL_API_KEY")
AI_MODEL_API_URL = os.getenv("AI_MODEL_API_URL") or DEFAULT_AI_MODEL_API_URL
AI_MODEL_NAME = os.getenv("AI_MODEL_NAME") or DEFAULT_AI_MODEL_NAME

app = FastAPI(title="Viral Product Hunter API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# AI Provider Configuration
# ============================================================

def _is_valid_key(key: str | None) -> bool:
    """Check if API key is non-empty and not a placeholder."""
    if not key or not isinstance(key, str):
        return False
    cleaned = key.strip()
    if not cleaned or cleaned.lower().startswith("your_"):
        return False
    return True


def _get_kimi_providers():
    """Build list of Kimi providers with available keys in priority order."""
    providers = []
    # Key preference: KIMI_API_KEY then MOONSHOT_API_KEY
    key = KIMI_API_KEY if _is_valid_key(KIMI_API_KEY) else (MOONSHOT_API_KEY if _is_valid_key(MOONSHOT_API_KEY) else None)
    if not key:
        return providers

    key = key.strip()

    if KIMI_API_URL and _is_valid_key(KIMI_API_URL):
        providers.append({
            "name": "Kimi Custom",
            "api_key": key,
            "api_url": KIMI_API_URL.strip(),
            "model_name": os.getenv("KIMI_MODEL_NAME") or DEFAULT_KIMI_MODEL_NAME,
        })
    else:
        providers.append({
            "name": "Kimi Platform",
            "api_key": key,
            "api_url": DEFAULT_KIMI_MODEL_API_URL,
            "model_name": DEFAULT_KIMI_MODEL_NAME,
        })
        providers.append({
            "name": "Kimi Code",
            "api_key": key,
            "api_url": DEFAULT_KIMI_CODE_API_URL,
            "model_name": DEFAULT_KIMI_CODE_MODEL_NAME,
        })

    return providers


def _get_openai_provider():
    """Build OpenAI provider if key is available."""
    if not _is_valid_key(OPENAI_API_KEY):
        return None
    return {
        "name": "OpenAI",
        "api_key": OPENAI_API_KEY.strip(),
        "api_url": DEFAULT_AI_MODEL_API_URL,
        "model_name": AI_MODEL_NAME,
    }


def _get_generic_provider():
    """Build generic/custom provider if key is available."""
    if not _is_valid_key(AI_MODEL_API_KEY):
        return None
    return {
        "name": "Generic",
        "api_key": AI_MODEL_API_KEY.strip(),
        "api_url": AI_MODEL_API_URL,
        "model_name": AI_MODEL_NAME,
    }


def get_ai_providers():
    """Get list of AI providers in priority order."""
    providers = []
    providers.extend(_get_kimi_providers())
    openai = _get_openai_provider()
    if openai:
        providers.append(openai)
    generic = _get_generic_provider()
    if generic:
        providers.append(generic)
    return providers


# ============================================================
# AI Ranking
# ============================================================

KIMI_MODELS = {DEFAULT_KIMI_MODEL_NAME, DEFAULT_KIMI_CODE_MODEL_NAME, "kimi-k3", "k3-256k"}


def _build_ai_payload(niche: str, products: list[dict]) -> dict:
    """Build the AI analysis payload."""
    product_list = []
    for idx, p in enumerate(products):
        product_list.append({
            "index": idx,
            "product": p.get("product", ""),
            "price": p.get("price"),
            "currency": p.get("currency", "USD"),
            "rating": p.get("rating"),
            "reviews": p.get("reviews"),
            "sales": p.get("sales"),
            "url": p.get("url", ""),
            "image": p.get("image", ""),
            "evidence": p.get("evidence", []),
            "badges": p.get("badges", []),
        })

    system_prompt = """You are a product research expert. Analyze each candidate product for viral e-commerce potential.

CRITERIA:
- Problem solving: Does it solve a clear pain point?
- Video demonstration potential: Is it visually engaging for TikTok/YouTube?
- Emotional appeal: Does it evoke strong emotions (joy, surprise, relief)?
- Shipping ease: Is it lightweight, non-fragile, easy to ship internationally?
- Competition: Is the niche underserved with room for new sellers?
- Bundling: Can it be bundled with complementary products?
- Supplier evidence: Is there strong supplier/manufacturer support?
- Year-round demand: Is demand consistent or seasonal?

REJECT if: unsafe, restricted, fragile, edible, medicinal, branded-risk (counterfeit/infringing), or difficult-to-ship (heavy, bulky, hazardous).

Return ONLY a valid JSON object with no other text."""

    user_prompt = (
        f"Niche: {niche}\n\n"
        f"Products:\n{json.dumps(product_list, indent=2)}\n\n"
        "For each product by index, return:\n"
        "- candidate_index: the original zero-based index\n"
        "- viral_score: integer 0-100\n"
        "- reason: short evidence-based reason (max 200 characters)\n\n"
        "Respond with a JSON object containing a 'ranked' array of objects, each with candidate_index, viral_score, and reason fields.\n"
        "Example: {\"ranked\": [{\"candidate_index\": 0, \"viral_score\": 82, \"reason\": \"...\"}]}\n"
        "Maximum 10 ranked products."
    )

    return {
        "model": None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }


def _get_provider_timeout(api_url: str) -> tuple[float, float]:
    """Get timeout based on provider URL."""
    if any(kimi_url in api_url for kimi_url in [DEFAULT_KIMI_MODEL_API_URL, DEFAULT_KIMI_CODE_API_URL]):
        return (30, 240)
    return (30, 60)


def _get_provider_payload(payload: dict, provider: dict) -> dict:
    """Customize payload for specific provider."""
    final_payload = copy.deepcopy(payload)
    final_payload["model"] = provider["model_name"]

    if provider["model_name"] in KIMI_MODELS:
        final_payload["reasoning_effort"] = "low"
        final_payload["max_completion_tokens"] = 4096
        if "moonshot" in provider["api_url"].lower():
            final_payload["response_format"] = {"type": "json_object"}
        else:
            final_payload.pop("response_format", None)
    else:
        if "temperature" not in final_payload:
            final_payload["temperature"] = 0.3

    return final_payload


def _call_ai_provider(provider: dict, payload: dict) -> tuple[dict | None, str | None]:
    """Call an AI provider and return (parsed_response, error_message)."""
    api_url = provider["api_url"]
    api_key = provider["api_key"]
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    final_payload = _get_provider_payload(payload, provider)

    if "response_format" in final_payload and "moonshot" not in api_url.lower():
        final_payload.pop("response_format")

    timeout = _get_provider_timeout(api_url)

    try:
        response = requests.post(
            api_url,
            json=final_payload,
            headers=headers,
            timeout=timeout,
        )
        response.raise_for_status()
        return response.json(), None
    except requests.exceptions.HTTPError as error:
        status_code = getattr(getattr(error, "response", None), "status_code", None)
        if status_code == 401:
            return None, f"AI provider authentication failed (401): {provider['name']}"
        elif status_code == 429:
            return None, f"AI provider quota/rate limit reached (429): {provider['name']}"
        else:
            return None, f"AI provider HTTP {status_code}: {provider['name']}"
    except Exception as error:
        return None, f"AI provider error: {provider['name']}: {str(error)[:200]}"


def analyze_with_ai_model(niche: str, products: list[dict]) -> tuple[list[dict], str | None]:
    """
    Analyze products with AI model and return ranked products with AI scores.
    Returns (ranked_products, warning_message).
    """
    if not products:
        return [], None

    providers = get_ai_providers()
    if not providers:
        sorted_products = sorted(products, key=lambda p: p.get("viral_score", 0), reverse=True)[:10]
        return sorted_products, "No AI provider configured. Using local scores."

    base_payload = _build_ai_payload(niche, products)
    last_error = None

    for provider in providers:
        response_json, error = _call_ai_provider(provider, base_payload)

        if error:
            last_error = error
            continue

        if not response_json:
            last_error = f"Empty response from {provider['name']}"
            continue

        try:
            ranked_data_raw = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(ranked_data_raw, str):
                ranked_data = json.loads(ranked_data_raw)
            else:
                ranked_data = ranked_data_raw

            ranked = ranked_data.get("ranked", [])
            if not isinstance(ranked, list):
                last_error = f"Invalid response format from {provider['name']}"
                continue

            valid_ranked = []
            seen_indexes = set()
            for entry in ranked:
                if not isinstance(entry, dict):
                    continue
                candidate_index = entry.get("candidate_index")
                if candidate_index is None or not isinstance(candidate_index, int):
                    continue
                if candidate_index < 0 or candidate_index >= len(products):
                    continue
                if candidate_index in seen_indexes:
                    continue
                seen_indexes.add(candidate_index)

                viral_score = entry.get("viral_score")
                if viral_score is None:
                    viral_score = 0
                viral_score = max(0, min(100, int(viral_score)))

                reason = entry.get("reason", "")
                if isinstance(reason, str):
                    reason = reason[:200]

                original = copy.deepcopy(products[candidate_index])
                original["ai_viral_score"] = viral_score
                original["ai_reason"] = reason
                valid_ranked.append(original)

                if len(valid_ranked) >= 10:
                    break

            valid_ranked.sort(key=lambda p: p.get("ai_viral_score", 0), reverse=True)
            return valid_ranked, None

        except (json.JSONDecodeError, ValueError, KeyError) as error:
            last_error = f"Failed to parse response from {provider['name']}: {str(error)[:100]}"
            continue

    if last_error and ("401" in last_error or "authentication failed" in last_error.lower()):
        warning = "AI provider authentication failed. Using local scores."
    elif last_error and ("429" in last_error or "quota" in last_error.lower() or "rate limit" in last_error.lower()):
        warning = "AI provider quota/rate limit reached. Using local scores."
    else:
        warning = "All AI providers failed. Using local scores."

    sorted_products = sorted(products, key=lambda p: p.get("viral_score", 0), reverse=True)[:10]
    return sorted_products, warning


# ============================================================
# Supplier Sourcing
# ============================================================

# Known currency symbols
CURRENCY_SYMBOLS = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "CNY", "RMB": "CNY", "A$": "AUD", "C$": "CAD"}


def build_supplier_payloads(product_name: str) -> list[dict]:
    """Build exactly two supplier job payloads for Alibaba and AliExpress."""
    alibaba_url = f"https://www.alibaba.com/trade/search?SearchText={quote_plus(product_name)}"
    aliexpress_url = f"https://www.aliexpress.com/wholesale?SearchText={quote_plus(product_name)}"

    return [
        {
            "url": alibaba_url,
            "proxy_pool": DECODO_PROXY_POOL,
            "headless": "html",
        },
        {
            "url": aliexpress_url,
            "proxy_pool": DECODO_PROXY_POOL,
            "headless": "html",
        },
    ]


def _clean_supplier_title(text: str) -> str:
    """Clean up raw text for supplier listing title."""
    if not text:
        return ""
    cleaned = html.unescape(text.strip())
    cleaned = re.sub(r"[\$\€\£\¥]\s*[\d]+[.,]?[\d]*.*", "", cleaned)
    cleaned = re.sub(
        r"(?:See preview|Similar items|Free shipping|Extra \d+% off|Delivery:|\d+ sold).*",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


class MarketplaceParser(HTMLParser):
    """HTML parser for extracting product listings from marketplace pages."""

    def __init__(self, platform: str):
        super().__init__()
        self.platform = platform
        self.listings = []
        self.seen_urls = set()
        self.in_anchor = False
        self.current_url = None
        self.current_text = ""
        self.candidate_links_count = 0
        self.stopped = False

    def handle_starttag(self, tag, attrs):
        if self.stopped:
            return
        attrs_dict = dict(attrs)
        if tag == "a" and "href" in attrs_dict:
            url = attrs_dict["href"].strip()
            if self._is_listing_url(url):
                self.candidate_links_count += 1
                can_url = self._canonicalize_url(url)
                if can_url and can_url not in self.seen_urls:
                    self.in_anchor = True
                    self.current_url = can_url
                    self.current_text = ""

    def handle_endtag(self, tag):
        if self.stopped:
            return
        if tag == "a" and self.in_anchor:
            self.in_anchor = False
            if self.current_url and self.current_text.strip():
                self._save_listing()
            self.current_url = None
            self.current_text = ""

    def handle_data(self, data):
        if self.stopped:
            return
        if self.in_anchor:
            self.current_text += " " + data

    def _is_listing_url(self, url: str) -> bool:
        """Check if URL is a marketplace listing."""
        if not url:
            return False
        url_lower = url.lower()
        if self.platform == "alibaba":
            return "product-detail" in url_lower or "/product/" in url_lower
        elif self.platform == "aliexpress":
            return "/item/" in url_lower
        return False

    def _canonicalize_url(self, url: str) -> str:
        """Canonicalize listing URL by removing query strings and fragments."""
        if not url:
            return ""
        if url.startswith("//"):
            url = f"https:{url}"
        elif url.startswith("/"):
            url = f"https://www.{self.platform}.com{url}"
        try:
            parsed = urlparse(url)
            scheme = parsed.scheme if parsed.scheme else "https"
            return f"{scheme}://{parsed.netloc}{parsed.path}"
        except Exception:
            return url

    def _save_listing(self):
        """Save a listing if it has both title and URL."""
        if not self.current_url or self.current_url in self.seen_urls:
            return
        title = _clean_supplier_title(self.current_text)
        if not title or len(title) < 3:
            return

        price, currency = _extract_price_from_text(self.current_text)

        listing = {
            "supplier_url": self.current_url,
            "supplier_title": title,
            "platform": self.platform,
        }
        if price is not None:
            listing["unit_price"] = price
        if currency:
            listing["currency"] = currency

        self.seen_urls.add(self.current_url)
        self.listings.append(listing)
        if len(self.listings) >= 5:
            self.stopped = True


def _extract_price_from_text(text: str) -> tuple[float | None, str | None]:
    """Extract numeric price and currency from text."""
    if not text or not isinstance(text, str):
        return None, None

    currency = None
    for symbol, code in CURRENCY_SYMBOLS.items():
        if symbol in text:
            currency = code
            break

    matches = re.findall(r"(?:[\$\€\£\¥])?\s*([\d]+[.,][\d]{2})", text)
    if matches:
        for match_str in matches:
            try:
                val = float(match_str.replace(",", ""))
                if val > 0:
                    return val, currency or "USD"
            except ValueError:
                pass

    match = re.search(r"([\d]+[.,]?[\d]*)", text)
    if match:
        try:
            price = float(match.group(1).replace(",", ""))
            if price > 0:
                return price, currency
        except ValueError:
            pass

    return None, currency


def _normalize_supplier_offer(raw: dict, platform: str, product_reference: str, 
                            source_url: str, candidate_index: int) -> dict | None:
    """Normalize a supplier offer from parsed content or HTML."""
    # Must have nonblank title
    title = raw.get("supplier_title", raw.get("title", raw.get("name", "")))
    if not title or not isinstance(title, str) or not title.strip():
        return None
    title = html.unescape(title.strip())

    # Must have URL
    url = raw.get("supplier_url", raw.get("url", raw.get("link", "")))
    if not url or not isinstance(url, str):
        return None

    # Extract price
    price = None
    price_range = None
    currency = raw.get("currency", "")

    # Try unit_price first
    unit_price = raw.get("unit_price")
    if unit_price is not None:
        price = _to_numeric(unit_price)

    # Try price
    if price is None:
        price_val = raw.get("price")
        if price_val is not None:
            price = _to_numeric(price_val)

    # Try price_lower/price_min
    if price is None:
        price_min = _to_numeric(raw.get("price_lower") or raw.get("price_min"))
        if price_min is not None:
            price = price_min

    # Try price range
    if price is None:
        min_price = _to_numeric(raw.get("min_price") or raw.get("price_range", {}).get("min"))
        max_price = _to_numeric(raw.get("max_price") or raw.get("price_range", {}).get("max"))
        if min_price is not None and max_price is not None:
            price_range = {"min": min_price, "max": max_price}
            price = None  # Clear unit_price if we have a range

    # If no price at all, try to extract from text fields
    if price is None and price_range is None:
        for field in ["price_text", "price_str", "display_price"]:
            if field in raw and isinstance(raw[field], str):
                extracted_price, extracted_currency = _extract_price_from_text(raw[field])
                if extracted_price is not None:
                    price = extracted_price
                    if extracted_currency and not currency:
                        currency = extracted_currency

    # Detect currency from symbol in title or other fields
    if not currency:
        for field in ["title", "supplier_title", "price_text"]:
            if field in raw and isinstance(raw[field], str):
                for symbol, code in CURRENCY_SYMBOLS.items():
                    if symbol in raw[field]:
                        currency = code
                        break
                if currency:
                    break

    # Default currency
    if not currency:
        currency = "USD"

    # Extract MOQ
    moq = _to_numeric(raw.get("min_order_quantity") or raw.get("moq") or raw.get("min_order"))
    if moq is None:
        moq = 1

    # Extract shipping info
    shipping = raw.get("shipping_info", raw.get("shipping", raw.get("delivery", "")))
    if isinstance(shipping, str):
        shipping = shipping.strip()[:200]
    else:
        shipping = ""

    # Extract rating, orders, reviews
    rating = _to_numeric(raw.get("rating") or raw.get("score") or raw.get("star"))
    orders = _to_numeric(raw.get("orders") or raw.get("total_orders") or raw.get("sold"))
    reviews = _to_numeric(raw.get("reviews") or raw.get("review_count") or raw.get("total_reviews"))

    # Evidence from badges, labels, etc.
    evidence = []
    for field in ["badges", "labels", "tags", "evidence"]:
        val = raw.get(field)
        if val:
            if isinstance(val, str):
                if val.strip():
                    evidence.append(val)
            elif isinstance(val, list):
                for item in val:
                    if isinstance(item, str) and item.strip():
                        evidence.append(item)

    offer = {
        "product_reference": product_reference,
        "supplier_platform": platform,
        "supplier_title": title,
        "currency": currency,
        "min_order_quantity": int(moq) if moq else 1,
        "shipping_info": shipping,
        "supplier_url": url,
        "rating": rating,
        "orders": orders,
        "reviews": reviews,
        "supplier_evidence": evidence,
        "provenance": {
            "source": "marketplace",
            "platform": platform,
            "search_url": source_url,
            "candidate_index": candidate_index,
        },
    }

    if price is not None:
        offer["unit_price"] = price
    if price_range is not None:
        offer["price_range"] = price_range

    return offer


def _parse_marketplace_content(content: Any, platform: str, product_reference: str,
                              source_url: str, candidate_index: int, max_offers: int = 5) -> list[dict]:
    """Parse marketplace content and extract normalized offers."""
    offers = []

    if isinstance(content, str):
        # HTML content - parse with HTMLParser
        parser = MarketplaceParser(platform)
        parser.feed(content)
        for raw_listing in parser.listings:
            offer = _normalize_supplier_offer(
                raw_listing, platform, product_reference, source_url, candidate_index
            )
            if offer:
                offers.append(offer)
            if len(offers) >= max_offers:
                break
    elif isinstance(content, dict):
        # Parsed JSON - walk and find product-like records
        records = []
        _find_product_records(content, records, max_records=max_offers * 2)
        for raw_record in records[:max_offers]:
            # Convert to a format _normalize_supplier_offer can handle
            raw_offer = {
                "supplier_title": raw_record.get("title") or raw_record.get("product") or raw_record.get("name"),
                "supplier_url": raw_record.get("url") or raw_record.get("link") or raw_record.get("href"),
                "unit_price": raw_record.get("price") or raw_record.get("unit_price"),
                "currency": raw_record.get("currency"),
                "min_order_quantity": raw_record.get("min_order_quantity") or raw_record.get("moq"),
                "shipping_info": raw_record.get("shipping") or raw_record.get("shipping_info"),
                "rating": raw_record.get("rating"),
                "orders": raw_record.get("orders") or raw_record.get("total_orders"),
                "reviews": raw_record.get("reviews") or raw_record.get("review_count"),
                "badges": raw_record.get("badges"),
            }
            offer = _normalize_supplier_offer(
                raw_offer, platform, product_reference, source_url, candidate_index
            )
            if offer:
                offers.append(offer)
            if len(offers) >= max_offers:
                break

    return offers


def _find_product_records(data: Any, records: list, max_records: int = 10, depth: int = 0) -> None:
    """Recursively find product-like records in parsed JSON."""
    if len(records) >= max_records:
        return
    if depth > 10:  # Prevent deep recursion
        return

    if isinstance(data, dict):
        # Check if this looks like a product record
        has_title = any(k in data for k in ["title", "product", "name", "product_title", "product_name"])
        has_price = any(k in data for k in ["price", "unit_price", "current_price", "sale_price"])
        if has_title and has_price:
            records.append(copy.deepcopy(data))
            if len(records) >= max_records:
                return
        # Recurse into values
        for value in data.values():
            _find_product_records(value, records, max_records, depth + 1)
    elif isinstance(data, list):
        for item in data:
            _find_product_records(item, records, max_records, depth + 1)


def _scrape_supplier_payload(source_name: str, payload: dict) -> dict:
    """Scrape a single supplier marketplace URL through Decodo."""
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Basic {DECODO_AUTH_TOKEN}",
    }
    try:
        response = requests.post(
            DECODO_API_URL,
            json=payload,
            headers=headers,
            timeout=(10, DECODO_REQUEST_TIMEOUT),
        )
        response.raise_for_status()
        parsed = response.json()
        data = parse_decodo_content(parsed)
        # Get the actual content
        if isinstance(data, dict):
            data = get_parsed_results(data)
        compacted_data = compact_parsed_content(source_name, data)
        trimmed_data = trim_decodo_result({"data": compacted_data}, 12000)
        return {"source": source_name, "request": payload, "data": trimmed_data.get("data"), "error": None}
    except Exception as error:
        message = str(error)
        if hasattr(error, 'response') and error.response is not None:
            resp = error.response
            message = f"HTTP {resp.status_code}"
            try:
                detail = resp.text[:500] if len(resp.text) > 500 else resp.text
                if detail:
                    message += f": {detail}"
            except Exception:
                pass
        return {"source": source_name, "request": payload, "data": None, "error": message}


def validate_suppliers_for_products(products: list[dict]) -> tuple[list[dict], dict]:
    """
    Validate suppliers for top 10 products.
    Returns (supplier_data, supplier_summary).
    """
    # Take at most 10 products with nonblank product names
    product_list = products[:10]
    product_list = [p for p in product_list if p.get("product") and isinstance(p.get("product"), str) and p["product"].strip()]

    # Build all jobs: 2 per product (Alibaba + AliExpress)
    all_jobs = []
    for idx, product in enumerate(product_list):
        product_name = product["product"].strip()
        payloads = build_supplier_payloads(product_name)
        for j, payload in enumerate(payloads):
            platform = "alibaba" if j == 0 else "aliexpress"
            all_jobs.append({
                "product_reference": product_name,
                "candidate_index": idx,
                "platform": platform,
                "payload": payload,
                "source_name": f"{platform}__{idx}",
            })

    # Execute concurrently
    worker_count = min(max(1, DECODO_SUPPLIER_MAX_WORKERS), len(all_jobs))
    results = []

    def process_job(job):
        return _scrape_supplier_payload(job["source_name"], job["payload"])

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        scraped_results = list(executor.map(process_job, all_jobs))

    # Group results by product
    product_suppliers = {}
    for i, job in enumerate(all_jobs):
        result = scraped_results[i]
        product_ref = job["product_reference"]
        platform = job["platform"]
        candidate_idx = job["candidate_index"]
        search_url = job["payload"].get("url", "")

        if product_ref not in product_suppliers:
            product_suppliers[product_ref] = {
                "product_name": product_ref,
                "candidate_index": candidate_idx,
                "supplier_results": [],
                "alibaba_offers": [],
                "aliexpress_offers": [],
            }

        # Parse the result
        data = result.get("data")
        error = result.get("error")

        if error:
            # Failed request - add error entry
            product_suppliers[product_ref]["supplier_results"].append({
                "platform": platform,
                "error": error,
                "offers": [],
                "has_data": False,
            })
        else:
            # Parse the content
            offers = _parse_marketplace_content(
                data, platform, product_ref, search_url, candidate_idx
            )
            for off_idx, offer in enumerate(offers):
                if isinstance(offer, dict) and "provenance" in offer:
                    offer["provenance"]["offer_index"] = off_idx
            product_suppliers[product_ref]["supplier_results"].append({
                "platform": platform,
                "error": None,
                "offers": offers,
                "has_data": True,
            })
            # Also add to platform-specific list
            if platform == "alibaba":
                product_suppliers[product_ref]["alibaba_offers"].extend(offers)
            else:
                product_suppliers[product_ref]["aliexpress_offers"].extend(offers)

    # Build supplier data and summary
    supplier_data = list(product_suppliers.values())

    total_requests = len(all_jobs)
    successful_requests = sum(1 for r in scraped_results if r.get("error") is None)
    failed_requests = sum(1 for r in scraped_results if r.get("error") is not None)
    empty_requests = sum(1 for r in scraped_results if r.get("error") is None and (r.get("data") is None or not r.get("data")))

    alibaba_count = sum(len(p["alibaba_offers"]) for p in supplier_data)
    aliexpress_count = sum(len(p["aliexpress_offers"]) for p in supplier_data)

    supplier_summary = {
        "products_validated": len(product_list),
        "total_supplier_requests": total_requests,
        "successful_supplier_requests": successful_requests,
        "failed_supplier_requests": failed_requests,
        "empty_supplier_requests": empty_requests,
        "alibaba_offer_count": alibaba_count,
        "aliexpress_offer_count": aliexpress_count,
    }

    return supplier_data, supplier_summary


# ============================================================
# Discovery Workflow
# ============================================================

def run_discovery(niche: str) -> dict:
    """
    Run discovery workflow: scrape, compact, summarize, extract, rank.
    Returns dict with niche, initial_products, discovery_summary, discovery_data, and ranking_warnings.
    """
    discovery_data_raw = scrape_with_decodo(niche)
    discovery_data = json.loads(discovery_data_raw)

    discovery_summary = [summarize_decodo_result(r) for r in discovery_data]
    initial_products = extract_products(discovery_data)

    # Rank with AI
    ranked_products, ai_warning = analyze_with_ai_model(niche, initial_products)

    # Build ranking warnings list
    ranking_warnings = []
    if ai_warning:
        ranking_warnings.append(ai_warning)

    return {
        "niche": niche,
        "initial_products": ranked_products,
        "discovery_summary": discovery_summary,
        "discovery_data": discovery_data,
        "ranking_warnings": ranking_warnings,
    }


# ============================================================
# Decodo Content Handling
# ============================================================

TITLE_ALIASES = ["product_title", "product_name", "title", "name"]
PRICE_ALIASES = ["price", "current_price", "sale_price", "original_price", "price_lower", "price_min"]
CURRENCY_ALIASES = ["currency", "price_currency", "currency_code"]
URL_ALIASES = ["url", "link", "product_url", "href", "item_url"]
IMAGE_ALIASES = ["image", "thumbnail", "image_url", "image_urls", "photos"]
RATING_ALIASES = ["rating", "product_rating", "star_rating", "stars"]
REVIEWS_ALIASES = ["reviews", "review_count", "num_reviews", "total_reviews"]
SALES_ALIASES = ["sales", "orders", "sales_volume", "total_sales", "order_count"]
BADGES_ALIASES = ["badges", "badge", "labels", "tags"]
ASIN_ALIASES = ["asin", "product_asin", "item_asin"]


def stringify_decodo_content(content: Any) -> Any:
    """Parse strings as JSON when possible; leave HTML/Markdown/text as strings."""
    if isinstance(content, str):
        try:
            return json.loads(content)
        except (json.JSONDecodeError, ValueError):
            return content
    elif isinstance(content, dict):
        return {k: stringify_decodo_content(v) for k, v in content.items()}
    elif isinstance(content, list):
        return [stringify_decodo_content(item) for item in content]
    return content


def parse_decodo_content(content: Any) -> Any:
    """Stringify and parse Decodo content recursively."""
    return stringify_decodo_content(content)


def get_parsed_results(content: dict) -> dict:
    """
    Parse content from Decodo response.
    First read content.get("results", content).
    If that value is not a dict, return it directly.
    If it is a dict and its nested results value is also a dict, return that nested dict.
    Otherwise return the first dict.
    """
    results = content.get("results", content)
    if not isinstance(results, dict):
        return results
    if "results" in results and isinstance(results["results"], dict):
        return results["results"]
    return results


def compact_parsed_content(source_name: str, content: Any) -> Any:
    """Compact parsed content based on source type."""
    if not isinstance(content, dict):
        return content

    compacted = copy.deepcopy(content)

    if "amazon" in source_name.lower():
        if "organic" in compacted and isinstance(compacted["organic"], list):
            compacted["organic"] = compacted["organic"][:25]
            for product in compacted["organic"]:
                if isinstance(product, dict):
                    _compact_amazon_product(product)
        if "sponsored" in compacted and isinstance(compacted["sponsored"], list):
            compacted["sponsored"] = compacted["sponsored"][:25]
            for product in compacted["sponsored"]:
                if isinstance(product, dict):
                    _compact_amazon_product(product)

    elif "google" in source_name.lower():
        if "organic" in compacted and isinstance(compacted["organic"], list):
            compacted["organic"] = compacted["organic"][:10]
        if "ai_overviews" in compacted and isinstance(compacted["ai_overviews"], list):
            compacted["ai_overviews"] = compacted["ai_overviews"][:3]
        if "related_questions" in compacted and isinstance(compacted["related_questions"], list):
            compacted["related_questions"] = compacted["related_questions"][:10]
        for key in ["shopping", "products", "shopping_results", "product_results"]:
            if key in compacted and isinstance(compacted[key], list):
                compacted[key] = compacted[key][:25]

    return compacted


def _compact_amazon_product(product: dict) -> None:
    """Keep only useful fields for Amazon products and normalize URLs."""
    useful_fields = {
        "position", "title", "price", "currency", "rating", "reviews",
        "sales", "badges", "asin", "url", "image", "product_url",
        "link", "image_url", "thumbnail"
    }
    keys_to_remove = []
    for key in product:
        if key not in useful_fields:
            keys_to_remove.append(key)
    for key in keys_to_remove:
        del product[key]

    for key in ("url", "product_url", "link"):
        if key in product and isinstance(product[key], str):
            product[key] = _normalize_amazon_url(product[key])


def trim_decodo_result(result: dict, max_content_chars: int = 3500) -> dict:
    """Deep-copy and trim result content to max characters."""
    trimmed = copy.deepcopy(result)
    _trim_content(trimmed, max_content_chars)
    return trimmed


def _trim_content(data: Any, max_chars: int) -> int:
    """Recursively trim string content, return chars used."""
    if isinstance(data, dict):
        for key, value in list(data.items()):
            if isinstance(value, str):
                if len(value) > max_chars:
                    data[key] = value[:max_chars] + "..."
            elif isinstance(value, (dict, list)):
                _trim_content(value, max_chars)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            if isinstance(item, str):
                if len(item) > max_chars:
                    data[i] = item[:max_chars] + "..."
            elif isinstance(item, (dict, list)):
                _trim_content(item, max_chars)


def _resolve_value(data: Any, aliases: list[str]) -> Any:
    """Try to resolve a value from data using list of alias keys."""
    if not isinstance(data, dict):
        return None
    for alias in aliases:
        if alias in data:
            value = data[alias]
            if value is not None:
                return value
    return None


def _to_numeric(value: Any) -> float | None:
    """Convert value to float if possible."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            cleaned = re.sub(r"[^\d.]", "", str(value))
            if cleaned:
                return float(cleaned)
        except (ValueError, TypeError):
            pass
    return None


def _get_nested_image(value: Any) -> str | None:
    """Extract a string URL from image value which may be list or dict."""
    if value is None:
        return None
    if isinstance(value, str):
        return value if value.strip() else None
    if isinstance(value, list):
        for item in value:
            if isinstance(item, str) and item.strip():
                return item
            if isinstance(item, dict):
                url = _resolve_value(item, ["url", "link", "src", "href"])
                if isinstance(url, str) and url.strip():
                    return url
    if isinstance(value, dict):
        url = _resolve_value(value, ["url", "link", "src", "href"])
        if isinstance(url, str) and url.strip():
            return url
    return None


def _normalize_amazon_url(url: str) -> str | None:
    """Expand relative Amazon URLs to absolute."""
    if not url or not isinstance(url, str):
        return None
    url = url.strip()
    if not url:
        return None
    if url.startswith("http"):
        return url
    if url.startswith("/"):
        return f"https://www.amazon.com{url}"
    if "." in url and not url.startswith("http"):
        return f"https://{url}"
    return url


def _is_structured_product(candidate: dict) -> bool:
    """Check if candidate has nonblank title plus at least one product signal."""
    title = None
    for alias in TITLE_ALIASES:
        if alias in candidate and candidate[alias] and isinstance(candidate[alias], str):
            title = candidate[alias].strip()
            if title:
                break

    if not title:
        return False

    has_price = _resolve_value(candidate, PRICE_ALIASES) is not None
    has_rating = _resolve_value(candidate, RATING_ALIASES) is not None
    has_reviews = _resolve_value(candidate, REVIEWS_ALIASES) is not None
    has_sales = _resolve_value(candidate, SALES_ALIASES) is not None
    has_image = _resolve_value(candidate, IMAGE_ALIASES) is not None

    return has_price or has_rating or has_reviews or has_sales or has_image


def _normalize_product_title(title: str) -> str:
    """Normalize title for deduplication: lowercase, collapse non-alphanumeric runs to spaces."""
    if not title or not isinstance(title, str):
        return ""
    normalized = re.sub(r"[^a-zA-Z0-9\s]", " ", title.lower())
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def _extract_evidence(candidate: dict) -> list[str]:
    """Extract evidence strings from candidate."""
    evidence = []
    badges = _resolve_value(candidate, BADGES_ALIASES)
    if badges:
        if isinstance(badges, str):
            if badges.strip():
                evidence.append(badges)
        elif isinstance(badges, list):
            for b in badges:
                if isinstance(b, str) and b.strip():
                    evidence.append(b)
                elif isinstance(b, dict):
                    for v in b.values():
                        if isinstance(v, str) and v.strip():
                            evidence.append(v)
    return evidence


def _has_best_seller_evidence(badges: Any) -> bool:
    """Check if badges contain best-seller or Amazon's Choice."""
    if badges is None:
        return False
    if isinstance(badges, str):
        badges_lower = badges.lower()
        return "best seller" in badges_lower or "amazon's choice" in badges_lower
    if isinstance(badges, list):
        for b in badges:
            if isinstance(b, str):
                if "best seller" in b.lower() or "amazon's choice" in b.lower():
                    return True
            elif isinstance(b, dict):
                for v in b.values():
                    if isinstance(v, str) and ("best seller" in v.lower() or "amazon's choice" in v.lower()):
                        return True
    return False


def _calculate_viral_score(product: dict) -> int:
    """Calculate viral score for a product."""
    score = 35

    price = product.get("price")
    if price is not None and 5 <= price <= 50:
        score += 12

    rating = product.get("rating")
    if rating is not None:
        try:
            rating_score = round(float(rating) * 3)
            score += rating_score
        except (ValueError, TypeError):
            pass

    reviews = product.get("reviews")
    sales = product.get("sales")
    volume = reviews or sales
    if volume is not None:
        try:
            volume_num = _to_numeric(volume)
            if volume_num and volume_num > 0:
                volume_score = round(math.log10(volume_num) * 5)
                score += volume_score
        except (ValueError, TypeError):
            pass

    if product.get("image"):
        score += 5

    if product.get("url"):
        score += 5

    badges = product.get("badges", [])
    if _has_best_seller_evidence(badges):
        score += 10

    return min(score, 100)


def _walk_and_collect(data: Any, source: str, collected: list) -> None:
    """Recursively walk data to collect product candidates."""
    if isinstance(data, dict):
        if _is_structured_product(data):
            candidate = copy.deepcopy(data)
            candidate["_source"] = source
            collected.append(candidate)
        for value in data.values():
            _walk_and_collect(value, source, collected)
    elif isinstance(data, list):
        for item in data:
            _walk_and_collect(item, source, collected)


def extract_products(discovery_results: list[dict], limit: int = 25) -> list[dict]:
    """
    Extract products from discovery results.
    Returns normalized, deduplicated products sorted by viral_score descending.
    """
    candidates = []

    for result in discovery_results:
        source_name = result.get("source", "unknown")
        content = result.get("data")
        if not content:
            continue

        parsed = parse_decodo_content(content)
        if isinstance(parsed, dict):
            parsed = get_parsed_results(parsed)

        compacted = compact_parsed_content(source_name, parsed)
        _walk_and_collect(compacted, source_name, candidates)

    products = []
    seen = {}

    for candidate in candidates:
        source = candidate.pop("_source", "unknown")
        evidence = _extract_evidence(candidate)

        title = None
        for alias in TITLE_ALIASES:
            if alias in candidate and candidate[alias] and isinstance(candidate[alias], str):
                title = candidate[alias].strip()
                if title:
                    break

        if not title:
            continue

        norm_title = _normalize_product_title(title)

        price_val = _to_numeric(_resolve_value(candidate, PRICE_ALIASES))
        currency = _resolve_value(candidate, CURRENCY_ALIASES)
        if currency and isinstance(currency, str):
            currency = currency.strip().upper()
        url = _resolve_value(candidate, URL_ALIASES)
        if url and isinstance(url, str):
            url = _normalize_amazon_url(url)
        image = _get_nested_image(_resolve_value(candidate, IMAGE_ALIASES))

        rating_val = _to_numeric(_resolve_value(candidate, RATING_ALIASES))
        reviews_val = _resolve_value(candidate, REVIEWS_ALIASES)
        if reviews_val is not None:
            reviews_val = _to_numeric(reviews_val)
        sales_val = _resolve_value(candidate, SALES_ALIASES)
        if sales_val is not None:
            sales_val = _to_numeric(sales_val)
        badges = _resolve_value(candidate, BADGES_ALIASES)
        asin = _resolve_value(candidate, ASIN_ALIASES)
        if asin and isinstance(asin, str):
            asin = asin.strip().upper()

        product = {
            "product": title,
            "source": source,
            "price": price_val,
            "currency": currency,
            "url": url,
            "image": image,
            "evidence": evidence,
            "rating": rating_val,
            "reviews": reviews_val,
            "sales": sales_val,
            "badges": badges,
            "asin": asin,
            "viral_score": 0,
        }

        has_price = price_val is not None
        has_rating = rating_val is not None
        has_reviews = reviews_val is not None
        has_sales = sales_val is not None
        has_image = image is not None

        if not (has_price or has_rating or has_reviews or has_sales or has_image):
            continue

        if norm_title in seen:
            existing_idx = seen[norm_title]
            existing = products[existing_idx]
            if source not in existing["source"]:
                existing["source"] = f"{existing['source']}, {source}"
            for e in evidence:
                if e not in existing["evidence"]:
                    existing["evidence"].append(e)
            if existing["price"] is None and price_val is not None:
                existing["price"] = price_val
            if not existing.get("currency") and currency:
                existing["currency"] = currency
            if not existing.get("url") and url:
                existing["url"] = url
            if not existing.get("image") and image:
                existing["image"] = image
        else:
            seen[norm_title] = len(products)
            products.append(product)

    for product in products:
        product["viral_score"] = _calculate_viral_score(product)

    products.sort(key=lambda p: p["viral_score"], reverse=True)
    return products[:limit]


def summarize_decodo_result(result: dict) -> dict:
    """
    Return one summary row per source.
    """
    source_name = result.get("source", "unknown")
    error = result.get("error")
    data = result.get("data", {})

    product_count = _count_products_in_content(data)
    response_count = _count_responses_in_content(data)

    http_status = None
    if error and isinstance(error, str):
        match = re.search(r"HTTP (\d{3})", error)
        if match:
            http_status = int(match.group(1))

    parser_status = "success" if not error else "failed"

    content_chars = 0
    if isinstance(data, dict):
        content_chars = len(json.dumps(data))
    elif isinstance(data, str):
        content_chars = len(data)

    price_count = _count_evidence_in_content(data, PRICE_ALIASES)
    url_count = _count_evidence_in_content(data, URL_ALIASES)
    image_count = _count_evidence_in_content(data, IMAGE_ALIASES)

    captcha_detected = _detect_captcha(data) or (error and _detect_captcha(error))

    preview = _get_preview(data, product_count)

    return {
        "source": source_name,
        "error": error if error else "",
        "extracted_product_count": product_count,
        "response_count": response_count,
        "http_status": http_status,
        "parser_status": parser_status,
        "content_chars": content_chars,
        "price_count": price_count,
        "url_count": url_count,
        "image_count": image_count,
        "captcha_detected": captcha_detected,
        "preview": preview[:200] if preview else "",
    }


def _count_products_in_content(content: Any) -> int:
    if not isinstance(content, (dict, list)):
        return 0
    count = 0
    if isinstance(content, dict):
        if _is_structured_product(content):
            count += 1
        for value in content.values():
            count += _count_products_in_content(value)
    elif isinstance(content, list):
        for item in content:
            count += _count_products_in_content(item)
    return count


def _count_responses_in_content(content: Any) -> int:
    if isinstance(content, list):
        return len(content)
    if isinstance(content, dict):
        if "results" in content:
            results = content["results"]
            if isinstance(results, list):
                return len(results)
            if isinstance(results, dict):
                return len(results)
        return 1
    return 0


def _count_evidence_in_content(content: Any, aliases: list[str]) -> int:
    count = 0
    if isinstance(content, dict):
        for key in aliases:
            if key in content and content[key] is not None:
                count += 1
        for value in content.values():
            count += _count_evidence_in_content(value, aliases)
    elif isinstance(content, list):
        for item in content:
            count += _count_evidence_in_content(item, aliases)
    return count


def _detect_captcha(content: Any) -> bool:
    keywords = ["captcha", "verify you are human", "unusual traffic", "access denied"]
    text = ""
    if isinstance(content, str):
        text = content.lower()
    elif isinstance(content, dict):
        text = json.dumps(content).lower()
    elif isinstance(content, list):
        text = json.dumps(content).lower()
    for keyword in keywords:
        if keyword in text:
            return True
    return False


def _get_preview(content: Any, product_count: int) -> str:
    if product_count == 0:
        if isinstance(content, str):
            return content[:200]
        elif isinstance(content, dict):
            return json.dumps(content)[:200]
        return ""
    titles = []
    _extract_titles_from_content(content, titles, max_titles=5)
    if titles:
        return ", ".join(titles[:5])
    return ""


def _extract_titles_from_content(content: Any, titles: list, max_titles: int = 5) -> None:
    if len(titles) >= max_titles:
        return
    if isinstance(content, dict):
        for alias in TITLE_ALIASES:
            if alias in content and content[alias] and isinstance(content[alias], str):
                title = content[alias].strip()
                if title and title not in titles:
                    titles.append(title)
                if len(titles) >= max_titles:
                    return
        for value in content.values():
            _extract_titles_from_content(value, titles, max_titles)
    elif isinstance(content, list):
        for item in content:
            _extract_titles_from_content(item, titles, max_titles)


def build_source_urls(niche: str) -> list[dict]:
    google_queries = [
        f"viral {niche} products",
        f"problem solving {niche} products",
        f"TikTok {niche} gadgets",
        f"Amazon best selling {niche} under $50",
        f"lightweight {niche} products easy to ship",
        f"{niche} accessories under $50",
    ]

    sources = []
    for idx, query in enumerate(google_queries, start=1):
        sources.append({
            "source": f"Google Search {idx}",
            "payload": {
                "target": "google_search",
                "query": query,
                "headless": "html",
                "parse": True,
                "page_count": 1,
                "google_results_language": "en",
            },
        })

    first_google_query = google_queries[0]

    sources.append({
        "source": "Amazon Search",
        "payload": {
            "target": "amazon_search",
            "query": f"{niche} under $50",
            "page_from": "1",
            "parse": True,
        },
    })

    sources.append({
        "source": "Reddit Search",
        "payload": {
            "target": "universal",
            "url": f"https://www.reddit.com/search.json?q={quote_plus(first_google_query)}&sort=relevance&t=year&limit=25",
        },
    })

    sources.append({
        "source": "TikTok Shop Search",
        "payload": {
            "target": "tiktok_shop_search",
            "query": niche,
            "parse": True,
        },
    })

    sources.append({
        "source": "YouTube Search",
        "payload": {
            "target": "youtube_search",
            "query": f"viral {niche} products TikTok",
        },
    })

    return sources


def scrape_with_decodo_payload(source_name: str, payload: dict) -> dict:
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "authorization": f"Basic {DECODO_AUTH_TOKEN}",
    }
    try:
        response = requests.post(
            DECODO_API_URL,
            json=payload,
            headers=headers,
            timeout=(10, DECODO_REQUEST_TIMEOUT),
        )
        response.raise_for_status()
        parsed = response.json()
        data = parse_decodo_content(parsed)
        compacted_data = compact_parsed_content(source_name, data)
        trimmed_data = trim_decodo_result({"data": compacted_data}, 12000)
        return {"source": source_name, "request": payload, "data": trimmed_data.get("data"), "error": None}
    except Exception as error:
        message = str(error)
        if hasattr(error, 'response') and error.response is not None:
            resp = error.response
            message = f"HTTP {resp.status_code}"
            try:
                detail = resp.text[:500] if len(resp.text) > 500 else resp.text
                if detail:
                    message += f": {detail}"
            except Exception:
                pass
        return {"source": source_name, "request": payload, "data": None, "error": message}


def scrape_with_decodo(niche: str, logger=None) -> str:
    sources = build_source_urls(niche)
    worker_count = min(max(1, DECODO_MAX_WORKERS), len(sources))

    def process_source(source_dict):
        return scrape_with_decodo_payload(source_dict["source"], source_dict["payload"])

    if logger:
        logger.info("Starting discovery scraping")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(process_source, sources))

    if logger:
        for result in results:
            if "error" in result:
                logger.info(f"Discovery failed for {result['source']}: {result.get('error', 'unknown')}")
            else:
                logger.info(f"Discovery succeeded for {result['source']}")

    return json.dumps(results)


# ============================================================
# Final Ranking
# ============================================================

# Fixed currency conversion rates for tutorial
CURRENCY_RATES = {
    "USD": 1.0,
    "EUR": 1.09,
    "GBP": 1.28,
    "CNY": 0.14,
    "RMB": 0.14,
    "AUD": 0.66,
    "CAD": 0.73,
}


def convert_to_usd(amount: float | None, currency: str) -> float | None:
    """Convert amount to USD using fixed tutorial rates."""
    if amount is None or not isinstance(amount, (int, float)):
        return None
    if not math.isfinite(amount) or amount <= 0:
        return None
    rate = CURRENCY_RATES.get(currency.upper() if currency else "")
    if rate is None:
        return None
    return amount * rate


def get_credible_price(offer: dict) -> tuple[float | None, str | None]:
    """Extract credible price in USD from offer.
    Uses unit_price, or min of price_range. Returns (price_usd, currency_or_None).
    """
    if not isinstance(offer, dict):
        return None, None

    currency = offer.get("currency", "USD")
    if not currency or not isinstance(currency, str):
        currency = "USD"

    # Try unit_price first
    unit_price = offer.get("unit_price")
    if unit_price is not None:
        price_val = _to_numeric(unit_price)
        if price_val is not None and price_val > 0 and math.isfinite(price_val):
            price_usd = convert_to_usd(price_val, currency)
            return price_usd, currency if price_usd is not None else None

    # Try price_range min
    price_range = offer.get("price_range")
    if isinstance(price_range, dict):
        min_price = _to_numeric(price_range.get("min"))
        if min_price is not None and min_price > 0 and math.isfinite(min_price):
            price_usd = convert_to_usd(min_price, currency)
            return price_usd, currency if price_usd is not None else None

    return None, None


def build_final_ranking_input(products: list[dict], supplier_data: list[dict]) -> tuple[list[dict], list[dict]]:
    """
    Build normalized candidates and offers for final AI ranking.
    Returns (candidates, offers) with stable zero-based indexes.
    Each candidate has index, product fields.
    Each offer has candidate_index, offer_index, platform, price_usd, and normalized fields.
    """
    candidates = []
    all_offers = []

    for cand_idx, product in enumerate(products):
        candidate = {
            "index": cand_idx,
            "product": product.get("product", ""),
            "price": product.get("price"),
            "currency": product.get("currency", "USD"),
            "viral_score": product.get("viral_score", product.get("ai_viral_score", 0)),
            "rating": product.get("rating"),
            "reviews": product.get("reviews"),
            "sales": product.get("sales"),
            "image": product.get("image", ""),
            "url": product.get("url", ""),
            "source": product.get("source", ""),
            "evidence": product.get("evidence", []),
        }
        candidates.append(candidate)

        # Find offers for this candidate
        for supplier in supplier_data:
            if supplier.get("candidate_index") == cand_idx or supplier.get("product_name") == product.get("product"):
                for offer_idx, offer_result in enumerate(supplier.get("supplier_results", [])):
                    platform = offer_result.get("platform", "unknown")
                    for off_idx, offer in enumerate(offer_result.get("offers", [])):
                        price_usd, currency = get_credible_price(offer)
                        normalized_offer = {
                            "candidate_index": cand_idx,
                            "offer_index": off_idx,
                            "platform": platform,
                            "supplier_title": offer.get("supplier_title", ""),
                            "supplier_url": offer.get("supplier_url", ""),
                            "unit_price": offer.get("unit_price"),
                            "currency": offer.get("currency", "USD"),
                            "price_usd": price_usd,
                            "min_order_quantity": offer.get("min_order_quantity", 1),
                            "shipping_info": offer.get("shipping_info", ""),
                            "rating": offer.get("rating"),
                            "orders": offer.get("orders"),
                            "reviews": offer.get("reviews"),
                            "supplier_evidence": offer.get("supplier_evidence", []),
                        }
                        all_offers.append(normalized_offer)

    return candidates, all_offers


def _score_offer_for_fallback(offer: dict) -> float:
    """Score an offer for deterministic fallback selection (0-100).
    Weights: cost desirability 35%, evidence 30%, MOQ 20%, shipping 15%.
    """
    score = 0.0
    price_usd = offer.get("price_usd")

    # Cost desirability (35%): lower is better, score 0-35
    # Assume price <= 50 is ideal for dropshipping
    if price_usd is not None and price_usd > 0:
        if price_usd <= 5:
            score += 35
        elif price_usd <= 15:
            score += 30
        elif price_usd <= 30:
            score += 20
        elif price_usd <= 50:
            score += 10
        else:
            score += 5  # Higher prices get minimal score
    else:
        score += 0  # No credible price

    # Supplier evidence (30%): more evidence = better
    evidence = offer.get("supplier_evidence", [])
    if evidence:
        score += min(30, len(evidence) * 5)

    # MOQ (20%): lower is better
    moq = offer.get("min_order_quantity", 1)
    if moq is not None:
        if moq == 1:
            score += 20
        elif moq <= 5:
            score += 15
        elif moq <= 20:
            score += 10
        elif moq <= 100:
            score += 5
        else:
            score += 2
    else:
        score += 0

    # Shipping (15%): check for positive signals
    shipping = offer.get("shipping_info", "")
    if shipping and isinstance(shipping, str):
        shipping_lower = shipping.lower()
        if "free" in shipping_lower:
            score += 15
        elif "fast" in shipping_lower or "express" in shipping_lower:
            score += 10
        elif shipping.strip():
            score += 5
    else:
        score += 0

    return min(score, 100)


def _compute_final_score_fallback(product: dict, selected_offer: dict | None, cand_offers: list[dict] | None = None) -> float:
    """Compute final score using deterministic fallback.
    Weights: initial viral 30%, supplier cost 18%, evidence 15%, MOQ 10%, shipping 8%, margin 12%, availability 7%.
    """
    viral_score = product.get("viral_score", product.get("ai_viral_score", 0))
    score = viral_score * 0.30

    if selected_offer:
        price_usd = selected_offer.get("price_usd")
        # Supplier cost component (18%)
        if price_usd is not None and price_usd > 0:
            score += min(18, (50.0 - price_usd) / 50.0 * 18) if price_usd <= 50 else 0
        else:
            score += 0

        # Supplier evidence (15%)
        evidence = selected_offer.get("supplier_evidence", [])
        score += min(15, len(evidence) * 3)

        # MOQ (10%)
        moq = selected_offer.get("min_order_quantity", 1)
        if moq is not None:
            if moq == 1:
                score += 10
            elif moq <= 5:
                score += 7
            elif moq <= 20:
                score += 4
            else:
                score += 2
        else:
            score += 0

        # Shipping (8%)
        shipping = selected_offer.get("shipping_info", "")
        if shipping and isinstance(shipping, str):
            shipping_lower = shipping.lower()
            if "free" in shipping_lower:
                score += 8
            elif "fast" in shipping_lower or "express" in shipping_lower:
                score += 5
            elif shipping.strip():
                score += 3
        else:
            score += 0

        # Margin (12%): estimate based on typical retail margin (40 if unknown)
        retail_price = product.get("price")
        retail_currency = product.get("currency", "USD")
        retail_usd = convert_to_usd(retail_price, retail_currency) if retail_price is not None else None
        if retail_usd is not None and retail_usd > 0 and price_usd is not None and price_usd > 0:
            if retail_usd > price_usd:
                margin_pct = (retail_usd - price_usd) / retail_usd
                score += min(12.0, margin_pct * 12.0)
            else:
                score += 0.0
        else:
            score += (40.0 / 100.0) * 12.0  # 4.8 points if unknown

        # Availability (7%): 100 for two platforms, 50 for one, 0 for none
        platforms = set()
        if cand_offers:
            for off in cand_offers:
                if off.get("price_usd") is not None and off.get("platform"):
                    platforms.add(str(off["platform"]).lower())
        else:
            if selected_offer.get("platform"):
                platforms.add(str(selected_offer["platform"]).lower())

        if len(platforms) >= 2:
            score += 7.0
        elif len(platforms) == 1:
            score += 3.5
        else:
            score += 0.0
    else:
        # No selected offer - zero for supplier components
        score += 0

    return min(score, 100)


def final_rank_products(niche: str, products: list[dict], 
                         supplier_data: list[dict]) -> tuple[list[dict], str | None]:
    """
    Final AI ranking with supplier awareness.
    Returns (final_products, warning_message).
    """
    if not products:
        return [], None

    # Build normalized input
    candidates, all_offers = build_final_ranking_input(products, supplier_data)

    # Group offers by candidate
    offers_by_candidate = {}
    for offer in all_offers:
        ci = offer.get("candidate_index")
        if ci not in offers_by_candidate:
            offers_by_candidate[ci] = []
        offers_by_candidate[ci].append(offer)

    # Build prompt
    providers = get_ai_providers()
    if not providers:
        # Use deterministic fallback
        return _final_rank_fallback(products, supplier_data, offers_by_candidate, candidates)

    # Prepare candidate data for AI
    candidate_data = []
    for ci, candidate in enumerate(candidates):
        cand_offers = offers_by_candidate.get(ci, [])
        offer_summary = []
        for oi, offer in enumerate(cand_offers):
            offer_summary.append({
                "index": oi,
                "platform": offer.get("platform", "unknown"),
                "price_usd": offer.get("price_usd"),
                "moq": offer.get("min_order_quantity"),
                "shipping": offer.get("shipping_info", "")[:50],
                "evidence": offer.get("supplier_evidence", [])[:3],
            })

        candidate_data.append({
            "index": ci,
            "product": candidate.get("product", ""),
            "viral_score": candidate.get("viral_score", 0),
            "retail_price": candidate.get("price"),
            "retail_currency": candidate.get("currency", "USD"),
            "offers": offer_summary,
        })

    system_prompt = """You are a product research expert. Perform final supplier-aware ranking.

CRITERIA for each candidate:
- Viral potential: Inherit from initial ranking
- Credible supplier cost: Low cost is better for margins
- Evidence quality: Supplier badges, ratings, reviews
- MOQ: Lower minimum order quantity is better
- Shipping practicality: Free/fast shipping is better
- Potential margin: (Retail - Supplier) / Retail, higher is better
- Availability: Present on both Alibaba and AliExpress is ideal

REJECT candidates if:
- No credible supplier offer (price_usd must be positive)
- Supplier price exceeds typical dropshipping ranges
- Unreliable supplier (no evidence, poor ratings)

For EACH candidate exactly once, return:
- candidate_index: the original zero-based index
- final_score: integer 0-100
- selected_offer_index: the zero-based index of the selected offer for this candidate, or null if rejected
- supplier_price_reasoning: brief reason for selection or rejection (max 150 chars)
- rejection_reason: reason if candidate is rejected (max 150 chars), omit or null if selected

Return ONLY a valid JSON object with 'ranked' array. No other text."""

    user_prompt = f"""Niche: {niche}

Candidates with offers:
{json.dumps(candidate_data, indent=2)}

For each candidate by index, return the required fields in a JSON object.
Maximum 10 ranked products.
Respond with {{'ranked': [{{...}}]}} only."""

    payload = {
        "model": None,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "response_format": {"type": "json_object"},
        "temperature": 0.3,
    }

    last_error = None
    for provider in providers:
        final_payload = _get_provider_payload(payload, provider)
        response_json, error = _call_ai_provider(provider, final_payload)

        if error:
            last_error = error
            continue

        if not response_json:
            last_error = f"Empty response from {provider['name']}"
            continue

        try:
            ranked_data_raw = response_json.get("choices", [{}])[0].get("message", {}).get("content", "")
            if isinstance(ranked_data_raw, str):
                ranked_data = json.loads(ranked_data_raw)
            else:
                ranked_data = ranked_data_raw

            ranked = ranked_data.get("ranked", [])
            if not isinstance(ranked, list):
                last_error = f"Invalid response format from {provider['name']}"
                continue

            # Validate and process
            valid_ranked = []
            seen_indexes = set()
            all_valid = True

            for entry in ranked:
                if not isinstance(entry, dict):
                    all_valid = False
                    break

                candidate_index = entry.get("candidate_index")
                if candidate_index is None or not isinstance(candidate_index, int):
                    all_valid = False
                    break
                if candidate_index < 0 or candidate_index >= len(candidates):
                    all_valid = False
                    break
                if candidate_index in seen_indexes:
                    all_valid = False
                    break
                seen_indexes.add(candidate_index)

                # Get final_score
                final_score = entry.get("final_score")
                if final_score is None:
                    all_valid = False
                    break
                if not isinstance(final_score, (int, float)) or not math.isfinite(final_score):
                    all_valid = False
                    break
                final_score = max(0, min(100, int(final_score)))

                # Get selected_offer_index
                selected_offer_idx = entry.get("selected_offer_index")
                candidate_offers = offers_by_candidate.get(candidate_index, [])

                if selected_offer_idx is not None:
                    # Must be numeric and valid
                    if not isinstance(selected_offer_idx, (int, float)) or not math.isfinite(selected_offer_idx):
                        selected_offer_idx = None
                    elif isinstance(selected_offer_idx, float):
                        selected_offer_idx = int(selected_offer_idx)
                    elif int(selected_offer_idx) != selected_offer_idx:
                        selected_offer_idx = None
                    else:
                        # Check if offer index is valid for this candidate
                        if selected_offer_idx < 0 or selected_offer_idx >= len(candidate_offers):
                            selected_offer_idx = None
                        else:
                            # Check if the offer is credible (has price_usd)
                            selected_offer = candidate_offers[selected_offer_idx]
                            if selected_offer.get("price_usd") is None:
                                selected_offer_idx = None

                # Get rejection_reason
                rejection_reason = entry.get("rejection_reason")
                if rejection_reason and isinstance(rejection_reason, str):
                    rejection_reason = rejection_reason[:150]
                elif rejection_reason:
                    rejection_reason = str(rejection_reason)[:150]

                # Get supplier_price_reasoning
                supplier_price_reasoning = entry.get("supplier_price_reasoning", "")
                if isinstance(supplier_price_reasoning, str):
                    supplier_price_reasoning = supplier_price_reasoning[:150]

                # Validate: if no offer selected, must have rejection reason
                if selected_offer_idx is None and not rejection_reason:
                    all_valid = False
                    break

                # Build final product record
                original_product = products[candidate_index]
                final_product = copy.deepcopy(original_product)
                final_product["final_score"] = final_score
                final_product["ranking_method"] = "ai"
                final_product["supplier_price_reasoning"] = supplier_price_reasoning
                final_product["rejection_reason"] = rejection_reason or ""

                if selected_offer_idx is not None:
                    selected_offer = candidate_offers[selected_offer_idx]
                    final_product["selected_supplier_offer"] = copy.deepcopy(selected_offer)
                else:
                    final_product["selected_supplier_offer"] = None

                valid_ranked.append(final_product)

                if len(valid_ranked) >= 10:
                    break

            if len(seen_indexes) != len(candidates):
                all_valid = False

            if all_valid and valid_ranked:
                valid_ranked.sort(key=lambda p: p.get("final_score", 0), reverse=True)
                return valid_ranked, None
            else:
                last_error = f"Invalid AI response format from {provider['name']}"

        except (json.JSONDecodeError, ValueError, KeyError) as error:
            last_error = f"Failed to parse response from {provider['name']}: {str(error)[:100]}"
            continue

    # Fallback to deterministic ranking
    return _final_rank_fallback(products, supplier_data, offers_by_candidate, candidates)


def _final_rank_fallback(products: list[dict], supplier_data: list[dict],
                         offers_by_candidate: dict, candidates: list[dict]) -> tuple[list[dict], str]:
    """Deterministic fallback for final ranking."""
    final_products = []

    for ci, product in enumerate(products):
        candidate_offers = offers_by_candidate.get(ci, [])

        # Find best credible offer
        best_offer = None
        best_score = -1
        for offer in candidate_offers:
            if offer.get("price_usd") is not None:  # Only credible offers
                offer_score = _score_offer_for_fallback(offer)
                if offer_score > best_score:
                    best_score = offer_score
                    best_offer = offer

        # Compute final score
        final_score = _compute_final_score_fallback(product, best_offer, candidate_offers)

        # Build final product
        final_product = copy.deepcopy(product)
        final_product["final_score"] = round(final_score)
        final_product["ranking_method"] = "deterministic_fallback"
        final_product["selected_supplier_offer"] = copy.deepcopy(best_offer) if best_offer else None
        final_product["_candidate_index"] = ci

        if best_offer:
            final_product["supplier_price_reasoning"] = "Deterministic fallback: best cost-evidence-MOQ-shipping balance"
            final_product["rejection_reason"] = ""
        else:
            final_product["supplier_price_reasoning"] = ""
            final_product["rejection_reason"] = "No credible supplier offers found"

        final_products.append(final_product)

    # Sort by final_score descending, then by candidate_index
    final_products.sort(key=lambda p: (-p.get("final_score", 0), p.get("_candidate_index", 0)))
    for p in final_products:
        p.pop("_candidate_index", None)

    return final_products, "AI final ranking unavailable. Using deterministic fallback based on supplier cost, evidence, MOQ, and shipping."


def run_sourcing_and_final_ranking(niche: str, initial_products: list[dict]) -> dict:
    """
    Run supplier sourcing and final ranking pipeline.
    Returns dict with final_products, supplier_summary, supplier_data, ranking_warnings.
    """
    # Source suppliers for the initial products
    supplier_data, supplier_summary = validate_suppliers_for_products(initial_products)

    # Build ranking warnings
    ranking_warnings = []
    for p in supplier_data:
        for sr in p.get("supplier_results", []):
            if sr.get("error"):
                warning = f"Supplier search failed for {p['product_name']} on {sr['platform']}: {sr['error']}"
                ranking_warnings.append(warning)

    # Run final ranking
    final_products, final_warning = final_rank_products(niche, initial_products, supplier_data)
    if final_warning:
        ranking_warnings.append(final_warning)

    return {
        "final_products": final_products,
        "supplier_summary": supplier_summary,
        "supplier_data": supplier_data,
        "ranking_warnings": ranking_warnings,
    }


class HuntRequest(BaseModel):
    niche: str


class SourceRequest(BaseModel):
    niche: str
    initial_products: list[dict]


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/discover")
async def discover(request: HuntRequest):
    """Run discovery only: scrape, compact, summarize, extract, rank."""
    discovery_data_raw = scrape_with_decodo(request.niche)
    discovery_data = json.loads(discovery_data_raw)

    discovery_summary = [summarize_decodo_result(r) for r in discovery_data]
    initial_products = extract_products(discovery_data)

    # Rank with AI
    ranked_products, ai_warning = analyze_with_ai_model(request.niche, initial_products)

    # Build ranking warnings list
    ranking_warnings = []
    if ai_warning:
        ranking_warnings.append(ai_warning)

    return {
        "niche": request.niche,
        "initial_products": ranked_products,
        "discovery_summary": discovery_summary,
        "discovery_data": discovery_data,
        "ranking_warnings": ranking_warnings,
        "message": f"Discovery complete: {len(ranked_products)} products ranked",
    }


@app.post("/source")
async def source(request: SourceRequest):
    """Run supplier sourcing and final ranking."""
    result = run_sourcing_and_final_ranking(request.niche, request.initial_products)
    return {
        **result,
        "niche": request.niche,
        "message": f"Sourcing and final ranking complete: {len(result['final_products'])} products with suppliers",
    }


@app.post("/hunt")
async def hunt(request: HuntRequest):
    """Complete synchronous pipeline: discover + source + final rank."""
    # Step 1: Discovery
    discovery = run_discovery(request.niche)

    # Step 2: Sourcing and final ranking
    sourcing = run_sourcing_and_final_ranking(request.niche, discovery["initial_products"])

    # Merge results
    result = {
        **discovery,
        **sourcing,
        "message": "Viral product hunt with supplier sourcing and final ranking complete",
    }

    # Merge ranking warnings
    result["ranking_warnings"] = discovery.get("ranking_warnings", []) + sourcing.get("ranking_warnings", [])

    return result


if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=BACKEND_PORT, reload=True)
