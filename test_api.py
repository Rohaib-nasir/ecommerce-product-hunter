"""
Mocked unit tests for api.py - does not make live network calls.
"""
import copy
import json
import math
import os
import unittest
from unittest.mock import MagicMock, patch

import requests

# Set test token before importing api module
os.environ["DECODO_AUTH_TOKEN"] = "test-token"

import api


class TestHealth(unittest.TestCase):
    """Test health check endpoint."""

    def test_health_endpoint(self):
        result = api.health()
        import inspect
        if inspect.iscoroutine(result):
            import asyncio
            result = asyncio.run(result)
        self.assertEqual(result, {"status": "ok"})


class TestBuildSourceUrls(unittest.TestCase):
    """Test discovery source URL builder."""

    def test_returns_ten_sources(self):
        sources = api.build_source_urls("pet gadgets")
        self.assertEqual(len(sources), 10)

    def test_source_order(self):
        sources = api.build_source_urls("cat toy")
        source_names = [s["source"] for s in sources]
        expected_names = [
            "Google Search 1",
            "Google Search 2",
            "Google Search 3",
            "Google Search 4",
            "Google Search 5",
            "Google Search 6",
            "Amazon Search",
            "Reddit Search",
            "TikTok Shop Search",
            "YouTube Search",
        ]
        self.assertEqual(source_names, expected_names)

    def test_amazon_payload(self):
        sources = api.build_source_urls("dog leash")
        amazon = sources[6]
        self.assertEqual(amazon["payload"]["target"], "amazon_search")
        self.assertEqual(amazon["payload"]["query"], "dog leash under $50")
        self.assertEqual(amazon["payload"]["page_from"], "1")
        self.assertTrue(amazon["payload"]["parse"])

    def test_reddit_payload(self):
        sources = api.build_source_urls("pet gadgets")
        reddit = sources[7]
        self.assertEqual(reddit["payload"]["target"], "universal")
        self.assertIn("reddit.com/search.json", reddit["payload"]["url"])
        self.assertNotIn("headless", reddit["payload"])


class TestCompactionAndTrimming(unittest.TestCase):
    """Test compaction and trimming helpers."""

    def test_compact_amazon_organic_limit_and_url_normalization(self):
        raw_amazon = {
            "organic": [
                {
                    "title": f"Item {i}",
                    "price": 10 + i,
                    "extra": "discard",
                    "url": f"/dp/B00000{i}"
                }
                for i in range(30)
            ]
        }
        compacted = api.compact_parsed_content("Amazon Search", raw_amazon)
        self.assertEqual(len(compacted["organic"]), 25)
        self.assertNotIn("extra", compacted["organic"][0])
        self.assertTrue(compacted["organic"][0]["url"].startswith("https://www.amazon.com/dp/"))

    def test_compact_google_limits(self):
        raw_google = {
            "organic": [{"title": f"Item {i}"} for i in range(15)],
            "ai_overviews": [{"text": f"Overview {i}"} for i in range(5)],
            "related_questions": [{"question": f"Q {i}"} for i in range(15)],
            "shopping": [{"title": f"Shop {i}"} for i in range(30)],
        }
        compacted = api.compact_parsed_content("Google Search 1", raw_google)
        self.assertEqual(len(compacted["organic"]), 10)
        self.assertEqual(len(compacted["ai_overviews"]), 3)
        self.assertEqual(len(compacted["related_questions"]), 10)
        self.assertEqual(len(compacted["shopping"]), 25)

    def test_trim_decodo_result(self):
        result = {
            "data": {
                "short": "hello",
                "long": "a" * 5000,
                "nested": {"deep": "b" * 5000}
            }
        }
        trimmed = api.trim_decodo_result(result, 100)
        self.assertEqual(len(trimmed["data"]["long"]), 103)  # 100 chars + "..."
        self.assertTrue(trimmed["data"]["long"].endswith("..."))
        self.assertTrue(trimmed["data"]["nested"]["deep"].endswith("..."))


class TestProductExtraction(unittest.TestCase):
    """Test product extraction and normalization."""

    def test_extract_products_deduplication(self):
        discovery_results = [
            {
                "source": "Google Search 1",
                "data": {
                    "organic": [
                        {
                            "title": "Smart Cat Laser Toy",
                            "price": 19.99,
                            "currency": "USD",
                            "url": "https://example.com/cat-laser",
                            "image": "https://example.com/img1.jpg",
                            "badges": ["Best Seller"]
                        }
                    ]
                }
            },
            {
                "source": "Amazon Search",
                "data": {
                    "organic": [
                        {
                            "title": "Smart Cat Laser Toy!",
                            "price": 19.99,
                            "currency": "USD",
                            "url": "https://amazon.com/dp/B00123",
                            "image": "https://example.com/img2.jpg",
                            "rating": 4.8
                        }
                    ]
                }
            }
        ]
        products = api.extract_products(discovery_results)
        self.assertEqual(len(products), 1)
        prod = products[0]
        self.assertIn("Google Search 1", prod["source"])
        self.assertIn("Amazon Search", prod["source"])
        self.assertGreater(prod["viral_score"], 0)

    def test_viral_score_calculation(self):
        prod = {
            "product": "Test Gadget",
            "price": 25.0,
            "rating": 4.5,
            "reviews": 100,
            "image": "https://example.com/img.jpg",
            "url": "https://example.com",
            "badges": ["Amazon's Choice"]
        }
        score = api._calculate_viral_score(prod)
        self.assertGreaterEqual(score, 80)
        self.assertLessEqual(score, 100)


class TestAIRanking(unittest.TestCase):
    """Test AI provider configuration and ranking."""

    def test_ai_key_validation(self):
        self.assertFalse(api._is_valid_key(None))
        self.assertFalse(api._is_valid_key(""))
        self.assertFalse(api._is_valid_key("   "))
        self.assertFalse(api._is_valid_key("your_openai_api_key_here"))
        self.assertTrue(api._is_valid_key("sk-validkey12345"))

    @patch.object(api, "OPENAI_API_KEY", "sk-test-openai")
    @patch.object(api, "KIMI_API_KEY", "your_placeholder_key")
    @patch.object(api, "MOONSHOT_API_KEY", None)
    @patch.object(api, "AI_MODEL_API_KEY", None)
    def test_ai_provider_priority(self):
        providers = api.get_ai_providers()
        self.assertEqual(len(providers), 1)
        self.assertEqual(providers[0]["name"], "OpenAI")

    @patch("requests.post")
    @patch.object(api, "OPENAI_API_KEY", "sk-test-key")
    def test_analyze_with_ai_model_success(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps({
                            "ranked": [
                                {
                                    "candidate_index": 0,
                                    "viral_score": 95,
                                    "reason": "Solves a clear problem with great video potential."
                                }
                            ]
                        })
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        products = [
            {"product": "Laser Toy", "price": 15.0, "viral_score": 70}
        ]
        ranked, warning = api.analyze_with_ai_model("pet gadgets", products)
        self.assertIsNone(warning)
        self.assertEqual(len(ranked), 1)
        self.assertEqual(ranked[0]["ai_viral_score"], 95)
        self.assertEqual(ranked[0]["ai_reason"], "Solves a clear problem with great video potential.")

    @patch("requests.post")
    @patch.object(api, "OPENAI_API_KEY", "sk-invalid-key")
    def test_ai_401_warning(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_error = requests.exceptions.HTTPError("401 Unauthorized", response=mock_response)
        mock_post.side_effect = mock_error

        products = [{"product": "Laser Toy", "viral_score": 70}]
        ranked, warning = api.analyze_with_ai_model("pet gadgets", products)
        self.assertIsNotNone(warning)
        self.assertIn("authentication failed", warning.lower())
        self.assertEqual(len(ranked), 1)

    @patch("requests.post")
    @patch.object(api, "OPENAI_API_KEY", "sk-quota-exceeded")
    def test_ai_429_warning(self, mock_post):
        mock_response = MagicMock()
        mock_response.status_code = 429
        mock_error = requests.exceptions.HTTPError("429 Rate Limit", response=mock_response)
        mock_post.side_effect = mock_error

        products = [{"product": "Laser Toy", "viral_score": 70}]
        ranked, warning = api.analyze_with_ai_model("pet gadgets", products)
        self.assertIsNotNone(warning)
        self.assertIn("quota/rate limit", warning.lower())
        self.assertEqual(len(ranked), 1)


class TestSupplierSourcing(unittest.TestCase):
    """Test supplier sourcing and parsing."""

    def test_build_supplier_payloads(self):
        payloads = api.build_supplier_payloads("pet laser toy")
        self.assertEqual(len(payloads), 2)
        self.assertIn("alibaba.com", payloads[0]["url"])
        self.assertIn("aliexpress.com", payloads[1]["url"])
        self.assertEqual(payloads[0]["proxy_pool"], "premium")
        self.assertEqual(payloads[0]["headless"], "html")

    def test_marketplace_parser_alibaba(self):
        html_content = """
        <html>
            <body>
                <a href="https://www.alibaba.com/product-detail/Interactive-Cat-Laser-Pointer-Toy_1600123.html">
                    Interactive Cat Laser Pointer Toy $2.50
                </a>
            </body>
        </html>
        """
        parser = api.MarketplaceParser("alibaba")
        parser.feed(html_content)
        self.assertEqual(len(parser.listings), 1)
        self.assertEqual(parser.listings[0]["platform"], "alibaba")
        self.assertIn("Interactive Cat Laser Pointer Toy", parser.listings[0]["supplier_title"])

    @patch("api._scrape_supplier_payload")
    def test_one_marketplace_failing(self, mock_scrape):
        def side_effect(source_name, payload):
            if "alibaba" in source_name:
                return {
                    "source": source_name,
                    "request": payload,
                    "data": '<html><body><a href="https://www.alibaba.com/product-detail/p1.html">Laser Toy $3.00</a></body></html>',
                    "error": None
                }
            else:
                return {
                    "source": source_name,
                    "request": payload,
                    "data": None,
                    "error": "HTTP 500 Server Error"
                }
        mock_scrape.side_effect = side_effect

        products = [{"product": "Cat Laser Toy"}]
        supplier_data, summary = api.validate_suppliers_for_products(products)

        self.assertEqual(summary["products_validated"], 1)
        self.assertEqual(summary["successful_supplier_requests"], 1)
        self.assertEqual(summary["failed_supplier_requests"], 1)
        self.assertEqual(len(supplier_data[0]["alibaba_offers"]), 1)
        self.assertEqual(len(supplier_data[0]["aliexpress_offers"]), 0)

    @patch("api._scrape_supplier_payload")
    def test_missing_offers(self, mock_scrape):
        mock_scrape.return_value = {
            "source": "mock",
            "request": {},
            "data": "<html><body>No products found</body></html>",
            "error": None
        }
        products = [{"product": "Rare Niche Product"}]
        supplier_data, summary = api.validate_suppliers_for_products(products)
        self.assertEqual(summary["alibaba_offer_count"], 0)
        self.assertEqual(summary["aliexpress_offer_count"], 0)


class TestCurrencyAndPrices(unittest.TestCase):
    """Test currency conversions and price credibility."""

    def test_currency_conversions(self):
        self.assertAlmostEqual(api.convert_to_usd(10.0, "USD"), 10.0)
        self.assertAlmostEqual(api.convert_to_usd(10.0, "EUR"), 10.9)
        self.assertAlmostEqual(api.convert_to_usd(10.0, "GBP"), 12.8)
        self.assertAlmostEqual(api.convert_to_usd(100.0, "CNY"), 14.0)
        self.assertAlmostEqual(api.convert_to_usd(100.0, "RMB"), 14.0)
        self.assertAlmostEqual(api.convert_to_usd(10.0, "AUD"), 6.6)
        self.assertAlmostEqual(api.convert_to_usd(10.0, "CAD"), 7.3)

    def test_invalid_currency_and_non_finite_price(self):
        self.assertIsNone(api.convert_to_usd(10.0, "XYZ"))
        self.assertIsNone(api.convert_to_usd(-5.0, "USD"))
        self.assertIsNone(api.convert_to_usd(float("nan"), "USD"))
        self.assertIsNone(api.convert_to_usd(float("inf"), "USD"))

    def test_credible_price_extraction(self):
        offer_unit = {"unit_price": 5.0, "currency": "USD"}
        price_usd, curr = api.get_credible_price(offer_unit)
        self.assertEqual(price_usd, 5.0)
        self.assertEqual(curr, "USD")

        offer_range = {"price_range": {"min": 3.0, "max": 8.0}, "currency": "EUR"}
        price_usd, curr = api.get_credible_price(offer_range)
        self.assertAlmostEqual(price_usd, 3.27)
        self.assertEqual(curr, "EUR")

        offer_invalid = {"unit_price": -10.0, "currency": "USD"}
        price_usd, curr = api.get_credible_price(offer_invalid)
        self.assertIsNone(price_usd)


class TestFinalRankingAndProvenance(unittest.TestCase):
    """Test final ranking logic, validation, and offer provenance."""

    def test_offer_provenance(self):
        raw_offer = {
            "supplier_title": "Cat Laser Pointer",
            "supplier_url": "https://www.alibaba.com/product-detail/123.html",
            "unit_price": 2.50,
            "currency": "USD"
        }
        normalized = api._normalize_supplier_offer(
            raw_offer, "alibaba", "Cat Laser Pointer", "https://search.url", 0
        )
        self.assertIsNotNone(normalized)
        prov = normalized["provenance"]
        self.assertEqual(prov["source"], "marketplace")
        self.assertEqual(prov["platform"], "alibaba")
        self.assertEqual(prov["search_url"], "https://search.url")
        self.assertEqual(prov["candidate_index"], 0)

    def test_final_rank_deterministic_fallback(self):
        products = [
            {"product": "Cat Laser Toy", "price": 20.0, "currency": "USD", "viral_score": 80},
            {"product": "Dog Chew Bone", "price": 15.0, "currency": "USD", "viral_score": 75},
        ]
        supplier_data = [
            {
                "product_name": "Cat Laser Toy",
                "candidate_index": 0,
                "supplier_results": [
                    {
                        "platform": "alibaba",
                        "error": None,
                        "offers": [
                            {
                                "supplier_title": "Alibaba Laser Toy",
                                "supplier_url": "https://alibaba.com/p1",
                                "unit_price": 3.0,
                                "currency": "USD",
                                "min_order_quantity": 1,
                                "shipping_info": "Free shipping",
                                "supplier_evidence": ["Verified Supplier"]
                            }
                        ]
                    }
                ],
                "alibaba_offers": [
                    {
                        "supplier_title": "Alibaba Laser Toy",
                        "supplier_url": "https://alibaba.com/p1",
                        "unit_price": 3.0,
                        "currency": "USD",
                        "min_order_quantity": 1,
                        "shipping_info": "Free shipping",
                        "supplier_evidence": ["Verified Supplier"]
                    }
                ],
                "aliexpress_offers": []
            },
            {
                "product_name": "Dog Chew Bone",
                "candidate_index": 1,
                "supplier_results": [],
                "alibaba_offers": [],
                "aliexpress_offers": []
            }
        ]

        final_products, warning = api.final_rank_products("pet gadgets", products, supplier_data)
        self.assertIsNotNone(warning)
        self.assertEqual(len(final_products), 2)
        top_product = final_products[0]
        self.assertEqual(top_product["product"], "Cat Laser Toy")
        self.assertEqual(top_product["ranking_method"], "deterministic_fallback")
        self.assertIsNotNone(top_product["selected_supplier_offer"])
        self.assertEqual(top_product["selected_supplier_offer"]["supplier_title"], "Alibaba Laser Toy")

        second_product = final_products[1]
        self.assertEqual(second_product["product"], "Dog Chew Bone")
        self.assertIsNone(second_product["selected_supplier_offer"])
        self.assertEqual(second_product["rejection_reason"], "No credible supplier offers found")


class TestFullPipeline(unittest.TestCase):
    """Test full pipeline endpoints."""

    @patch("api.scrape_with_decodo")
    @patch("api.validate_suppliers_for_products")
    @patch.object(api, "OPENAI_API_KEY", None)
    @patch.object(api, "KIMI_API_KEY", None)
    def test_hunt_pipeline(self, mock_validate, mock_scrape):
        mock_scrape.return_value = json.dumps([
            {
                "source": "Google Search 1",
                "error": None,
                "data": {
                    "organic": [
                        {
                            "title": "Viral Pet Fountain",
                            "price": 29.99,
                            "currency": "USD",
                            "url": "https://example.com/fountain",
                            "image": "https://example.com/img.jpg"
                        }
                    ]
                }
            }
        ])

        mock_validate.return_value = (
            [
                {
                    "product_name": "Viral Pet Fountain",
                    "candidate_index": 0,
                    "supplier_results": [],
                    "alibaba_offers": [],
                    "aliexpress_offers": []
                }
            ],
            {
                "products_validated": 1,
                "total_supplier_requests": 2,
                "successful_supplier_requests": 2,
                "failed_supplier_requests": 0,
                "empty_supplier_requests": 2,
                "alibaba_offer_count": 0,
                "aliexpress_offer_count": 0,
            }
        )

        request_data = api.HuntRequest(niche="pet gadgets")
        import asyncio
        import inspect

        result = api.hunt(request_data)
        if inspect.iscoroutine(result):
            result = asyncio.run(result)

        expected_keys = {
            "niche", "message", "initial_products", "final_products",
            "supplier_summary", "supplier_data", "discovery_summary",
            "discovery_data", "ranking_warnings"
        }
        self.assertTrue(expected_keys.issubset(result.keys()))
        self.assertEqual(result["niche"], "pet gadgets")
        self.assertEqual(len(result["initial_products"]), 1)
        self.assertEqual(len(result["final_products"]), 1)


if __name__ == "__main__":
    unittest.main()