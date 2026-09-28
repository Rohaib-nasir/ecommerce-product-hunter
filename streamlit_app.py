"""
Streamlit frontend for the Viral Product Hunter application.
"""
import copy
import io
import os
import requests
import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# Page Configuration & CSS
# ============================================================
st.set_page_config(
    page_title="Viral Product Hunter AI",
    page_icon="⚡",
    layout="wide",
)

st.markdown("""
<style>
    .main .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem;
    }
    .stContainer {
        border-radius: 8px;
        padding: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Mock Data Samples
# ============================================================
MOCK_DISCOVER = {
    "niche": "pet gadgets",
    "initial_products": [
        {
            "product": "Interactive Smart Cat Laser Ball",
            "source": "Google Search 1, Amazon Search",
            "price": 19.99,
            "currency": "USD",
            "url": "https://www.amazon.com/dp/B08123456",
            "image": "https://images.unsplash.com/photo-1545249390-6bdfa286032f?w=300",
            "viral_score": 92,
            "ai_viral_score": 94,
            "ai_reason": "High video engagement potential on TikTok, solves boredom for indoor cats.",
            "rating": 4.8,
            "reviews": 1250,
            "sales": 5000,
            "evidence": ["Best Seller", "Amazon's Choice"],
            "badges": ["Best Seller"],
            "asin": "B08123456"
        },
        {
            "product": "Ultrasonic Dog Bark Control Device",
            "source": "Google Search 2, TikTok Shop Search",
            "price": 24.50,
            "currency": "USD",
            "url": "https://shop.tiktok.com/bark-control",
            "image": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=300",
            "viral_score": 85,
            "ai_viral_score": 88,
            "ai_reason": "Clear problem-solver for noisy pets, strong before/after video hook.",
            "rating": 4.5,
            "reviews": 840,
            "sales": 3100,
            "evidence": ["TikTok Viral"],
            "badges": [],
            "asin": None
        },
        {
            "product": "Self-Cleaning One-Click Pet Grooming Brush",
            "source": "Amazon Search",
            "price": 14.99,
            "currency": "USD",
            "url": "https://www.amazon.com/dp/B09876543",
            "image": None,  # Missing image sample
            "viral_score": 78,
            "ai_viral_score": 80,
            "ai_reason": "Satisfying viral video potential showing one-click fur release mechanism.",
            "rating": 4.6,
            "reviews": 2100,
            "sales": 8000,
            "evidence": ["Amazon's Choice"],
            "badges": ["Amazon's Choice"],
            "asin": "B09876543"
        }
    ],
    "discovery_summary": [
        {"source": "Google Search 1", "parser_status": "success", "extracted_product_count": 5, "error": ""},
        {"source": "Google Search 2", "parser_status": "success", "extracted_product_count": 3, "error": ""},
        {"source": "Amazon Search", "parser_status": "success", "extracted_product_count": 8, "error": ""},
        {"source": "TikTok Shop Search", "parser_status": "success", "extracted_product_count": 4, "error": ""},
        {"source": "Reddit Search", "parser_status": "success", "extracted_product_count": 2, "error": ""}
    ],
    "discovery_data": [
        {"source": "Google Search 1", "data": {"organic": [{"title": "Interactive Smart Cat Laser Ball", "price": 19.99}]}}
    ],
    "ranking_warnings": [],
    "message": "Discovery complete: 3 products ranked"
}

MOCK_SOURCE = {
    "niche": "pet gadgets",
    "final_products": [
        {
            "product": "Interactive Smart Cat Laser Ball",
            "source": "Google Search 1, Amazon Search",
            "price": 19.99,
            "currency": "USD",
            "url": "https://www.amazon.com/dp/B08123456",
            "image": "https://images.unsplash.com/photo-1545249390-6bdfa286032f?w=300",
            "viral_score": 92,
            "ai_viral_score": 94,
            "final_score": 91,
            "ranking_method": "ai",
            "supplier_price_reasoning": "Excellent margin ($19.99 retail vs $3.50 supplier), verified Gold Supplier.",
            "rejection_reason": "",
            "selected_supplier_offer": {
                "product_reference": "Interactive Smart Cat Laser Ball",
                "supplier_platform": "alibaba",
                "supplier_title": "Wholesale Smart Automatic Rolling Cat Ball Toy",
                "unit_price": 3.50,
                "price_usd": 3.50,
                "currency": "USD",
                "min_order_quantity": 10,
                "shipping_info": "Free Shipping via ePacket",
                "supplier_url": "https://www.alibaba.com/product-detail/smart-cat-ball_1600.html",
                "rating": 4.9,
                "orders": 15000,
                "reviews": 1200,
                "supplier_evidence": ["Verified Supplier", "Trade Assurance"]
            }
        },
        {
            "product": "Ultrasonic Dog Bark Control Device",
            "source": "Google Search 2, TikTok Shop Search",
            "price": 24.50,
            "currency": "USD",
            "url": "https://shop.tiktok.com/bark-control",
            "image": "https://images.unsplash.com/photo-1583511655857-d19b40a7a54e?w=300",
            "viral_score": 85,
            "ai_viral_score": 88,
            "final_score": 82,
            "ranking_method": "deterministic_fallback",
            "supplier_price_reasoning": "Deterministic fallback: best cost-evidence-MOQ-shipping balance.",
            "rejection_reason": "",
            "selected_supplier_offer": {
                "product_reference": "Ultrasonic Dog Bark Control Device",
                "supplier_platform": "alibaba",
                "supplier_title": "Handheld Ultrasonic Dog Trainer Bark Stopper",
                "unit_price": 5.20,
                "price_usd": 5.20,
                "currency": "USD",
                "min_order_quantity": 20,
                "shipping_info": "DHL Express $15",
                "supplier_url": "https://www.alibaba.com/product-detail/bark-control_1601.html",
                "rating": 4.5,
                "orders": 890,
                "reviews": 110,
                "supplier_evidence": ["Gold Supplier"]
            }
        },
        {
            "product": "Self-Cleaning One-Click Pet Grooming Brush",
            "source": "Amazon Search",
            "price": 14.99,
            "currency": "USD",
            "url": "https://www.amazon.com/dp/B09876543",
            "image": None,
            "viral_score": 78,
            "ai_viral_score": 80,
            "final_score": 23,
            "ranking_method": "deterministic_fallback",
            "supplier_price_reasoning": "",
            "rejection_reason": "No credible supplier offers found",  # Missing offers sample
            "selected_supplier_offer": None
        }
    ],
    "supplier_summary": {
        "products_validated": 3,
        "total_supplier_requests": 6,
        "successful_supplier_requests": 5,
        "failed_supplier_requests": 1,
        "empty_supplier_requests": 1,
        "alibaba_offer_count": 2,
        "aliexpress_offer_count": 1
    },
    "supplier_data": [
        {
            "product_name": "Interactive Smart Cat Laser Ball",
            "candidate_index": 0,
            "supplier_results": [
                {"platform": "alibaba", "error": None, "has_data": True},
                {"platform": "aliexpress", "error": None, "has_data": True}
            ],
            "alibaba_offers": [
                {
                    "supplier_title": "Wholesale Smart Automatic Rolling Cat Ball Toy",
                    "unit_price": 3.50,
                    "currency": "USD",
                    "min_order_quantity": 10,
                    "shipping_info": "Free Shipping via ePacket",
                    "rating": 4.9,
                    "orders": 15000,
                    "supplier_url": "https://www.alibaba.com/product-detail/smart-cat-ball_1600.html"
                }
            ],
            "aliexpress_offers": [
                {
                    "supplier_title": "Automatic Cat Ball Toy LED Light",
                    "unit_price": 4.10,
                    "currency": "USD",
                    "min_order_quantity": 1,
                    "shipping_info": "Free Shipping",
                    "rating": 4.6,
                    "orders": 2400,
                    "supplier_url": "https://www.aliexpress.com/item/1005002.html"
                }
            ]
        },
        {
            "product_name": "Ultrasonic Dog Bark Control Device",
            "candidate_index": 1,
            "supplier_results": [
                {"platform": "alibaba", "error": None, "has_data": True},
                {"platform": "aliexpress", "error": "AliExpress search timeout after 120s", "has_data": False}  # Failed marketplace sample
            ],
            "alibaba_offers": [
                {
                    "supplier_title": "Handheld Ultrasonic Dog Trainer Bark Stopper",
                    "price_range": {"min": 4.80, "max": 6.50},
                    "currency": "USD",
                    "min_order_quantity": 20,
                    "shipping_info": "DHL Express $15",
                    "rating": 4.5,
                    "orders": 890,
                    "supplier_url": "https://www.alibaba.com/product-detail/bark-control_1601.html"
                }
            ],
            "aliexpress_offers": []
        },
        {
            "product_name": "Self-Cleaning One-Click Pet Grooming Brush",
            "candidate_index": 2,
            "supplier_results": [
                {"platform": "alibaba", "error": None, "has_data": False},
                {"platform": "aliexpress", "error": None, "has_data": False}
            ],
            "alibaba_offers": [],
            "aliexpress_offers": []  # Empty offers sample
        }
    ],
    "ranking_warnings": [
        "Supplier search failed for Ultrasonic Dog Bark Control Device on aliexpress: AliExpress search timeout after 120s",
        "AI final ranking unavailable. Using deterministic fallback based on supplier cost, evidence, MOQ, and shipping."
    ],
    "message": "Sourcing and final ranking complete: 3 products with suppliers"
}


# ============================================================
# Helper Functions
# ============================================================
def fmt_val(val, default="N/A"):
    if val is None or val == "" or str(val).lower() in ("none", "nan"):
        return default
    return str(val)


def fmt_score(val):
    if val is None:
        return 0
    try:
        return int(float(val))
    except (ValueError, TypeError):
        return 0


# ============================================================
# Sidebar Configuration & Health Indicator
# ============================================================
st.sidebar.header("Settings")

default_backend_url = os.getenv("BACKEND_API_URL") or "http://127.0.0.1:8000"
default_read_timeout = int(os.getenv("BACKEND_READ_TIMEOUT", 600))

backend_url = st.sidebar.text_input("Backend API URL", value=default_backend_url)
read_timeout = st.sidebar.number_input("Backend Read Timeout (s)", value=default_read_timeout, min_value=10, max_value=3600)
mock_mode = st.sidebar.toggle("MOCK mode", value=False)

# Backend Health Status Indicator
if mock_mode:
    st.sidebar.success("🟢 Backend online (MOCK)")
else:
    try:
        health_resp = requests.get(f"{backend_url}/health", timeout=3)
        if health_resp.status_code == 200 and health_resp.json().get("status") == "ok":
            st.sidebar.success("🟢 Backend online")
        else:
            st.sidebar.error("🔴 Backend offline")
    except Exception:
        st.sidebar.error("🔴 Backend offline")

# ============================================================
# Main Header & Controls
# ============================================================
st.title("Viral Product Hunter AI")
st.caption("Discover trending e-commerce products, source suppliers from Alibaba & AliExpress, and rank candidates.")

if "hunt_results" not in st.session_state:
    st.session_state.hunt_results = None
if "sourcing_error" not in st.session_state:
    st.session_state.sourcing_error = None
if "is_hunting" not in st.session_state:
    st.session_state.is_hunting = False

col_input, col_btn = st.columns([4, 1])
with col_input:
    niche = st.text_input(
        "Enter a product niche:",
        value="pet gadgets",
        label_visibility="collapsed",
        placeholder="Enter a product niche (e.g. pet gadgets)",
    )
with col_btn:
    hunt_clicked = st.button(
        "Hunt Products",
        type="primary",
        use_container_width=True,
        disabled=st.session_state.is_hunting,
    )

# ============================================================
# Staged Execution Flow
# ============================================================
if hunt_clicked:
    st.session_state.is_hunting = True
    st.session_state.hunt_results = None
    st.session_state.sourcing_error = None

    timeout_config = (10, int(read_timeout))

    if mock_mode:
        with st.status("Discovering products...", expanded=True) as status:
            discover_res = copy.deepcopy(MOCK_DISCOVER)
            initial_count = len(discover_res.get("initial_products", []))
            status.update(label=f"Discovered {initial_count} products")

            status.update(label="Sourcing Alibaba and AliExpress offers...")
            source_res = copy.deepcopy(MOCK_SOURCE)
            status.update(label="Complete!", state="complete")

        merged = {**discover_res, **source_res}
        merged["ranking_warnings"] = (
            discover_res.get("ranking_warnings", []) + source_res.get("ranking_warnings", [])
        )
        st.session_state.hunt_results = merged
        st.session_state.is_hunting = False
    else:
        discover_res = None
        # Stage 1: Discovery
        try:
            with st.status("Discovering products...", expanded=True) as status:
                resp = requests.post(f"{backend_url}/discover", json={"niche": niche}, timeout=timeout_config)
                resp.raise_for_status()
                discover_res = resp.json()
                initial_count = len(discover_res.get("initial_products", []))
                status.update(label=f"Discovered {initial_count} products")

                # Store discovery results early in case sourcing fails
                st.session_state.hunt_results = copy.deepcopy(discover_res)

                # Stage 2: Sourcing
                status.update(label="Sourcing Alibaba and AliExpress offers...")
                resp_src = requests.post(
                    f"{backend_url}/source",
                    json={
                        "niche": niche,
                        "initial_products": discover_res.get("initial_products", []),
                    },
                    timeout=timeout_config,
                )
                resp_src.raise_for_status()
                source_res = resp_src.json()

                status.update(label="Complete!", state="complete")

                merged = {**discover_res, **source_res}
                merged["ranking_warnings"] = (
                    discover_res.get("ranking_warnings", []) + source_res.get("ranking_warnings", [])
                )
                st.session_state.hunt_results = merged
        except requests.exceptions.RequestException as error:
            if discover_res is None:
                st.error(f"Failed to connect to the backend: {error}. Please start FastAPI at {backend_url}")
            else:
                st.session_state.sourcing_error = f"Supplier sourcing failed: {error}"
        finally:
            st.session_state.is_hunting = False

# ============================================================
# Display Warnings & Results
# ============================================================
if st.session_state.sourcing_error:
    st.error(st.session_state.sourcing_error)

if st.session_state.hunt_results:
    results = st.session_state.hunt_results

    # Render ranking warnings above tabs
    ranking_warnings = results.get("ranking_warnings", [])
    for warning in ranking_warnings:
        st.warning(warning)

    tab_initial, tab_supplier, tab_final, tab_discovery = st.tabs([
        "Initial Ranking",
        "Supplier Sourcing",
        "Final Ranking",
        "Discovery Details",
    ])

    # ------------------------------------------------------------
    # Tab 1: Initial Ranking
    # ------------------------------------------------------------
    with tab_initial:
        initial_products = results.get("initial_products", [])
        if initial_products:
            display_initial_rows = []
            for idx, p in enumerate(initial_products, start=1):
                score = fmt_score(p.get("ai_viral_score", p.get("viral_score", 0)))
                price = p.get("price")
                currency = p.get("currency", "USD")
                price_str = f"{currency} ${price:.2f}" if (price is not None and isinstance(price, (int, float))) else "N/A"
                url = p.get("url") or ""
                img = p.get("image") or ""

                display_initial_rows.append({
                    "Rank": idx,
                    "Image": img if img else None,
                    "Product": fmt_val(p.get("product")),
                    "Viral Score": score,
                    "Retail Price": price_str,
                    "Currency": fmt_val(currency),
                    "Source": fmt_val(p.get("source")),
                    "URL": url if url else "",
                })

            st.dataframe(
                display_initial_rows,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "Image": st.column_config.ImageColumn("Image", help="Product thumbnail"),
                    "Product": st.column_config.TextColumn("Product"),
                    "Viral Score": st.column_config.ProgressColumn("Viral Score", min_value=0, max_value=100, format="%d"),
                    "Retail Price": st.column_config.TextColumn("Retail Price"),
                    "Currency": st.column_config.TextColumn("Currency"),
                    "Source": st.column_config.TextColumn("Source"),
                    "URL": st.column_config.LinkColumn("Product Link", display_text="View Product"),
                },
                hide_index=True,
                use_container_width=True,
            )
        else:
            st.info("No products extracted from discovery sources.")

    # ------------------------------------------------------------
    # Tab 2: Supplier Sourcing
    # ------------------------------------------------------------
    with tab_supplier:
        supplier_summary = results.get("supplier_summary", {})
        supplier_data = results.get("supplier_data", [])

        if supplier_summary:
            col1, col2, col3, col4, col5 = st.columns(5)
            col1.metric("Products Validated", supplier_summary.get("products_validated", 0))
            col2.metric("Total Requests", supplier_summary.get("total_supplier_requests", 0))
            col3.metric("Successful", supplier_summary.get("successful_supplier_requests", 0))
            col4.metric("Failed", supplier_summary.get("failed_supplier_requests", 0))
            col5.metric("Empty", supplier_summary.get("empty_supplier_requests", 0))

            st.caption(
                f"Alibaba Offers: {supplier_summary.get('alibaba_offer_count', 0)} | "
                f"AliExpress Offers: {supplier_summary.get('aliexpress_offer_count', 0)}"
            )

            for product in supplier_data:
                product_name = fmt_val(product.get("product_name"), "Unknown Product")
                alibaba_offers = product.get("alibaba_offers", [])
                aliexpress_offers = product.get("aliexpress_offers", [])
                supplier_results = product.get("supplier_results", [])

                with st.expander(f"📦 {product_name} - Supplier Offers", expanded=False):
                    for sr in supplier_results:
                        if sr.get("error"):
                            st.warning(f"⚠️ {sr.get('platform', 'marketplace').upper()} search failed: {sr['error']}")

                    all_offers = []
                    for offer in alibaba_offers:
                        c = copy.deepcopy(offer)
                        c["platform"] = "Alibaba"
                        all_offers.append(c)
                    for offer in aliexpress_offers:
                        c = copy.deepcopy(offer)
                        c["platform"] = "AliExpress"
                        all_offers.append(c)

                    if not all_offers:
                        st.info("No offers found")
                    else:
                        offer_rows = []
                        for offer in all_offers:
                            price_str = "N/A"
                            if "unit_price" in offer and offer["unit_price"] is not None:
                                price_str = f"${offer['unit_price']:.2f}"
                            elif "price_range" in offer and offer["price_range"]:
                                pr = offer["price_range"]
                                price_str = f"${pr.get('min', 0):.2f} - ${pr.get('max', 0):.2f}"

                            moq = offer.get("min_order_quantity")
                            moq_str = str(moq) if moq is not None else "N/A"
                            shipping = fmt_val(offer.get("shipping_info"))
                            rating = offer.get("rating")
                            rating_str = f"{rating:.1f}" if (rating is not None and isinstance(rating, (int, float))) else "N/A"
                            orders = offer.get("orders")
                            orders_str = f"{int(orders):,}" if (orders is not None and isinstance(orders, (int, float))) else "N/A"
                            url = offer.get("supplier_url") or ""

                            offer_rows.append({
                                "Platform": offer.get("platform", "N/A"),
                                "Listing": fmt_val(offer.get("supplier_title"))[:60],
                                "Price/Range": price_str,
                                "Currency": fmt_val(offer.get("currency"), "USD"),
                                "MOQ": moq_str,
                                "Shipping": shipping[:40],
                                "Rating": rating_str,
                                "Orders": orders_str,
                                "Supplier Link": url if url else "",
                            })

                        st.dataframe(
                            offer_rows,
                            column_config={
                                "Platform": st.column_config.TextColumn("Platform"),
                                "Listing": st.column_config.TextColumn("Listing"),
                                "Price/Range": st.column_config.TextColumn("Price/Range"),
                                "Currency": st.column_config.TextColumn("Currency"),
                                "MOQ": st.column_config.TextColumn("MOQ"),
                                "Shipping": st.column_config.TextColumn("Shipping"),
                                "Rating": st.column_config.TextColumn("Rating"),
                                "Orders": st.column_config.TextColumn("Orders"),
                                "Supplier Link": st.column_config.LinkColumn("Supplier Link", display_text="View Offer"),
                            },
                            hide_index=True,
                            use_container_width=True,
                        )
        else:
            st.info("No supplier sourcing data available.")

    # ------------------------------------------------------------
    # Tab 3: Final Ranking
    # ------------------------------------------------------------
    with tab_final:
        final_products = results.get("final_products", [])
        if final_products:
            st.subheader("Top Ranked Products")
            top_3 = final_products[:3]
            cols = st.columns(len(top_3))
            for i, p in enumerate(top_3):
                with cols[i]:
                    with st.container(border=True):
                        rank_num = i + 1
                        score = fmt_score(p.get("final_score"))
                        product_title = fmt_val(p.get("product"))
                        method = p.get("ranking_method", "")
                        method_badge = "🤖 AI" if method == "ai" else "⚡ Fallback"

                        st.markdown(f"#### #{rank_num} {product_title}")
                        st.caption(f"Final Score: **{score}/100** ({method_badge})")

                        offer = p.get("selected_supplier_offer")
                        if offer:
                            sup_title = fmt_val(offer.get("supplier_title"))
                            cost_val = offer.get("price_usd") or offer.get("unit_price")
                            cost_str = f"${cost_val:.2f}" if (cost_val is not None and isinstance(cost_val, (int, float))) else "N/A"
                            moq_str = str(offer.get("min_order_quantity")) if offer.get("min_order_quantity") is not None else "N/A"
                            st.write(f"**Supplier:** {sup_title[:35]}...")
                            st.write(f"**USD Cost:** `{cost_str}` | **MOQ:** `{moq_str}`")
                            if p.get("supplier_price_reasoning"):
                                st.info(f"💡 {p['supplier_price_reasoning']}")
                        else:
                            st.write("**Supplier:** N/A")
                            rej = fmt_val(p.get("rejection_reason"), "No credible offers found")
                            st.warning(f"❌ {rej}")

            st.divider()

            # Full Final Dataframe
            display_final_rows = []
            for idx, p in enumerate(final_products, start=1):
                score = fmt_score(p.get("final_score"))
                offer = p.get("selected_supplier_offer")
                method = "AI" if p.get("ranking_method") == "ai" else "Fallback"

                sup_name = fmt_val(offer.get("supplier_title")) if offer else "N/A"

                cost_str = "N/A"
                if offer:
                    if "unit_price" in offer and offer["unit_price"] is not None:
                        cost_str = f"${offer['unit_price']:.2f}"
                    elif "price_range" in offer and offer["price_range"]:
                        pr = offer["price_range"]
                        cost_str = f"${pr.get('min', 0):.2f} - ${pr.get('max', 0):.2f}"
                    elif offer.get("price_usd") is not None:
                        cost_str = f"${offer['price_usd']:.2f}"

                moq_str = str(offer.get("min_order_quantity")) if (offer and offer.get("min_order_quantity") is not None) else "N/A"
                reasoning = fmt_val(p.get("supplier_price_reasoning") or p.get("rejection_reason"))
                sup_url = offer.get("supplier_url") if offer else None

                display_final_rows.append({
                    "Rank": idx,
                    "Product": fmt_val(p.get("product")),
                    "Final Score": score,
                    "Method": method,
                    "Selected Supplier": sup_name,
                    "Supplier Cost (USD)": cost_str,
                    "MOQ": moq_str,
                    "Reasoning / Rejection": reasoning,
                    "Supplier Link": sup_url if sup_url else "",
                })

            st.dataframe(
                display_final_rows,
                column_config={
                    "Rank": st.column_config.NumberColumn("Rank", format="%d"),
                    "Product": st.column_config.TextColumn("Product"),
                    "Final Score": st.column_config.ProgressColumn("Final Score", min_value=0, max_value=100, format="%d"),
                    "Method": st.column_config.TextColumn("Method"),
                    "Selected Supplier": st.column_config.TextColumn("Selected Supplier"),
                    "Supplier Cost (USD)": st.column_config.TextColumn("Supplier Cost (USD)"),
                    "MOQ": st.column_config.TextColumn("MOQ"),
                    "Reasoning / Rejection": st.column_config.TextColumn("Reasoning / Rejection"),
                    "Supplier Link": st.column_config.LinkColumn("Supplier Link", display_text="View Supplier"),
                },
                hide_index=True,
                use_container_width=True,
            )

            # CSV Download Button
            df_export = pd.DataFrame(display_final_rows)
            csv_bytes = df_export.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="📥 Download Final Ranking CSV",
                data=csv_bytes,
                file_name=f"final_ranking_{niche.replace(' ', '_')}.csv",
                mime="text/csv",
            )
        else:
            st.info("No final products with supplier ranking available.")

    # ------------------------------------------------------------
    # Tab 4: Discovery Details
    # ------------------------------------------------------------
    with tab_discovery:
        discovery_summary = results.get("discovery_summary", [])
        if discovery_summary:
            summary_rows = []
            for s in discovery_summary:
                summary_rows.append({
                    "Source": fmt_val(s.get("source")),
                    "Status": fmt_val(s.get("parser_status")),
                    "Extracted Products": s.get("extracted_product_count", 0),
                    "Error": fmt_val(s.get("error"), "None"),
                })
            st.dataframe(
                summary_rows,
                column_config={
                    "Source": st.column_config.TextColumn("Source"),
                    "Status": st.column_config.TextColumn("Status"),
                    "Extracted Products": st.column_config.NumberColumn("Extracted Products", format="%d"),
                    "Error": st.column_config.TextColumn("Error"),
                },
                hide_index=True,
                use_container_width=True,
            )

            if st.checkbox("Show raw discovery data", value=False):
                st.json(results.get("discovery_data", []))
        else:
            st.info("No discovery details available.")
