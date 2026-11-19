import os
import re
import time
from typing import Any, Dict, Optional
import pandas as pd
from google import genai
from google.genai import errors, types

from indian_formatting import format_indian_number, format_inr
from profit_analysis import summarize_sku_profit


_CLIENTS = {}
_READ_TIMEOUT_ERRORS = tuple(
    exception_type
    for backend in (errors.httpx, errors.httpx2)
    if backend is not None
    for exception_type in (getattr(backend, "ReadTimeout", None),)
    if exception_type is not None
)
_CONNECT_ERRORS = tuple(
    exception_type
    for backend in (errors.httpx, errors.httpx2)
    if backend is not None
    for exception_type in (getattr(backend, "ConnectError", None),)
    if exception_type is not None
)
_RETRYABLE_ERRORS = (errors.APIError,) + _READ_TIMEOUT_ERRORS + _CONNECT_ERRORS


DATA_KEYWORDS_PATTERN = re.compile(
    r"\b("
    r"stock|inventory|sale|sales|revenue|forecast|predict|demand|order|restock|"
    r"reorder|sku|price|margin|profit|loss|slow|fast|mover|unit|units|"
    r"trend|low|shortage|surplus|cover|dead\s*stock"
    r")\b",
    re.IGNORECASE,
)
CHART_REQUEST_PATTERN = re.compile(
    r"\b(chart|graph|plot|visuali[sz](?:e|ation))\b",
    re.IGNORECASE,
)


def needs_store_data(prompt: str) -> bool:
    return bool(DATA_KEYWORDS_PATTERN.search(prompt or ""))


def requests_chart(prompt: str) -> bool:
    return bool(CHART_REQUEST_PATTERN.search(prompt or ""))


def requested_chart_type(prompt: str) -> Optional[str]:
    if not requests_chart(prompt):
        return None

    text = (prompt or "").lower()
    if any(word in text for word in ("profit", "margin")):
        return "profit"
    if any(word in text for word in ("forecast", "demand", "predict")):
        return "forecast"
    if any(word in text for word in ("inventory", "stock", "reorder")):
        return "inventory"
    if any(word in text for word in ("sku", "product", "item")) and any(
        word in text for word in ("sales", "revenue")
    ):
        return "sku_sales"
    return "weekly_sales"


def build_business_context(
    settings: Dict[str, Any],
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    graph_type: Optional[str] = None,
    include_profit_data: bool = False,
) -> str:
    shop_name = settings.get("shop_name") or os.environ.get("DUKAAN_SHOP_NAME", "My Shop")
    total_revenue = float(sales_df["total_amount"].sum()) if not sales_df.empty else 0.0
    total_units_sold = int(sales_df["quantity"].sum()) if not sales_df.empty else 0

    low_stock = pd.DataFrame()
    if not inventory_df.empty:
        low_stock = inventory_df[inventory_df["stock_qty"] <= inventory_df["reorder_level"]]

    low_stock_lines = (
        "\n".join(
            f"- {r['item_name']} ({r['sku']}): {r['stock_qty']} left (reorder at {r['reorder_level']})"
            for _, r in low_stock.iterrows()
        )
        if not low_stock.empty
        else "- None"
    )

    forecast_summary = (
        "\n".join(
            f"- {r['forecast_date']}: predicted {r['predicted_units']:.1f} units "
            f"(est. revenue: {format_inr(r['predicted_revenue'])})"
            for _, r in forecast_df.iterrows()
        )
        if not forecast_df.empty
        else "- Insufficient sales history for forecast"
    )

    profit_section = ""
    if include_profit_data:
        sku_profit = summarize_sku_profit(sales_df)
        sku_profit = (
            sku_profit.sort_values("gross_profit", ascending=False, na_position="last")
            .head(14)
        )
        profit_scope = "top 14 SKUs by gross profit"
        if sku_profit.empty:
            profit_lines = "- No sales recorded"
        else:
            profit_lines = "\n".join(
                (
                    f"- {row['sku']} ({row['item_name']}): gross profit "
                    f"{format_inr(row['gross_profit'])} on "
                    f"{format_indian_number(int(row['costed_units']))} costed units"
                    + (
                        f"; {format_indian_number(int(row['uncosted_units']))} "
                        "units have no recorded cost"
                        if row["uncosted_units"]
                        else ""
                    )
                    if row["costed_units"] > 0
                    else (
                        f"- {row['sku']} ({row['item_name']}): unavailable; "
                        f"purchase cost is missing for "
                        f"{format_indian_number(int(row['uncosted_units']))} units"
                    )
                )
                for _, row in sku_profit.iterrows()
            )
        profit_section = f"""

Gross Profit by SKU ({profit_scope}; excludes overhead; only sales with recorded purchase cost are calculated):
{profit_lines}"""

    graph_data = ""
    if graph_type:
        graph_sections = []
        if graph_type == "weekly_sales":
            weekly_sales_lines = "- No sales recorded"
            if not sales_df.empty:
                sales = sales_df.copy()
                sales["sale_date"] = pd.to_datetime(sales["sale_date"], errors="coerce")
                sales = sales.dropna(subset=["sale_date"])
                sales["week_start"] = sales["sale_date"] - pd.to_timedelta(
                    sales["sale_date"].dt.day_of_week, unit="D"
                )
                weekly_sales = (
                    sales.groupby("week_start", as_index=False)
                    .agg(revenue=("total_amount", "sum"), units=("quantity", "sum"))
                    .sort_values("week_start")
                    .tail(26)
                )
                weekly_sales_lines = "\n".join(
                    f"- Week of {row['week_start']:%Y-%m-%d}: "
                    f"revenue {format_inr(row['revenue'])}, "
                    f"{format_indian_number(int(row['units']))} units"
                    for _, row in weekly_sales.iterrows()
                ) or "- No dated sales recorded"
            graph_sections.append(f"Recent Weekly Sales for Graphs:\n{weekly_sales_lines}")
        elif graph_type == "sku_sales":
            sku_sales_lines = "- No sales recorded"
            if not sales_df.empty:
                sku_sales = (
                    sales_df.groupby(["sku", "item_name"], as_index=False)
                    .agg(revenue=("total_amount", "sum"), units=("quantity", "sum"))
                    .sort_values("revenue", ascending=False)
                    .head(14)
                )
                sku_sales_lines = "\n".join(
                    f"- {row['sku']} ({row['item_name']}): "
                    f"revenue {format_inr(row['revenue'])}, "
                    f"{format_indian_number(int(row['units']))} units"
                    for _, row in sku_sales.iterrows()
                )
            graph_sections.append(f"Sales by SKU for Graphs (top 14 by revenue):\n{sku_sales_lines}")
        elif graph_type == "inventory":
            if inventory_df.empty:
                inventory_lines = "- No inventory recorded"
            else:
                closest_stock = (
                    inventory_df.assign(
                        stock_gap=inventory_df["stock_qty"] - inventory_df["reorder_level"]
                    )
                    .sort_values(["stock_gap", "stock_qty"], ascending=True)
                    .head(14)
                )
                inventory_lines = "\n".join(
                    f"- {row['sku']} ({row['item_name']}): {int(row['stock_qty'])} in stock, "
                    f"reorder at {int(row['reorder_level'])}"
                    for _, row in closest_stock.iterrows()
                )
            graph_sections.append(
                f"Inventory by SKU for Graphs (14 nearest reorder marks):\n{inventory_lines}"
            )

        graph_data = "\n\n" + "\n\n".join(graph_sections) if graph_sections else ""

    return f"""
Store Name: {shop_name}
Total Recorded Revenue: {format_inr(total_revenue)}
Total Units Sold: {format_indian_number(total_units_sold)}
Total Inventory SKUs: {len(inventory_df)}

Low-Stock Items:
{low_stock_lines}

Upcoming Demand Forecast:
{forecast_summary}

{profit_section}
{graph_data}
""".strip()


def generate_shop_advice_stream(
    prompt: str,
    settings: Dict[str, Any],
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
):
    clean_prompt = (prompt or "").strip()
    if not clean_prompt:
        yield "Please enter a message."
        return

    api_key = (
        os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GOOGLE_API_KEY")
        or (settings.get("gemini_api_key") if settings else "")
        or ""
    ).strip()

    if not api_key:
        yield (
            "Assistant configuration is not available on this device. "
            "Set GEMINI_API_KEY in the server environment to enable live responses."
        )
        return

    model_name = (
        os.environ.get("GEMINI_MODEL")
        or (settings.get("gemini_model") if settings else "")
        or "gemini-3.5-flash-lite"
    ).strip()
    if model_name.startswith("models/"):
        model_name = model_name.removeprefix("models/")

    custom_instructions = (
        os.environ.get("DUKAAN_SHOP_INSTRUCTIONS")
        or (settings.get("shop_instructions") if settings else "")
        or ""
    ).strip()

    graph_type = requested_chart_type(clean_prompt)
    profit_requested = bool(re.search(r"\b(profit|margin|loss)\b", clean_prompt, re.IGNORECASE))
    graph_requested = graph_type is not None
    include_data = needs_store_data(clean_prompt) or graph_requested
    if include_data:
        data_rule = (
            "DATA ANALYSIS REQUIRED: Store sales, inventory, and forecast data are provided below. "
            "Use those numbers directly to give concise, practical recommendations. "
            "Format all money in Indian rupees (₹) using Indian digit grouping, such as ₹24,22,527.00."
        )
        if profit_requested:
            data_rule += (
                " Only report gross profit for sales with recorded purchase cost; mark missing-cost sales as unavailable, "
                "and do not estimate unknown costs."
            )
        if graph_requested:
            data_rule += " Never invent or estimate missing chart values; summarize only the supplied data."
    else:
        data_rule = (
            "DATA ANALYSIS NOT REQUIRED: No store data analysis is needed for this message. "
            "Answer briefly and directly without pulling in unrequested sales or stock metrics."
        )

    system_parts = [data_rule]
    if custom_instructions:
        system_parts.append(f"Shop-Specific Instructions:\n{custom_instructions}")
    system_instruction = "\n\n".join(system_parts)

    if include_data:
        context_block = build_business_context(
            settings,
            inventory_df,
            sales_df,
            forecast_df,
            graph_type=graph_type,
            include_profit_data=profit_requested,
        )
        full_prompt = f"{context_block}\n\nUser Message:\n{clean_prompt}"
    else:
        full_prompt = f"User Message:\n{clean_prompt}"

    client = _CLIENTS.get(api_key)
    if client is None:
        client = genai.Client(
            api_key=api_key,
            http_options=types.HttpOptions(timeout=180000),
        )
        _CLIENTS[api_key] = client

    if model_name.startswith("gemini-2.5-"):
        thinking_config = types.ThinkingConfig(thinking_budget=0)
    elif model_name == "gemini-3.5-flash-lite":
        thinking_config = types.ThinkingConfig(
            thinking_level=types.ThinkingLevel.MINIMAL
        )
    else:
        thinking_config = types.ThinkingConfig(thinking_level=types.ThinkingLevel.LOW)
    for attempt in range(3):
        has_text = False
        try:
            response = client.models.generate_content_stream(
                model=model_name,
                contents=full_prompt,
                config=types.GenerateContentConfig(
                    system_instruction=system_instruction,
                    max_output_tokens=384,
                    thinking_config=thinking_config,
                ),
            )
            for chunk in response:
                text = chunk.text
                if text:
                    has_text = True
                    yield text
            if not has_text:
                yield "Gemini returned an empty response. Check the selected model and API configuration, then try again."
            return
        except _RETRYABLE_ERRORS as exc:
            is_service_unavailable = isinstance(exc, errors.APIError) and exc.code == 503
            is_read_timeout = isinstance(exc, _READ_TIMEOUT_ERRORS)
            is_connection_error = isinstance(exc, _CONNECT_ERRORS)
            max_attempts = 2 if is_read_timeout or is_connection_error else 3
            if (
                not (is_service_unavailable or is_read_timeout or is_connection_error)
                or has_text
                or attempt + 1 >= max_attempts
            ):
                if is_read_timeout:
                    raise TimeoutError(
                        "The advisor took too long to respond. It may be busy; please try again in a moment."
                    ) from exc
                if is_connection_error:
                    raise ConnectionError(
                        "Could not reach the advisor service. Check your internet or DNS connection, then try again."
                    ) from exc
                raise
            time.sleep(attempt + 1)


def generate_shop_advice(
    prompt: str,
    settings: Dict[str, Any],
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
) -> str:
    response = "".join(
        generate_shop_advice_stream(
            prompt,
            settings,
            inventory_df,
            sales_df,
            forecast_df,
        )
    )
    return response or "No response returned by Gemini."