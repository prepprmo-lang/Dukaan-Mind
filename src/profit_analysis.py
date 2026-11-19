import pandas as pd


def summarize_sku_profit(sales_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "sku",
        "item_name",
        "units_sold",
        "costed_units",
        "uncosted_units",
        "known_revenue",
        "purchase_cost",
        "gross_profit",
    ]
    if sales_df.empty:
        return pd.DataFrame(columns=columns)

    sales = sales_df.copy()
    if "unit_cost" not in sales.columns:
        sales["unit_cost"] = pd.NA

    has_cost = sales["unit_cost"].notna()
    sales["costed_units"] = sales["quantity"].where(has_cost, 0)
    sales["uncosted_units"] = sales["quantity"].where(~has_cost, 0)
    sales["known_revenue"] = sales["total_amount"].where(has_cost, 0.0)
    sales["purchase_cost"] = (sales["quantity"] * sales["unit_cost"]).where(has_cost, 0.0)
    sales["gross_profit"] = (sales["total_amount"] - sales["purchase_cost"]).where(
        has_cost, 0.0
    )

    summary = (
        sales.groupby(["sku", "item_name"], as_index=False)
        .agg(
            units_sold=("quantity", "sum"),
            costed_units=("costed_units", "sum"),
            uncosted_units=("uncosted_units", "sum"),
            known_revenue=("known_revenue", "sum"),
            purchase_cost=("purchase_cost", "sum"),
            gross_profit=("gross_profit", "sum"),
        )
    )
    summary.loc[summary["costed_units"] == 0, "gross_profit"] = pd.NA
    return summary[columns]
