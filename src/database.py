import sqlite3
import os
from datetime import date, timedelta
from typing import Any, Dict, Optional, Tuple
import pandas as pd
from indian_grocery_demo import generate_indian_grocery_demo

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_NAME = os.path.join(PROJECT_ROOT, "data", "dukaan_mind.db")


def get_connection(db_path: str = DB_NAME) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db(db_path: str = DB_NAME) -> None:
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS app_settings (
                id INTEGER PRIMARY KEY CHECK (id = 1),
                gemini_api_key TEXT DEFAULT '',
                gemini_model TEXT DEFAULT 'gemini-3.5-flash-lite',
                shop_name TEXT DEFAULT 'My Shop',
                shop_instructions TEXT DEFAULT '',
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        cur.execute(
            """
            INSERT OR IGNORE INTO app_settings (id, gemini_api_key, gemini_model, shop_name, shop_instructions)
            VALUES (1, '', 'gemini-3.5-flash-lite', 'My Shop', '')
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS inventory (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sku TEXT UNIQUE NOT NULL,
                item_name TEXT NOT NULL,
                category TEXT DEFAULT 'General',
                stock_qty INTEGER NOT NULL DEFAULT 0,
                reorder_level INTEGER NOT NULL DEFAULT 10,
                unit_price REAL NOT NULL DEFAULT 0.0,
                purchase_cost REAL
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS sales (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sale_date TEXT NOT NULL,
                sku TEXT NOT NULL,
                item_name TEXT NOT NULL,
                quantity INTEGER NOT NULL,
                total_amount REAL NOT NULL,
                unit_cost REAL,
                FOREIGN KEY (sku) REFERENCES inventory (sku)
            )
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS dataset_seeds (
                dataset_key TEXT PRIMARY KEY,
                loaded_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )

        inventory_columns = {
            row["name"] for row in cur.execute("PRAGMA table_info(inventory)").fetchall()
        }
        if "purchase_cost" not in inventory_columns:
            cur.execute("ALTER TABLE inventory ADD COLUMN purchase_cost REAL")

        sales_columns = {
            row["name"] for row in cur.execute("PRAGMA table_info(sales)").fetchall()
        }
        if "unit_cost" not in sales_columns:
            cur.execute("ALTER TABLE sales ADD COLUMN unit_cost REAL")

        conn.commit()
    finally:
        conn.close()


def get_settings(db_path: str = DB_NAME) -> Dict[str, Any]:
    conn = get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM app_settings WHERE id = 1").fetchone()
        if row:
            return dict(row)
        return {
            "gemini_api_key": "",
            "gemini_model": "gemini-3.5-flash-lite",
            "shop_name": "My Shop",
            "shop_instructions": "",
        }
    finally:
        conn.close()


def save_settings(
    gemini_api_key: str,
    gemini_model: str,
    shop_name: str,
    shop_instructions: str,
    db_path: str = DB_NAME,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            UPDATE app_settings
            SET gemini_api_key = ?,
                gemini_model = ?,
                shop_name = ?,
                shop_instructions = ?,
                updated_at = CURRENT_TIMESTAMP
            WHERE id = 1
            """,
            (
                gemini_api_key.strip(),
                gemini_model.strip() or "gemini-3.5-flash-lite",
                shop_name.strip() or "My Shop",
                shop_instructions.strip(),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def upsert_inventory_item(
    sku: str,
    item_name: str,
    category: str,
    stock_qty: int,
    reorder_level: int,
    unit_price: float,
    db_path: str = DB_NAME,
    purchase_cost: Optional[float] = None,
) -> None:
    conn = get_connection(db_path)
    try:
        conn.execute(
            """
            INSERT INTO inventory (
                sku, item_name, category, stock_qty, reorder_level, unit_price, purchase_cost
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                item_name = excluded.item_name,
                category = excluded.category,
                stock_qty = excluded.stock_qty,
                reorder_level = excluded.reorder_level,
                unit_price = excluded.unit_price,
                purchase_cost = COALESCE(excluded.purchase_cost, inventory.purchase_cost)
            """,
            (
                sku.strip().upper(),
                item_name.strip(),
                category.strip(),
                stock_qty,
                reorder_level,
                unit_price,
                purchase_cost,
            ),
        )
        conn.commit()
    finally:
        conn.close()


def get_inventory_df(db_path: str = DB_NAME) -> pd.DataFrame:
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM inventory ORDER BY item_name ASC", conn)
    finally:
        conn.close()


def record_sale(
    sale_date: str,
    sku: str,
    quantity: int,
    db_path: str = DB_NAME,
) -> Optional[str]:
    sku_clean = sku.strip().upper()
    conn = get_connection(db_path)
    try:
        cur = conn.cursor()
        item = cur.execute(
            "SELECT item_name, stock_qty, unit_price, purchase_cost FROM inventory WHERE sku = ?",
            (sku_clean,),
        ).fetchone()

        if not item:
            return f"SKU '{sku_clean}' was not found in inventory."
        if quantity <= 0:
            return "Quantity must be greater than zero."
        if item["stock_qty"] < quantity:
            return f"Insufficient stock for {item['item_name']} (available: {item['stock_qty']})."

        total_amount = round(float(item["unit_price"]) * quantity, 2)
        cur.execute(
            """
            INSERT INTO sales (
                sale_date, sku, item_name, quantity, total_amount, unit_cost
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                sale_date,
                sku_clean,
                item["item_name"],
                quantity,
                total_amount,
                item["purchase_cost"],
            ),
        )
        cur.execute(
            "UPDATE inventory SET stock_qty = stock_qty - ? WHERE sku = ?",
            (quantity, sku_clean),
        )
        conn.commit()
        return None
    finally:
        conn.close()


def get_sales_df(db_path: str = DB_NAME) -> pd.DataFrame:
    conn = get_connection(db_path)
    try:
        return pd.read_sql_query("SELECT * FROM sales ORDER BY sale_date ASC, id ASC", conn)
    finally:
        conn.close()


def upgrade_indian_grocery_demo_sales(db_path: str = DB_NAME) -> int:
    init_db(db_path)
    conn = get_connection(db_path)
    try:
        with conn:
            legacy_seed = conn.execute(
                "SELECT loaded_at FROM dataset_seeds WHERE dataset_key = ?",
                ("indian_grocery_180d_v1",),
            ).fetchone()
            upgraded_seed = conn.execute(
                "SELECT 1 FROM dataset_seeds WHERE dataset_key = ?",
                ("indian_grocery_180d_v2",),
            ).fetchone()
            if not legacy_seed or upgraded_seed:
                return 0

            seed_date = date.fromisoformat(str(legacy_seed["loaded_at"])[:10])
            first_date = seed_date - timedelta(days=180)
            _, old_sales = generate_indian_grocery_demo(
                days=180,
                seed=2026,
                start_date=first_date,
                weekly_variation=False,
            )
            _, updated_sales = generate_indian_grocery_demo(
                days=180,
                seed=2026,
                start_date=first_date,
                weekly_variation=True,
            )
            if len(old_sales) != len(updated_sales):
                raise RuntimeError("Demo history versions produced different sale counts.")

            def sale_signature(sale: Tuple[Any, ...]) -> Tuple[Any, ...]:
                return (
                    sale[0],
                    sale[1],
                    sale[2],
                    int(sale[3]),
                    round(float(sale[4]), 2),
                    round(float(sale[5]), 2) if sale[5] is not None else None,
                )

            existing_sales = conn.execute(
                """
                SELECT id, sale_date, sku, item_name, quantity, total_amount, unit_cost
                FROM sales
                WHERE sku LIKE 'DG-%' AND sale_date >= ? AND sale_date < ?
                ORDER BY id
                """,
                (first_date.isoformat(), seed_date.isoformat()),
            ).fetchall()
            matching_ids = {}
            for sale in existing_sales:
                matching_ids.setdefault(sale_signature(sale[1:]), []).append(sale["id"])

            updates = []
            for old_sale, updated_sale in zip(old_sales, updated_sales):
                matches = matching_ids.get(sale_signature(old_sale))
                if matches:
                    sale_id = matches.pop()
                    updates.append(
                        (
                            updated_sale[3],
                            updated_sale[4],
                            updated_sale[5],
                            sale_id,
                        )
                    )

            if not updates:
                return 0

            conn.executemany(
                """
                UPDATE sales
                SET quantity = ?, total_amount = ?, unit_cost = ?
                WHERE id = ?
                """,
                updates,
            )
            conn.execute(
                "INSERT OR IGNORE INTO dataset_seeds (dataset_key) VALUES (?)",
                ("indian_grocery_180d_v2",),
            )
            return len(updates)
    finally:
        conn.close()


def seed_indian_grocery_demo_data(db_path: str = DB_NAME) -> Tuple[int, int]:
    init_db(db_path)
    upgrade_indian_grocery_demo_sales(db_path)
    products, sales = generate_indian_grocery_demo()
    conn = get_connection(db_path)
    try:
        with conn:
            already_loaded = conn.execute(
                "SELECT 1 FROM dataset_seeds WHERE dataset_key IN (?, ?)",
                ("indian_grocery_180d_v1", "indian_grocery_180d_v2"),
            ).fetchone()
            if already_loaded:
                return 0, 0

            existing_skus = {
                row["sku"] for row in conn.execute("SELECT sku FROM inventory").fetchall()
            }
            new_products = [product for product in products if product[0] not in existing_skus]
            new_skus = {product[0] for product in new_products}
            new_sales = [sale for sale in sales if sale[1] in new_skus]

            conn.executemany(
                """
                INSERT INTO inventory (
                    sku, item_name, category, stock_qty, reorder_level, unit_price, purchase_cost
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                new_products,
            )
            conn.executemany(
                """
                INSERT INTO sales (
                    sale_date, sku, item_name, quantity, total_amount, unit_cost
                )
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                new_sales,
            )
            conn.execute(
                "INSERT INTO dataset_seeds (dataset_key) VALUES (?)",
                ("indian_grocery_180d_v2",),
            )
        return len(new_products), len(new_sales)
    finally:
        conn.close()


def seed_sample_data(db_path: str = DB_NAME) -> None:
    upgrade_indian_grocery_demo_sales(db_path)
    conn = get_connection(db_path)
    try:
        existing_inv = conn.execute("SELECT COUNT(*) AS cnt FROM inventory").fetchone()["cnt"]
    finally:
        conn.close()

    if existing_inv == 0:
        seed_indian_grocery_demo_data(db_path)