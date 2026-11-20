import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from database import (
    get_inventory_df,
    get_sales_df,
    get_settings,
    init_db,
    record_sale,
    reset_to_indian_grocery_demo_data,
    save_settings,
    upsert_inventory_item,
)
from indian_grocery_demo import generate_indian_grocery_demo


class DatabaseResetTests(unittest.TestCase):
    def test_reset_restores_demo_data_and_preserves_shop_settings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = str(Path(temp_dir) / "shop.db")
            init_db(db_path)
            save_settings("test-key", "test-model", "My Test Shop", "Keep these", db_path)
            upsert_inventory_item(
                "CUSTOM-1",
                "Test Item",
                "Test",
                10,
                2,
                5.0,
                db_path,
                purchase_cost=2.0,
            )
            self.assertIsNone(record_sale("2026-01-01", "CUSTOM-1", 1, db_path))

            expected_products, expected_sales = generate_indian_grocery_demo()
            counts = reset_to_indian_grocery_demo_data(db_path)
            inventory = get_inventory_df(db_path)
            sales = get_sales_df(db_path)
            settings = get_settings(db_path)

            self.assertEqual(counts, (len(expected_products), len(expected_sales)))
            self.assertEqual(set(inventory["sku"]), {product[0] for product in expected_products})
            self.assertEqual(len(sales), len(expected_sales))
            self.assertNotIn("CUSTOM-1", set(inventory["sku"]))
            self.assertEqual(settings["gemini_api_key"], "test-key")
            self.assertEqual(settings["shop_name"], "My Test Shop")
            self.assertEqual(settings["shop_instructions"], "Keep these")


if __name__ == "__main__":
    unittest.main()
