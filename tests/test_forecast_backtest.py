import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from app import compute_demand_forecast, evaluate_demand_forecast


class EvaluateDemandForecastTests(unittest.TestCase):
    def make_sales(self, days=56):
        dates = pd.date_range("2026-01-01", periods=days, freq="D")
        quantities = [4 if day.weekday() < 5 else 9 for day in dates]
        return pd.DataFrame(
            {
                "sale_date": dates.strftime("%Y-%m-%d"),
                "quantity": quantities,
                "total_amount": [quantity * 10 for quantity in quantities],
                "sku": "TEST",
            }
        )

    def test_backtest_compares_model_and_weekly_baseline_on_holdout_days(self):
        result = evaluate_demand_forecast(self.make_sales(), horizon_days=7)

        self.assertTrue(result["ready"])
        self.assertEqual(result["windows"], 4)
        self.assertEqual(result["evaluated_days"], 28)
        self.assertIsNotNone(result["model_wape"])
        self.assertEqual(result["baseline_wape"], 0)

    def test_backtest_anchors_missing_cutoff_dates_without_future_sales(self):
        sales = self.make_sales()
        cutoff_indices = [27, 34, 41, 48]
        sales = sales.drop(index=cutoff_indices).reset_index(drop=True)

        with patch("app.compute_demand_forecast", wraps=compute_demand_forecast) as forecast:
            result = evaluate_demand_forecast(sales, horizon_days=7)

        self.assertTrue(result["ready"])
        self.assertEqual(forecast.call_count, 4)
        for call, cutoff_index in zip(forecast.call_args_list, cutoff_indices):
            training_sales = call.kwargs["sales_df"]
            training_dates = pd.to_datetime(training_sales["sale_date"])
            self.assertEqual(training_dates.max(), pd.Timestamp("2026-01-01") + pd.Timedelta(days=cutoff_index))

    def test_backtest_reports_when_history_is_too_short(self):
        result = evaluate_demand_forecast(self.make_sales(days=20), horizon_days=7)

        self.assertFalse(result["ready"])
        self.assertIn("at least 21 days", result["reason"])


if __name__ == "__main__":
    unittest.main()
