import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pandas as pd
from google.genai import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import gemini_assistant
from gemini_assistant import generate_shop_advice_stream


class GeminiModelSelectionTests(unittest.TestCase):
    def test_default_model_uses_flash_lite_with_minimal_thinking(self):
        api_key = "gemini-model-selection-test-key"
        gemini_assistant._CLIENTS.pop(api_key, None)
        client = Mock()
        client.models.generate_content_stream.return_value = [
            SimpleNamespace(text="Ready.")
        ]
        empty = pd.DataFrame()

        with patch("gemini_assistant.genai.Client", return_value=client):
            response = list(
                generate_shop_advice_stream(
                    "Tell me a short greeting.",
                    {"gemini_api_key": api_key},
                    empty,
                    empty,
                    empty,
                )
            )

        request = client.models.generate_content_stream.call_args.kwargs
        self.assertEqual(request["model"], "gemini-3.5-flash-lite")
        self.assertEqual(
            request["config"].thinking_config.thinking_level,
            types.ThinkingLevel.MINIMAL,
        )
        self.assertEqual(response, ["Ready."])
        gemini_assistant._CLIENTS.pop(api_key, None)

    def test_shop_context_is_always_sent_and_access_scope_is_clear(self):
        api_key = "gemini-store-access-test-key"
        gemini_assistant._CLIENTS.pop(api_key, None)
        client = Mock()
        client.models.generate_content_stream.return_value = [
            SimpleNamespace(text="I can use the shop information provided to me.")
        ]
        empty = pd.DataFrame()

        with patch("gemini_assistant.genai.Client", return_value=client):
            list(
                generate_shop_advice_stream(
                    "How can you help me?",
                    {"gemini_api_key": api_key, "shop_name": "Mamata's Shop"},
                    empty,
                    empty,
                    empty,
                )
            )

        request = client.models.generate_content_stream.call_args.kwargs
        self.assertIn("Store Name: Mamata's Shop", request["contents"])
        self.assertIn("SHOP DATA ACCESS", request["config"].system_instruction)
        self.assertIn("never claim otherwise", request["config"].system_instruction)
        gemini_assistant._CLIENTS.pop(api_key, None)


if __name__ == "__main__":
    unittest.main()
