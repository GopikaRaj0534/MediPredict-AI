"""
tests/test_gemini_backend.py
------------------------------
Standalone unit and integration tests specifically for Gemini AI initialization,
key verification, env loading, startup diagnostics, and error handling categories.
"""

import os
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

import config
import chatbot


class GeminiBackendTestCase(unittest.TestCase):

    def test_env_path_resolution(self):
        diag = config.get_diagnostics()
        self.assertTrue(diag["env_path"].endswith(".env"))
        self.assertIn(diag["env_exists"], ["YES", "NO"])

    def test_placeholder_key_filtering(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "YOUR_GEMINI_API_KEY_HERE", "GOOGLE_API_KEY": ""}, clear=True):
            self.assertFalse(config.is_gemini_configured())
            self.assertIsNone(config.get_gemini_api_key())
            reply, error = chatbot.ask_gemini("Hello")
            self.assertIsNone(reply)
            self.assertIsNotNone(error)
            self.assertIn("Gemini API is not configured", error)

    def test_valid_env_key_detection(self):
        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyTestValidKey12345"}, clear=True):
            self.assertTrue(config.is_gemini_configured())
            self.assertEqual(config.get_gemini_api_key(), "AIzaSyTestValidKey12345")

    @patch("requests.post")
    def test_ask_gemini_success(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.json.return_value = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": "Hello! I am live Gemini AI response."}]
                    }
                }
            ]
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyTestValidKey12345"}, clear=True):
            reply, error = chatbot.ask_gemini("Hi there")
            self.assertIsNone(error)
            self.assertEqual(reply, "Hello! I am live Gemini AI response.")

    @patch("requests.post")
    def test_ask_gemini_invalid_key_category(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 400
        mock_resp.json.return_value = {
            "error": {
                "code": 400,
                "message": "API key not valid. Please pass a valid API key.",
                "status": "INVALID_ARGUMENT"
            }
        }
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyInvalidKey"}, clear=True):
            reply, error = chatbot.ask_gemini("Hello")
            self.assertIsNone(reply)
            self.assertIsNotNone(error)
            self.assertIn("Invalid Gemini API key provided", error)

    @patch("requests.post")
    def test_ask_gemini_rate_limit_category(self, mock_post):
        mock_resp = MagicMock()
        mock_resp.status_code = 429
        mock_post.return_value = mock_resp

        with patch.dict(os.environ, {"GEMINI_API_KEY": "AIzaSyTestKey"}, clear=True):
            reply, error = chatbot.ask_gemini("Hello")
            self.assertIsNone(reply)
            self.assertIsNotNone(error)
            self.assertIn("rate limit or quota exceeded", error)


if __name__ == "__main__":
    config.log_startup_diagnostics()
    unittest.main()
