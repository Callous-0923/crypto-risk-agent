import unittest
from unittest.mock import patch

import src.core.proxy as proxy_mod


class ProxyConfigTests(unittest.TestCase):
    def test_http_proxy_prefers_all_proxy(self):
        with patch.object(proxy_mod.settings, "all_proxy", "socks5://all"), \
             patch.object(proxy_mod.settings, "https_proxy", "socks5://https"), \
             patch.object(proxy_mod.settings, "http_proxy", "socks5://http"), \
             patch.object(proxy_mod.settings, "ws_proxy", "socks5://ws"):
            self.assertEqual(proxy_mod.get_http_proxy(), "socks5://all")

    def test_ws_proxy_falls_back_to_http_proxy(self):
        with patch.object(proxy_mod.settings, "all_proxy", "socks5://all"), \
             patch.object(proxy_mod.settings, "https_proxy", ""), \
             patch.object(proxy_mod.settings, "http_proxy", ""), \
             patch.object(proxy_mod.settings, "ws_proxy", ""):
            self.assertEqual(proxy_mod.get_ws_proxy(), "socks5://all")

    def test_httpx_client_kwargs_include_proxy(self):
        with patch.object(proxy_mod.settings, "all_proxy", ""), \
             patch.object(proxy_mod.settings, "https_proxy", ""), \
             patch.object(proxy_mod.settings, "http_proxy", "socks5://http"), \
             patch.object(proxy_mod.settings, "ws_proxy", ""):
            kwargs = proxy_mod.get_httpx_client_kwargs(service="test")
        self.assertEqual(kwargs["proxy"], "socks5://http")

    def test_openai_client_kwargs_include_http_client_when_proxy_enabled(self):
        with patch.object(proxy_mod.settings, "ark_api_key", "test-key"), \
             patch.object(proxy_mod.settings, "ark_base_url", "https://example.com"), \
             patch.object(proxy_mod.settings, "all_proxy", "socks5://all"), \
             patch.object(proxy_mod.settings, "https_proxy", ""), \
             patch.object(proxy_mod.settings, "http_proxy", ""), \
             patch.object(proxy_mod.settings, "ws_proxy", ""), \
             patch.object(proxy_mod, "DefaultHttpxClient", return_value=object()) as mock_client:
            kwargs = proxy_mod.get_openai_client_kwargs(service="test")
        self.assertEqual(kwargs["api_key"], "test-key")
        self.assertEqual(kwargs["base_url"], "https://example.com")
        self.assertIn("http_client", kwargs)
        mock_client.assert_called_once_with(proxy="socks5://all")


if __name__ == "__main__":
    unittest.main()
