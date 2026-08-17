"""Tests for the OpenWeather client (worker/weather.py)."""

import unittest
from unittest import mock

from worker.weather import WeatherService


class WeatherServiceTest(unittest.TestCase):
    def setUp(self):
        self.service = WeatherService(api_key="test-key")

    def test_available_when_key_present(self):
        self.assertTrue(self.service.available)

    def test_unavailable_without_key(self):
        svc = WeatherService(api_key=None)
        svc.api_key = None
        self.assertFalse(svc.available)

    def test_fetch_raises_without_key(self):
        svc = WeatherService(api_key="")
        svc.api_key = None
        with self.assertRaises(ValueError):
            svc.fetch("Pune")

    @mock.patch("worker.weather.requests.get")
    def test_geocode(self, mock_get):
        mock_get.return_value = mock.Mock(
            status_code=200,
            json=lambda: [{"lat": 18.52, "lon": 73.85, "name": "Pune", "country": "IN"}],
            raise_for_status=lambda: None,
        )
        lat, lon, city, country = self.service.geocode("Pune")
        self.assertEqual((lat, lon, city, country), (18.52, 73.85, "Pune", "IN"))

    @mock.patch("worker.weather.requests.get")
    def test_geocode_empty_raises(self, mock_get):
        mock_get.return_value = mock.Mock(
            status_code=200, json=lambda: [], raise_for_status=lambda: None
        )
        with self.assertRaises(ValueError):
            self.service.geocode("Nowhereland")

    @mock.patch("worker.weather.requests.get")
    def test_fetch_returns_weather_dict(self, mock_get):
        geo = {"lat": 18.52, "lon": 73.85, "name": "Pune", "country": "IN"}
        weather = {
            "main": {"temp": 28.5, "feels_like": 30.1, "humidity": 62},
            "weather": [{"description": "clear sky"}],
            "wind": {"speed": 3.4},
        }
        mock_get.return_value = mock.Mock(
            status_code=200, json=lambda: geo, raise_for_status=lambda: None
        )

        def side_effect(*args, **kwargs):
            if args[0].endswith("/geo/1.0/direct"):
                return mock.Mock(status_code=200, json=lambda: [geo], raise_for_status=lambda: None)
            return mock.Mock(status_code=200, json=lambda: weather, raise_for_status=lambda: None)

        mock_get.side_effect = side_effect

        result = self.service.fetch("Pune")
        self.assertEqual(result["city"], "Pune")
        self.assertEqual(result["temperature_c"], 28.5)
        self.assertEqual(result["humidity"], 62)
        self.assertEqual(result["description"], "clear sky")
        self.assertEqual(result["wind_speed_ms"], 3.4)

    @mock.patch("worker.weather.requests.get")
    def test_fetch_http_error_propagates(self, mock_get):
        mock_get.return_value = mock.Mock(
            status_code=401,
            raise_for_status=mock.Mock(side_effect=RuntimeError("Invalid API key")),
        )
        with self.assertRaises(RuntimeError):
            self.service.fetch("Pune")


if __name__ == "__main__":
    unittest.main()