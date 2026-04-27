from datetime import datetime, timezone

import pandas as pd
from fastapi.testclient import TestClient

from app.main import app


def test_market_history_endpoint_returns_ohlcv_points(monkeypatch) -> None:
    index = pd.to_datetime(
        [
            datetime(2026, 4, 27, 9, 30, tzinfo=timezone.utc),
            datetime(2026, 4, 27, 9, 35, tzinfo=timezone.utc),
        ]
    )
    frame = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [102.0, 103.0],
            "Low": [99.0, 100.5],
            "Close": [101.0, 102.5],
            "Volume": [1000, 1500],
        },
        index=index,
    )

    class FakeTicker:
        fast_info = {"currency": "USD", "previous_close": 99.5}

        def __init__(self, symbol: str) -> None:
            self.symbol = symbol

        def history(self, period: str, interval: str, auto_adjust: bool = False) -> pd.DataFrame:
            assert period == "1d"
            assert interval == "5m"
            assert auto_adjust is False
            return frame

    monkeypatch.setattr("app.services.market_service.yf.Ticker", FakeTicker)

    response = TestClient(app).get("/market/history/spy?range=1d")

    assert response.status_code == 200
    payload = response.json()
    assert payload["symbol"] == "SPY"
    assert payload["range"] == "1d"
    assert payload["interval"] == "5m"
    assert payload["previous_close"] == 99.5
    assert payload["high"] == 103.0
    assert payload["low"] == 99.0
    assert payload["volume"] == 2500
    assert len(payload["points"]) == 2
    assert payload["points"][0]["close"] == 101.0
