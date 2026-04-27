from fastapi import APIRouter, HTTPException

from app.schemas import MarketData, MarketHistory
from app.services.market_service import fetch_history, fetch_price

router = APIRouter(tags=["market"])


@router.get("/price/{symbol}", response_model=MarketData)
def get_price(symbol: str) -> dict[str, object]:
    try:
        return fetch_price(symbol.upper())
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch market data") from exc


@router.get("/market/history/{symbol}", response_model=MarketHistory)
def get_history(symbol: str, range: str = "1d") -> dict[str, object]:
    try:
        return fetch_history(symbol.upper(), range)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to fetch historical market data") from exc
