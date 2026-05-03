from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas import BacktestSummary, PortfolioSimulation
from app.services.backtest_service import get_backtest_summary, get_portfolio_simulation, run_backtest

router = APIRouter(prefix="/backtest", tags=["backtest"])


@router.post("/run", response_model=BacktestSummary)
def run_signal_backtest(db: Session = Depends(get_db), limit: int = 100) -> dict:
    return run_backtest(db, limit=limit)


@router.get("/summary", response_model=BacktestSummary)
def read_backtest_summary(db: Session = Depends(get_db)) -> dict:
    return get_backtest_summary(db)


@router.get("/portfolio", response_model=PortfolioSimulation)
def read_portfolio_simulation(db: Session = Depends(get_db)) -> dict:
    return get_portfolio_simulation(db)
