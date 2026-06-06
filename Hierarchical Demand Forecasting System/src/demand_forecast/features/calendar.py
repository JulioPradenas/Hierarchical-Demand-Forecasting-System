from __future__ import annotations

import numpy as np
import pandas as pd

from .base import FeatureBuilder

M5_EVENTS: list[str] = [
    "SuperBowl",
    "ValentinesDay",
    "PresidentsDay",
    "StPatricksDay",
    "Easter",
    "Cinco_De_Mayo",
    "MotherDay",
    "MemorialDay",
    "NBAFinalsStart",
    "NBAFinalsEnd",
    "FathersDay",
    "IndependenceDay",
    "LaborDay",
    "ColumbusDay",
    "Halloween",
    "EidAlAdha",
    "VeteransDay",
    "Thanksgiving",
    "Christmas",
    "NewYear",
    "OrthodoxChristmas",
    "MartinLutherKingDay",
]


class CalendarFeatureBuilder(FeatureBuilder):
    """Generates calendar and event features from the M5 calendar DataFrame.

    Events encoded as continuous distance features (days until/since),
    not binary flags, to preserve proximity information for GBDT.
    """

    def __init__(self, events: list[str] | None = None, horizon: int = 28) -> None:
        self.events = events or M5_EVENTS
        self.horizon = horizon

    def fit(self, df: pd.DataFrame) -> CalendarFeatureBuilder:
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        cal = df.copy()
        cal["date"] = pd.to_datetime(cal["date"])
        cal = cal.sort_values("date").reset_index(drop=True)

        cal["day_of_week"] = cal["date"].dt.dayofweek
        cal["day_of_month"] = cal["date"].dt.day
        cal["week_of_year"] = cal["date"].dt.isocalendar().week.astype(int)
        cal["month"] = cal["date"].dt.month
        cal["year"] = cal["date"].dt.year
        cal["is_weekend"] = (cal["day_of_week"] >= 5).astype(int)

        event_col = "event_name_1"
        if event_col in cal.columns:
            dates_arr = cal["date"].to_numpy()
            sentinel = float(self.horizon * 2)
            for event in self.events:
                event_mask = cal[event_col] == event
                event_dates = cal.loc[event_mask, "date"].to_numpy()
                if len(event_dates) == 0:
                    cal[f"days_until_{event}"] = sentinel
                    cal[f"days_since_{event}"] = sentinel
                    continue
                days_until = np.full(len(cal), sentinel)
                days_since = np.full(len(cal), sentinel)
                for i, d in enumerate(dates_arr):
                    future = event_dates[event_dates >= d]
                    past = event_dates[event_dates <= d]
                    if len(future) > 0:
                        days_until[i] = float((future[0] - d) / np.timedelta64(1, "D"))
                    if len(past) > 0:
                        days_since[i] = float((d - past[-1]) / np.timedelta64(1, "D"))
                cal[f"days_until_{event}"] = days_until
                cal[f"days_since_{event}"] = days_since

        return cal
