"""Transport-to-anywhere enumerator (IWM Decision C, dataset variant).

For the C dataset we reproduce the paper's "exhaustive" transportation: at each
travel-day state, enumerate EVERY mode (flight / self-drive / taxi) to EVERY
reachable destination from the leg's origin city, using the GLOBAL database
(not just the gold-route options in ref_info). Stay-days remain SKIP-only,
identical to the A variant. All non-transportation fields are unchanged from A
(still ref_info via admissible.enumerate_admissible).

Cost formulas are exactly the submodule's
tools/googleDistanceMatrix/apis.py:run_for_evaluation:
    self-driving base = int(distance_km * 0.05)
    taxi        base = int(distance_km)
then × ceil(people/5) for self-drive, × ceil(people/4) for taxi (per
hard_constraint.get_total_cost). Multi-day drives ('day' in duration) are
invalid (apis.py returns "No valid information" for them) and excluded.

The global DB lives outside the submodule at envs/travelplanner/global_db/
(gitignored; reproduce via the gdown command in NOTES.md).
"""
from __future__ import annotations
import math
import re
from pathlib import Path

import pandas as pd

from admissible import _leg, norm_city, _skip

_DIST_RE = re.compile(r"([\d.,]+)\s*km")


def _parse_km(dist_str: str) -> float | None:
    m = _DIST_RE.search(str(dist_str))
    if not m:
        return None
    return float(m.group(1).replace(",", ""))


def load_global_db(db_root: Path, needed_origin_dates: set, needed_origins: set) -> dict:
    """Build filtered indices so we don't hold the 305 MB flights CSV in full.

    Returns {"flights": {(origin, date): [flight_dict, ...]},
             "distance": {origin: [{destination, dist_str, km, duration}, ...]}}
    """
    db_root = Path(db_root)

    # --- flights: stream-filter to the (origin, date) pairs we actually need ---
    flights_idx: dict = {}
    cols = ["Flight Number", "Price", "DepTime", "ArrTime",
            "OriginCityName", "DestCityName", "FlightDate"]
    needed_origins_set = set(needed_origins)
    for chunk in pd.read_csv(db_root / "flights" / "clean_Flights_2022.csv",
                             usecols=cols, chunksize=500_000):
        chunk = chunk[chunk["OriginCityName"].isin(needed_origins_set)]
        if chunk.empty:
            continue
        chunk = chunk.rename(columns={"Flight Number": "FlightNumber"})
        for row in chunk.itertuples(index=False):
            key = (row.OriginCityName, row.FlightDate)
            if key not in needed_origin_dates:
                continue
            flights_idx.setdefault(key, []).append({
                "Flight Number": row.FlightNumber,
                "Price": row.Price, "DepTime": row.DepTime, "ArrTime": row.ArrTime,
                "OriginCityName": row.OriginCityName, "DestCityName": row.DestCityName,
            })

    # --- distance matrix: origin -> reachable destinations (valid, non-multi-day) ---
    dm = pd.read_csv(db_root / "googleDistanceMatrix" / "distance.csv")
    distance_idx: dict = {}
    for row in dm.itertuples(index=False):
        origin = row.origin
        if origin not in needed_origins_set:
            continue
        dur = row.duration
        dist = row.distance
        if pd.isna(dur) or pd.isna(dist):
            continue
        if "day" in str(dur):  # multi-day drive → invalid (matches apis.py)
            continue
        km = _parse_km(dist)
        if km is None:
            continue
        distance_idx.setdefault(origin, []).append({
            "destination": row.destination, "dist_str": str(dist),
            "km": km, "duration": str(dur)})

    return {"flights": flights_idx, "distance": distance_idx}


def enumerate_transportation_anywhere(day_idx: int, day_dict: dict,
                                      query: dict, gdb: dict) -> list[dict]:
    """All transportation actions to ANY reachable destination on a travel day.
    Stay-days → SKIP only (same as A)."""
    people = int(query["people_number"])
    day_num = day_idx + 1
    src, dst = _leg(day_dict)
    actions: list[dict] = []

    if src and dst and src != dst:
        dates = query.get("date") or []
        date = dates[day_idx] if day_idx < len(dates) else None

        # flights to anywhere from src on this date
        for fl in gdb["flights"].get((src, date), []):
            value = (f"Flight Number: {fl['Flight Number']}, "
                     f"from {fl['OriginCityName']} to {fl['DestCityName']}, "
                     f"Departure Time: {fl['DepTime']}, Arrival Time: {fl['ArrTime']}")
            actions.append({"action_type": "SET_TRANSPORTATION", "day": day_num,
                            "field": "transportation", "value": value,
                            "cost": float(fl["Price"]) * people})

        # self-drive + taxi to every reachable destination from src
        for d in gdb["distance"].get(src, []):
            dest = d["destination"]
            base_drive = int(d["km"] * 0.05)
            base_taxi = int(d["km"])
            actions.append({"action_type": "SET_TRANSPORTATION", "day": day_num,
                            "field": "transportation",
                            "value": (f"Self-driving, from {src} to {dest}, "
                                      f"duration: {d['duration']}, distance: {d['dist_str']}, "
                                      f"cost: {base_drive}"),
                            "cost": base_drive * math.ceil(people / 5)})
            actions.append({"action_type": "SET_TRANSPORTATION", "day": day_num,
                            "field": "transportation",
                            "value": (f"Taxi, from {src} to {dest}, "
                                      f"duration: {d['duration']}, distance: {d['dist_str']}, "
                                      f"cost: {base_taxi}"),
                            "cost": base_taxi * math.ceil(people / 4)})

    actions.append(_skip("transportation", day_num))
    return actions
