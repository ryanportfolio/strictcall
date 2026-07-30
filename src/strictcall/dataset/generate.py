"""Deterministic synthetic loyalty-program dataset.

Nothing is checked into git: the same seed rebuilds the same database
byte-for-byte anywhere. A handful of members are deliberately placed within
500 points of their next tier so questions like "who is close to Gold?" have
non-trivial answers.
"""

import datetime
import random
from importlib import resources
from pathlib import Path

import duckdb

TIERS = [
    (1, "Bronze", 0, 1.0),
    (2, "Silver", 5_000, 1.25),
    (3, "Gold", 15_000, 1.5),
    (4, "Platinum", 40_000, 2.0),
]

FIRST_NAMES = [
    "Ava",
    "Ben",
    "Carla",
    "Diego",
    "Elena",
    "Farid",
    "Grace",
    "Hugo",
    "Iris",
    "Jonas",
    "Kira",
    "Liam",
    "Mona",
    "Nadia",
    "Omar",
    "Priya",
    "Quinn",
    "Rosa",
    "Sam",
    "Tara",
]
LAST_NAMES = [
    "Alvarez",
    "Baker",
    "Chen",
    "Dubois",
    "Ekwueme",
    "Fischer",
    "Garcia",
    "Hansen",
    "Ivanov",
    "Jensen",
    "Kim",
    "Lopez",
    "Meyer",
    "Novak",
    "Okafor",
    "Patel",
    "Quist",
    "Rossi",
    "Silva",
    "Tanaka",
]
CITIES = [
    "Atlanta",
    "Boston",
    "Chicago",
    "Denver",
    "El Paso",
    "Frankfurt",
    "Geneva",
    "Houston",
    "Indianapolis",
    "Jacksonville",
    "Kansas City",
    "Lisbon",
]
CATEGORIES = ["grocery", "travel", "dining", "fuel", "online", "pharmacy"]
REWARDS = ["gift card", "flight upgrade", "hotel night", "merchandise", "statement credit"]

FIRST_DAY = datetime.date(2024, 8, 1)
LAST_DAY = datetime.date(2026, 7, 1)
NEAR_TIER_MEMBERS = 12


def _tier_for(points: int) -> int:
    tier_id = 1
    for tid, _, min_points, _ in TIERS:
        if points >= min_points:
            tier_id = tid
    return tier_id


def _random_date(rng: random.Random) -> datetime.date:
    span = (LAST_DAY - FIRST_DAY).days
    return FIRST_DAY + datetime.timedelta(days=rng.randrange(span))


def generate(
    con: duckdb.DuckDBPyConnection, seed: int = 42, members: int = 500
) -> duckdb.DuckDBPyConnection:
    """Populate an open DuckDB connection with the full loyalty schema and data."""
    rng = random.Random(seed)
    schema = (resources.files("strictcall.dataset") / "schema.sql").read_text(encoding="utf-8")
    con.execute(schema)

    con.executemany("INSERT INTO tiers VALUES (?, ?, ?, ?)", TIERS)

    member_rows = []
    for member_id in range(1, members + 1):
        first, last = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        member_rows.append(
            (
                member_id,
                f"{first} {last}",
                f"{first}.{last}.{member_id}@example.com".lower(),
                rng.choice(CITIES),
                _random_date(rng),
                1,  # placeholder; real tier assigned after transactions exist
            )
        )

    txn_rows = []
    earned: dict[int, int] = {}
    txn_id = 1
    for member_id in range(1, members + 1):
        for _ in range(rng.randint(1, 80)):
            amount = round(rng.uniform(5, 900), 2)
            points = int(amount)
            txn_rows.append(
                (txn_id, member_id, _random_date(rng), rng.choice(CATEGORIES), amount, points)
            )
            earned[member_id] = earned.get(member_id, 0) + points
            txn_id += 1

    # Top up a deterministic sample of members so they sit just below the next
    # tier threshold - the dataset's flagship demo question depends on this.
    candidates = [m for m in range(1, members + 1) if earned[m] < TIERS[-1][2]]
    for member_id in rng.sample(candidates, min(NEAR_TIER_MEMBERS, len(candidates))):
        next_min = next(min_pts for _, _, min_pts, _ in TIERS if min_pts > earned[member_id])
        target = next_min - rng.randint(50, 450)
        top_up = target - earned[member_id]
        if top_up <= 0:
            continue
        txn_rows.append((txn_id, member_id, _random_date(rng), "travel", float(top_up), top_up))
        earned[member_id] = target
        txn_id += 1

    member_rows = [
        (mid, name, email, city, joined, _tier_for(earned[mid]))
        for mid, name, email, city, joined, _ in member_rows
    ]
    con.executemany("INSERT INTO members VALUES (?, ?, ?, ?, ?, ?)", member_rows)
    con.executemany("INSERT INTO transactions VALUES (?, ?, ?, ?, ?, ?)", txn_rows)

    redemption_rows = []
    redemption_id = 1
    for member_id in range(1, members + 1):
        if rng.random() < 0.4:
            continue
        remaining = earned[member_id]
        for _ in range(rng.randint(1, 4)):
            spend = int(remaining * rng.uniform(0.05, 0.4))
            if spend < 100:
                break
            redemption_rows.append(
                (redemption_id, member_id, _random_date(rng), spend, rng.choice(REWARDS))
            )
            remaining -= spend
            redemption_id += 1
    con.executemany("INSERT INTO redemptions VALUES (?, ?, ?, ?, ?)", redemption_rows)
    return con


def build_database(path: str | Path, seed: int = 42, members: int = 500) -> Path:
    """Create (or overwrite) a DuckDB file with the generated dataset."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.unlink(missing_ok=True)
    con = duckdb.connect(str(path))
    try:
        generate(con, seed=seed, members=members)
    finally:
        con.close()
    return path
