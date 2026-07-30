import duckdb

from strictcall.dataset import generate

TABLES = ["tiers", "members", "transactions", "redemptions"]


def snapshot(con: duckdb.DuckDBPyConnection) -> dict:
    return {t: con.execute(f"SELECT * FROM {t} ORDER BY 1").fetchall() for t in TABLES}


def test_same_seed_rebuilds_identical_data():
    first = generate(duckdb.connect(":memory:"), seed=99, members=40)
    second = generate(duckdb.connect(":memory:"), seed=99, members=40)
    assert snapshot(first) == snapshot(second)


def test_different_seed_changes_data():
    first = generate(duckdb.connect(":memory:"), seed=1, members=40)
    second = generate(duckdb.connect(":memory:"), seed=2, members=40)
    assert snapshot(first) != snapshot(second)


def test_member_tiers_match_lifetime_earned_points(demo_con):
    mismatches = demo_con.execute(
        """
        SELECT COUNT(*) FROM members m
        JOIN v_balances b ON m.member_id = b.member_id
        JOIN tiers t ON m.tier_id = t.tier_id
        LEFT JOIN tiers nt ON nt.tier_id = t.tier_id + 1
        WHERE b.points_earned_total < t.min_points
           OR (nt.tier_id IS NOT NULL AND b.points_earned_total >= nt.min_points)
        """
    ).fetchone()[0]
    assert mismatches == 0


def test_no_member_has_a_negative_balance(demo_con):
    negative = demo_con.execute(
        "SELECT COUNT(*) FROM v_balances WHERE current_balance < 0"
    ).fetchone()[0]
    assert negative == 0


def test_near_tier_members_are_planted(demo_con):
    close = demo_con.execute(
        "SELECT COUNT(*) FROM v_balances WHERE points_to_next_tier BETWEEN 1 AND 500"
    ).fetchone()[0]
    assert close >= 10
