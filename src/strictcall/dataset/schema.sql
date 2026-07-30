CREATE TABLE tiers (
    tier_id INTEGER PRIMARY KEY,
    name VARCHAR NOT NULL,
    min_points INTEGER NOT NULL,
    points_multiplier DOUBLE NOT NULL
);

CREATE TABLE members (
    member_id INTEGER PRIMARY KEY,
    full_name VARCHAR NOT NULL,
    email VARCHAR NOT NULL,
    city VARCHAR NOT NULL,
    joined_at DATE NOT NULL,
    tier_id INTEGER NOT NULL REFERENCES tiers (tier_id)
);

CREATE TABLE transactions (
    txn_id INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL REFERENCES members (member_id),
    occurred_at DATE NOT NULL,
    category VARCHAR NOT NULL,
    amount_usd DECIMAL(10, 2) NOT NULL,
    points_earned INTEGER NOT NULL
);

CREATE TABLE redemptions (
    redemption_id INTEGER PRIMARY KEY,
    member_id INTEGER NOT NULL REFERENCES members (member_id),
    redeemed_at DATE NOT NULL,
    points_spent INTEGER NOT NULL,
    reward VARCHAR NOT NULL
);

CREATE VIEW v_balances AS
WITH earned AS (
    SELECT member_id, SUM(points_earned) AS points_earned_total
    FROM transactions
    GROUP BY member_id
),

spent AS (
    SELECT member_id, SUM(points_spent) AS points_spent_total
    FROM redemptions
    GROUP BY member_id
)

SELECT
    m.member_id,
    m.full_name,
    COALESCE(e.points_earned_total, 0) AS points_earned_total,
    COALESCE(s.points_spent_total, 0) AS points_spent_total,
    COALESCE(e.points_earned_total, 0) - COALESCE(s.points_spent_total, 0) AS current_balance,
    t.name AS current_tier,
    nt.name AS next_tier,
    nt.min_points - COALESCE(e.points_earned_total, 0) AS points_to_next_tier
FROM members AS m
LEFT JOIN earned AS e ON m.member_id = e.member_id
LEFT JOIN spent AS s ON m.member_id = s.member_id
INNER JOIN tiers AS t ON m.tier_id = t.tier_id
LEFT JOIN tiers AS nt ON nt.tier_id = t.tier_id + 1;
