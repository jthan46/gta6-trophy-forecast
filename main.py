"""GTA VI trophy rarity forecaster (Python 3, standard library only).

Reads one makefun/gta6-trophies-case/v1 case file per run from the input
directory and writes, for every trophy in the case, five quantiles of its
Exophase earn rate (percent of tracked players who hold it) at the case's
target instant. Quantiles are returned on the percent scale; all modelling
happens on the logit scale, which is the scale the contest scores on.

How the forecast is built
-------------------------
Exophase does not update its page continuously: the counts sit frozen
between refreshes, then jump. A case history is therefore compressed to its
distinct snapshots (reads where the players-tracked count or any rate
actually changed). On those snapshots two complementary views are fitted
per trophy and averaged in logit space:

1. Earners elasticity. A trophy's earners are E = rate/100 * players.
   Between refreshes, earners grow more slowly than the tracked-player
   count (newly tracked players hold fewer trophies), so E scales like
   players**alpha with alpha < 1 and the rate itself falls as the player
   base grows. Each trophy's alpha is estimated from its own refresh-to-
   refresh changes and shrunk toward the field-wide alpha; the tracked-
   player count is projected to the target from its log-linear trend over
   the distinct snapshots (damped, since launch growth decelerates).

2. Saturating time trend. logit(rate) is fitted against ln(1 + hours
   since the first read) over the distinct snapshots - the shape of a
   player base progressing through a game while diluting - with each
   trophy's slope shrunk toward the field's median slope, then read at
   the target, anchored on the last observed rate.

The spread around the median combines a random-walk term that grows with
the square root of the horizon, a term proportional to how far the median
was moved (projection risk), a counting-noise term for trophies held by
very few players, and a refresh-risk term that widens the band when the
page has been sitting unchanged - a stale page is one whose next refresh
will move every rate at once. That refresh risk is one-sided: a refresh
dilutes (the tracked base grows faster than any trophy's earners), it
almost never lifts every rate, so when the risk is present the lower
tail is widened more than the upper tail.

Everything is closed-form arithmetic on the case file: no randomness, no
network, no state between runs, so two runs of a case agree byte for byte.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# Earners elasticity of the whole field when no refresh pair can measure
# it: the one refresh the house described in its rules took the tracked
# players 638 -> 3,868 while total trophies earned went 943 -> 4,276, an
# elasticity of ln(4276/943) / ln(3868/638) ~= 0.84.
PRIOR_ALPHA = 0.84
# Pseudo-count of earners for a trophy nobody in the last read holds, so
# its forecast is a small positive rate instead of a hard zero.
PSEUDO_EARNERS = 0.35
# The tracked-player trend is extrapolated at this fraction of its fitted
# pace: launch-week growth decelerates, it does not compound forever.
TREND_DAMPING = 0.75
MAX_PLAYER_GROWTH = 30.0

_STD_NORMAL = NormalDist()


def clamp_rate(pct: float) -> float:
    return min(99.99, max(0.01, pct))


def logit_of(pct: float) -> float:
    p = clamp_rate(pct) / 100.0
    return math.log(p / (1.0 - p))


def percent_of(logit_value: float) -> float:
    if logit_value >= 0:
        return 100.0 / (1.0 + math.exp(-logit_value))
    e = math.exp(logit_value)
    return 100.0 * e / (1.0 + e)


def round4(value: float):
    scaled = math.floor(value * 10000 + 0.5)
    result = scaled / 10000
    return int(result) if float(result).is_integer() else result


def parse_time(text: str) -> float:
    moment = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=timezone.utc)
    return moment.timestamp()


def distinct_snapshots(history):
    """Collapse consecutive reads that show the identical page."""
    snapshots = []
    for read in history:
        key = (read.get("players_tracked"), tuple(read.get("rates") or ()))
        if snapshots and snapshots[-1][0] == key:
            continue
        snapshots.append((key, read))
    return [read for _, read in snapshots]


def snapshot_hours(snapshots, case):
    """Hours from the first snapshot's slot to each snapshot and target."""
    origin = parse_time(snapshots[0].get("slot") or snapshots[0]["at"])
    hours = [(parse_time(r.get("slot") or r["at"]) - origin) / 3600.0 for r in snapshots]
    target = (parse_time(case["target"]) - origin) / 3600.0
    checkpoint = (parse_time(case["at"]) - origin) / 3600.0
    return hours, target, checkpoint


def earners_of(read, index):
    rate = (read.get("rates") or [None] * (index + 1))[index]
    players = read.get("players_tracked")
    if rate is None or players is None:
        return None
    return rate / 100.0 * players


def pair_elasticities(snapshots, index=None):
    """(weight, alpha) pairs from consecutive snapshots with player growth.

    With index None the pair uses the field's total earners; with an index
    it uses that trophy's earners only.
    """
    pairs = []
    for before, after in zip(snapshots, snapshots[1:]):
        p_before = before.get("players_tracked") or 0
        p_after = after.get("players_tracked") or 0
        if p_after <= p_before or p_before <= 0:
            continue
        if index is None:
            e_before = before.get("trophies_earned")
            e_after = after.get("trophies_earned")
            if not e_before or not e_after:
                continue
        else:
            e_before = earners_of(before, index)
            e_after = earners_of(after, index)
            if not e_before or not e_after or e_before <= 0 or e_after <= 0:
                continue
        pairs.append((math.log(p_after / p_before),
                      math.log(e_after / e_before) / math.log(p_after / p_before)))
    return pairs


def weighted_mean_alpha(pairs, fallback):
    if not pairs:
        return fallback
    total_weight = sum(w for w, _ in pairs)
    if total_weight <= 0:
        return fallback
    return sum(w * a for w, a in pairs) / total_weight


def project_players(snapshots, hours, target_hour, horizon_hours, stale_hours):
    """Projected players_tracked at the target and the growth ratio."""
    last_players = snapshots[-1].get("players_tracked") or 0
    if last_players <= 0:
        return 1.0, 1.0
    if len(snapshots) >= 2:
        xs = hours
        ys = [math.log(r.get("players_tracked") or last_players) for r in snapshots]
        mean_x = sum(xs) / len(xs)
        mean_y = sum(ys) / len(ys)
        denom = sum((x - mean_x) ** 2 for x in xs)
        slope = 0.0
        if denom > 0:
            slope = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / denom
        slope = min(0.06, max(0.0, slope))
        growth_log = slope * horizon_hours * TREND_DAMPING
    else:
        # One frozen page: a refresh is owed. Its size shrinks with the
        # horizon actually available for it to land in, and grows when the
        # page has already sat stale for hours (launch night's first
        # refresh multiplied the tracked players by six).
        base = math.log(3.2) * (1.0 - math.exp(-horizon_hours / 2.5))
        stale_bonus = 1.0 + 0.5 * min(stale_hours, 4.0) / 4.0
        growth_log = base * stale_bonus
    ratio = min(MAX_PLAYER_GROWTH, math.exp(growth_log))
    return last_players * ratio, ratio


def trend_slopes(snapshots, hours, count):
    """Per-trophy slopes of logit(rate) on ln(1 + hours), plus their median."""
    xs = [math.log1p(h) for h in hours]
    raw = []
    for index in range(count):
        points = []
        for read, x in zip(snapshots, xs):
            rate = (read.get("rates") or [])[index] if index < len(read.get("rates") or []) else None
            if rate is not None and 0.0 < rate < 100.0:
                points.append((x, logit_of(rate)))
        if len(points) < 2 or points[-1][0] == points[0][0]:
            raw.append(None)
            continue
        mean_x = sum(p[0] for p in points) / len(points)
        mean_y = sum(p[1] for p in points) / len(points)
        denom = sum((p[0] - mean_x) ** 2 for p in points)
        if denom <= 0:
            raw.append(None)
            continue
        raw.append(sum((p[0] - mean_x) * (p[1] - mean_y) for p in points) / denom)
    measured = sorted(s for s in raw if s is not None)
    if measured:
        mid = len(measured) // 2
        median_slope = measured[mid] if len(measured) % 2 else 0.5 * (measured[mid - 1] + measured[mid])
    else:
        median_slope = 0.0
    slopes = []
    for index, slope in enumerate(raw):
        n = 0
        if slope is not None:
            n = sum(1 for read in snapshots
                    if index < len(read.get("rates") or [])
                    and (read.get("rates") or [])[index] is not None
                    and 0.0 < (read.get("rates") or [])[index] < 100.0)
        if slope is None:
            slopes.append(median_slope)
        else:
            slopes.append((n * slope + 2.0 * median_slope) / (n + 2.0))
    return slopes


def forecast_case(case):
    trophies = case["trophies"]
    history = case.get("history") or []
    horizon_hours = case["horizon_minutes"] / 60.0
    horizon_days = case["horizon_minutes"] / 1440.0
    levels = case.get("levels") or [0.05, 0.25, 0.5, 0.75, 0.95]
    z_values = [_STD_NORMAL.inv_cdf(level) for level in levels]

    if not history:
        rows = []
        for trophy in trophies:
            centre = logit_of(1.0)
            rows.append({"id": trophy["id"],
                         "q": [round4(percent_of(centre + z * 1.2)) for z in z_values]})
        return {"trophies": rows}

    snapshots = distinct_snapshots(history)
    hours, target_hour, checkpoint_hour = snapshot_hours(snapshots, case)
    last = snapshots[-1]
    last_players = last.get("players_tracked") or 0
    # How long the page has shown exactly what the last snapshot shows.
    stale_hours = max(0.0, checkpoint_hour - hours[-1])

    field_alpha = weighted_mean_alpha(pair_elasticities(snapshots), PRIOR_ALPHA)
    projected_players, growth_ratio = project_players(
        snapshots, hours, target_hour, horizon_hours, stale_hours)
    slopes = trend_slopes(snapshots, hours, len(trophies))
    x_last = math.log1p(hours[-1])
    x_target = math.log1p(target_hour)

    rows = []
    for index, trophy in enumerate(trophies):
        # Last observed rate for this trophy, searching back past nulls.
        last_rate = None
        for read in reversed(snapshots):
            rates = read.get("rates") or []
            if index < len(rates) and rates[index] is not None:
                last_rate = rates[index]
                break
        if last_rate is None:
            last_rate = 0.0
        last_logit = logit_of(last_rate)

        # View 1: earners elasticity against the projected player base.
        alpha = weighted_mean_alpha(
            pair_elasticities(snapshots, index), field_alpha)
        own_pairs = len(pair_elasticities(snapshots, index))
        alpha = (own_pairs * alpha + 2.0 * field_alpha) / (own_pairs + 2.0)
        alpha = min(1.25, max(0.0, alpha))
        earners_last = (last_rate / 100.0) * last_players if last_players > 0 else 0.0
        earners_proj = max(earners_last, PSEUDO_EARNERS) * (growth_ratio ** alpha)
        if projected_players > 0:
            rate_a = clamp_rate(100.0 * earners_proj / projected_players)
        else:
            rate_a = last_rate
        logit_a = logit_of(rate_a)

        # View 2: saturating time trend, anchored on the last rate.
        logit_b = last_logit + slopes[index] * (x_target - x_last)

        centre = 0.5 * (logit_a + logit_b)

        # Spread: random walk + projection risk + counting noise + the
        # refresh risk of a page that has been sitting still.
        walk = 0.40 * math.sqrt(max(horizon_days, 0.0))
        projection_risk = 0.30 * abs(centre - last_logit)
        if earners_last > 0:
            counting = min(0.5, math.sqrt(1.0 / earners_last))
        else:
            counting = 0.5
        refresh_risk = 0.0
        if stale_hours >= 2.0 or len(snapshots) == 1:
            refresh_risk = 0.15 + 0.5 * (1.0 - math.exp(-horizon_hours / 6.0))
        sigma = math.sqrt(walk ** 2 + projection_risk ** 2 + counting ** 2 + refresh_risk ** 2)
        sigma = min(2.2, max(0.30, sigma))

        # One-sided refresh risk: stretch the lower tail, ease the upper.
        skew = min(1.0, refresh_risk / 0.4) if refresh_risk > 0 else 0.0
        sigma_down = sigma * (1.0 + 0.40 * skew)
        sigma_up = sigma * (1.0 - 0.15 * skew)
        quantiles = [
            round4(percent_of(centre + z * (sigma_down if z < 0 else sigma_up)))
            for z in z_values
        ]
        for k in range(1, len(quantiles)):
            if quantiles[k] < quantiles[k - 1]:
                quantiles[k] = quantiles[k - 1]
        rows.append({"id": trophy["id"], "q": quantiles})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        text = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / f"{path.stem}.out").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
