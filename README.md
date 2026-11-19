# gta6-trophy-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jthan46/gta6-trophy-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/jthan46/gta6-trophy-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A Python forecaster that projects every GTA VI trophy's Exophase earn rate to 26 November 2026, blending earners elasticity against player growth with a saturating time trend on the logit scale. The case history is first compressed to its distinct snapshots, because Exophase leaves its page frozen between refreshes and then jumps. Each trophy's earners are then modelled two ways - scaling with the projected tracked-player count at that trophy's own shrunk elasticity, and following a logit trend against ln(1 + hours) - and the two views are averaged into a median with an honest, horizon-aware spread whose lower tail widens when a stale page makes a diluting refresh likely. The program is standard-library only, deterministic, and writes one forecast file per case file it finds in the input directory.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=bjydwhj1&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
