#!/usr/bin/env python3
from __future__ import annotations

import argparse

from run_increment_event_suite_v1_3 import run

FEATURES = [
    "fruit_cluster_num",
    "fruit_cluster_num_lag1",
    "days_since_previous",
    "days_since_crop_start",
]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    run(args.run_id, features=FEATURES)
