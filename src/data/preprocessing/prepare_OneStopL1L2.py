#!/usr/bin/env python3
"""
sync_onestop_downloads.py

Replaces a long bash sequence of:
- mkdir -p .../downloads and .../precomputed_events for multiple variants
- rm -rf .../downloads/* and .../precomputed_events/*
- cp -r SRC/. DST/downloads/

Usage:
  python sync_onestop_downloads.py
Optional:
  python sync_onestop_downloads.py --base /data --dry-run
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

from sympy import Q

from src.configs import data
from concurrent.futures import ProcessPoolExecutor, as_completed


DATASETS = ["OneStopL2", "OneStopL1L2"]
MODES = ["ORD", "REP"]  # ORD -> ordinary, REP -> repeated
CONDS = ["ALL", "GATH", "HUNT"]

SUBDIRS_TO_PREPARE = ["downloads", "precomputed_events", "precomputed_reading_measures"]


COND_MAP = {
    "ALL": "All",
    "GATH": "Gathering",
    "HUNT": "Hunting",
}

DATES = {
    "OneStopL1L2": "20250526_20260119",
    "OneStopL2": "20260119",
}

SOURCE_ROOT = {
    "OneStopL1L2": Path(f"/data/home/shared/proficiency/onestop_L1L2/reports/public/{DATES['OneStopL1L2']}"),
    "OneStopL2": Path(f"/data/home/shared/proficiency/onestop_L2/reports/public/{DATES['OneStopL2']}"),
}

MODE_MAP = {
    "ORD": "ordinary",
    "REP": "repeated",
}

FILE_CATAGS = ["ia", "fixations"]

FILE_NAMES = {
    "ia": "ia_Paragraph.csv",
    "fixations": "fixations_Paragraph.csv",
}

NEW_FILE_NAMES = {
    "ia": "combined_ia.csv",
    "fixations": "combined_fixations.csv",
}


def sync_data_to_folders(dataset_org, mode, cond):
    for subdir in SUBDIRS_TO_PREPARE:
        makdir_str= f"mkdir data/{dataset_org}_{mode}_{cond}/{subdir}"
        if not os.path.exists(f"data/{dataset_org}_{mode}_{cond}/{subdir}"):
            os.system(makdir_str)
        rm_str = f"rm -rf data/{dataset_org}_{mode}_{cond}/{subdir}/*"
        if dataset_org == "OneStopL2":
            cp_str = f"cp -r {SOURCE_ROOT[dataset_org]}/{MODE_MAP[mode]}/{COND_MAP[cond]}/with_metadata/. data/{dataset_org}_{mode}_{cond}/{subdir}/"
        else:
            cp_str = f"cp -r {SOURCE_ROOT[dataset_org]}/{MODE_MAP[mode]}/{COND_MAP[cond]}/. data/{dataset_org}_{mode}_{cond}/{subdir}/"
        os.system(rm_str)
        os.system(cp_str)

def prepare_final_data_folders(dataset_org, mode, cond):
    for file in FILE_CATAGS:
        for subdir in SUBDIRS_TO_PREPARE[1:]:
            run_str = f"cp data/{dataset_org}_{mode}_{cond}/{subdir}/{FILE_NAMES[file]} data/{dataset_org}_{mode}_{cond}/{subdir}/{NEW_FILE_NAMES[file]}"

def prepare_all_folders(datasets=DATASETS, modes=MODES, conditions=CONDS, dry_run=False):
    for dataset_org in datasets:
        for mode in modes:
            for cond in conditions:
                if dry_run:
                    print(f"[DRY RUN] Would prepare folders and sync data for: {dataset_org}_{mode}_{cond}")
                else:
                    print(f"{dataset_org}_{mode}_{cond}")
                    sync_data_to_folders(dataset_org, mode, cond)
                    prepare_final_data_folders(dataset_org, mode, cond)

def run_dataset(dataset_org, mode, cond, dry_run=False):
    # assuming preperation is done, run the final processing script
    preprocess_str = f"python src/data/preprocessing/preprocess_data.py --dataset {dataset_org}_{mode}_{cond}"
    create_folds_str = f"python src/data/preprocessing/create_folds.py --dataset {dataset_org}_{mode}_{cond}"
    if dry_run:
        print(f"[DRY RUN] Would execute: {preprocess_str}")
        print(f"[DRY RUN] Would execute: {create_folds_str}")
    else:
        os.system(preprocess_str)
        os.system(create_folds_str)

def run_all_datasests(datasets=DATASETS, modes=MODES, conditions=CONDS):
    for dataset_org in datasets:
        for mode in modes:
            for cond in conditions:
                run_dataset(dataset_org, mode, cond)



def run_all_datasets_parallel(datasets=DATASETS, modes=MODES, conditions=CONDS, max_workers=None, dry_run=False):
    """Run dataset processing in parallel."""
    tasks = [
        (dataset_org, mode, cond, dry_run)
        for dataset_org in datasets
        for mode in modes
        for cond in conditions
    ]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(run_dataset, *task): task for task in tasks}
        for future in as_completed(futures):
            task = futures[future]
            try:
                future.result()
                print(f"Completed: {task}")
            except Exception as e:
                print(f"Failed: {task} with error: {e}")



def main():
    prepare_all_folders(dry_run=True)
    run_all_datasets_parallel(max_workers = 12, dry_run=True)

if __name__ == "__main__":
    main()