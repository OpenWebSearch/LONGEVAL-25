#!/usr/bin/env python3
from ir_datasets_longeval import load


if __name__ == "__main__":
    dataset = load('longeval-sci/clef-2025-test')
    for snapshot in dataset.get_datasets():
        print(snapshot.get_snapshot() + ':' + str([i.get_snapshot() for i in snapshot.get_prior_datasets()]))
