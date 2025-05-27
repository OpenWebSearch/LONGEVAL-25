#!/usr/bin/env python3
from pathlib import Path
from shutil import copy

import click
import pandas as pd
import pyterrier as pt
from ir_datasets_longeval import load
from baseline import get_index
import gzip
from tqdm import tqdm
from tirex_tracker import tracking
from glob import glob


def process_dataset(ir_dataset, output_directory, query_intents, navigational_run, informational_run):
    if (output_directory / "run.txt.gz").exists():
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        navigational_stage = pt.io.read_results(navigational_run / "run.txt.gz")
        informational_stage = pt.io.read_results(navigational_run / "run.txt.gz")
        
        ranking = []

        for query in tqdm(list(ir_dataset.queries_iter())):
            query_text = " ".join(tokeniser.getTokens(query.default_text()))
            docs = []

            if query.query_id in snapshots:
                docs_from_core = sorted([i for i in snapshots[query.query_id].keys()], key=lambda i: snapshots[query.query_id][i])
                for i in docs_from_core:
                    docs.extend([str(i)])
            covered_docs = set(docs)

            for _, i in retriever.search(query_text).iterrows():
                if i["docno"] in covered_docs:
                    continue
                docs.extend([i["docno"]])

            pos = 0
            for i in docs[:1000]:
                ranking.extend([f"{query.query_id} Q0 {i} {pos} {1000-pos} fusion-with-core"])

        with gzip.open(output_directory / "run.txt.gz", "wt") as f:
            for r in ranking:
                f.write(r + '\n')
        copy(index_directory / "index-ir-metadata.yml", output_directory / "index-ir-metadata.yml")

def load_intents(intent_file):
    ret = {}
    for tsv_file in glob(f"{intent_file}/*.tsv"):
        df = pd.read_csv(tsv_file, sep="\t")
        for _, i in df.iterrows():
            if i["query_text"] not in ret:
                ret[i["query_text"]] = []

            if 'snorkel_intent' in i:
                ret[i["query_text"]].extend([i["snorkel_intent"]])
            else:
                ret[i["query_text"]].extend([i["snorkel_label"]])

    return {k: {'exploratory': len([i for i in v if i == 'exploratory'])/len(v), 'navigational': len([i for i in v if i == 'navigational'])/len(v)} for k, v in ret.items()}

@click.command()
@click.option("--dataset", type=str, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--query-intents", type=Path, required=True, help="The file that contains the query-intent predictions.")
@click.option("--navigational-run", type=Path, required=True, help="The navigational run")
@click.option("--informational-run", type=Path, required=True, help="The informational run")
def main(dataset, output, query_intents, navigational_run, informational_run):
    ir_dataset = load(dataset)
    intents = load_intents(query_intents)
    sub_collections = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()
    for k in ["galaxy", "ris", "stratosphere", "sexuality"]:
        print(k, intents[k])

    for snapshot in sub_collections:
        process_dataset(snapshot, output / snapshot.get_snapshot(), intents, navigational_run / snapshot.get_snapshot(), informational_run / snapshot.get_snapshot())

    # The ir-metadata description of your approach
    ir_metadata = Path(__file__).parent / "intent-ir-metadata.yml"

    copy(ir_metadata, output / "ir-metadata.yml")


if __name__ == "__main__":
    main()
