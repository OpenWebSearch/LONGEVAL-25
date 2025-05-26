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


def process_dataset(ir_dataset, index_directory, output_directory, snapshots):
    if (output_directory / "run.txt.gz").exists():
        return

    index = get_index(ir_dataset, index_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        retriever = pt.terrier.Retriever(index, wmodel="BM25")
        # PyTerrier needs to use pre-tokenized queries
        tokeniser = pt.java.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()
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

def load_snapshot_to_core_ranking(core_documents):
    ret = {}
    df = pd.read_json(core_documents, lines=True)
    for _, i in df.iterrows():
        if i['snapshot'] not in ret:
            ret[i['snapshot']] = {}
        if i["query_id"] not in ret[i['snapshot']]:
            ret[i['snapshot']][i["query_id"]] = {}
        ret[i['snapshot']][i["query_id"]][i["doc_id"]] = i["Rank"]
    return ret

@click.command()
@click.option("--dataset", type=str, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--core-documents", type=Path, required=True, help="The file that contains the core documents.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(dataset, output, index, core_documents):
    ir_dataset = load(dataset)
    snapshot_to_core_ranking = load_snapshot_to_core_ranking(core_documents)
    sub_collections = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()

    for snapshot in sub_collections:
        process_dataset(snapshot, index / snapshot.get_snapshot(), output / snapshot.get_snapshot(), snapshot_to_core_ranking[snapshot.get_snapshot()])

    # The ir-metadata description of your approach
    ir_metadata = Path(__file__).parent / "fusion-ir-metadata.yml"

    copy(ir_metadata, output / "ir-metadata.yml")


if __name__ == "__main__":
    main()
