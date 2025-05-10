#!/usr/bin/env python3
from pathlib import Path
from shutil import copy

import click
import pandas as pd
import pyterrier as pt
from ir_datasets_longeval import load

from tirex_tracker import tracking


def get_index(ir_dataset, index_directory):
    # PyTerrier needs an absolute path
    index_directory = index_directory.resolve().absolute()

    if (
        not index_directory.exists()
        or not (index_directory / "data.properties").exists()
    ):
        with tracking(export_file_path=index_directory / "index-ir-metadata.yml"):
            # build the index
            indexer = pt.IterDictIndexer(
                str(index_directory), overwrite=True, meta={"docno": 100, "text": 20480}, properties={"metaindex.compressed.reverse.allow.duplicates": True}
            )

            # you can do some custom document processing here
            docs = (
                {"docno": i.doc_id, "text": i.default_text()}
                for i in ir_dataset.docs_iter()
            )
            indexer.index(docs)

    return pt.IndexFactory.of(str(index_directory))


def process_dataset(ir_dataset, index_directory, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    index = get_index(ir_dataset, index_directory)
    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        retriever = pt.terrier.Retriever(index, wmodel="PL2")

        # potentially do some query processing
        topics = pd.DataFrame(
            [
                {"qid": i.query_id, "query": i.default_text()}
                for i in ir_dataset.queries_iter()
            ]
        )

        # PyTerrier needs to use pre-tokenized queries
        tokeniser = pt.java.autoclass(
            "org.terrier.indexing.tokenisation.Tokeniser"
        ).getTokeniser()

        topics["query"] = topics["query"].apply(
            lambda i: " ".join(tokeniser.getTokens(i))
        )

        run = retriever(topics)
        pt.io.write_results(run, output_directory / "run.txt.gz")
        copy(index_directory / "index-ir-metadata.yml", output_directory / "index-ir-metadata.yml")


@click.command()
@click.option("--dataset", type=str, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
def main(dataset, output, index):
    ir_dataset = load(dataset)
    sub_collections = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()

    for snapshot in sub_collections:
        process_dataset(snapshot, index / snapshot.get_snapshot(), output / snapshot.get_snapshot())

    # The ir-metadata description of your approach
    ir_metadata = Path(__file__).parent / "ir-metadata.yml"

    copy(ir_metadata, output / "ir-metadata.yml")


if __name__ == "__main__":
    main()
