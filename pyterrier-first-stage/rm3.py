#!/usr/bin/env python3
from pathlib import Path
from shutil import copy
from typing import Optional
import click
import pandas as pd
import pyterrier as pt
from ir_datasets_longeval import load
from baseline import get_index
from tirex_tracker import tracking


def process_dataset(ir_dataset, index_dir, prior_stage_dir, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        index = get_index(ir_dataset, index_directory)
        prior_stage = pt.io.read_results(prior_stage_dir / "run.txt.gz")
        bm25 = pt.terrier.Retriever(index, wmodel="BM25")
        pipeline = prior_stage >> pt.rewrite.RM3(index) >> bm25

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

        run = normalize_run(pipeline(topics), "rm3")
        pt.io.write_results(run, output_directory / "run.txt.gz")


@click.command()
@click.option("--dataset", type=str, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
@click.option("--prior-stage", type=Path, required=True, help="The directory with the prior stage rankings directory.")
def main(dataset, output, index, prior_stage):
    ir_dataset = load(dataset)
    datasets = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()

    for d in datasets:
        process_dataset(d, index / d.get_snapshot(), prior_stage / d.get_snapshot(), output / d.get_snapshot())

    # The ir-metadata description of your approach
    ir_metadata = Path(__file__).parent / "rm3-ir-metadata.yml"
    copy(ir_metadata, output / "ir-metadata.yml")


if __name__ == "__main__":
    main()