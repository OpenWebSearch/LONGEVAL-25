#!/usr/bin/env python3
from pathlib import Path
from shutil import copy
from typing import Optional
import click
import pandas as pd
import pyterrier as pt
from ir_datasets_longeval import load
from tira.third_party_integrations import normalize_run
from tirex_tracker import tracking


class QrelBoost(pt.Transformer):
    def __init__(self, dataset, memory: Optional[int] = None):
        super().__init__()
        self.dataset = dataset
        self.memory = memory if memory is not None else len(dataset.prior_datasets)

    def boost(self, doc, _lambda=0.7, mu=2):
        if doc["label"] == 1:
            return doc["score"] * _lambda**2
        elif doc["label"] == 2:
            return doc["score"] * _lambda**2 * mu
        else:
            return doc["score"] * (1 - _lambda) ** 2

    def apply_boost(self, df, prior_dataset):
        assert prior_dataset != self.dataset.timestamp.strftime(
            "%Y-%m"
        ), "Cannot apply boost to the same sub-collection"

        historic_qrels = pt.io.read_qrels(prior_dataset.qrels_path())

        df = df.merge(historic_qrels, on=["docno", "qid"], how="left")
        df = df.drop_duplicates(keep="first")

        df["score"] = df.apply(self.boost, axis=1)
        df.drop(columns=["label"], inplace=True)
        return df

    def transform(self, df):
        print(f"Applying boost to {self.memory} prior datasets")
        # df["score"] = df.groupby("qid")["score"].transform(lambda x: x / x.max())

        for prior_dataset in self.dataset.get_prior_datasets()[: self.memory]:
            if prior_dataset.has_qrels():
                df = self.apply_boost(df, prior_dataset).copy()
            else:
                print(
                    f"Skipping prior dataset {prior_dataset.get_snapshot()}, no qrels available"
                )
                continue

        df["rank"] = df.groupby("qid")["score"].rank(ascending=False).astype(int)

        df = df.sort_values(["qid", "rank"])

        return df

def process_dataset(ir_dataset, prior_stage_dir, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        prior_stage = pt.io.read_results(prior_stage_dir / "run.txt.gz")
        if ir_dataset.get_prior_datasets():
            print(">>> Using QrelBoost")
            pipeline = prior_stage >> QrelBoost(ir_dataset)
        else:
            print(">>> No prior datasets, using BM25")
            pipeline = prior_stage

        topics = pd.DataFrame(
            [
                {"qid": i.query_id, "query": i.default_text()}
                for i in ir_dataset.queries_iter()
            ]
        )

        run = normalize_run(pipeline(topics), "qrel-boost-core")
        pt.io.write_results(run, output_directory / "run.txt.gz")


@click.command()
@click.option("--dataset", type=str, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--prior-stage", type=Path, required=True, help="The directory with the prior stage rankings directory.")
def main(dataset, output, prior_stage):
    ir_dataset = load(dataset)
    datasets = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()

    for d in datasets:
        process_dataset(d, prior_stage / d.get_snapshot(), output / d.get_snapshot())

    # The ir-metadata description of your approach
    ir_metadata = Path(__file__).parent / "qrel-boost-ir-metadata.yml"

    copy(ir_metadata, output / "ir-metadata.yml")


if __name__ == "__main__":
    main()