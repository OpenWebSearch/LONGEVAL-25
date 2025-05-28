#!/usr/bin/env python3
from collections import defaultdict
import gzip
import json
from pathlib import Path
from shutil import copy

import click
import ir_datasets
import pandas as pd
import pyterrier as pt
from ir_datasets_longeval import load, LongEvalSciDataset
from document_clustering.clustering import Clustering

from tirex_tracker import tracking


class ClusterBooster(pt.Transformer):
    def __init__(self, dataset: LongEvalSciDataset, cluster_dir: Path, max_boost_factor: float = 0.1):
        super().__init__()
        self.dataset = dataset
        self.cluster_dir = cluster_dir
        self.max_boost_factor = max_boost_factor

        self.clusters_train, self.clusters_test = self.get_clusters()
        self.cluster_scores = self.get_cluster_scores()

    def transform(self, inp):
        scores = []

        for row in inp.itertuples():
            if row.qid in self.cluster_scores:
                cluster_score = self.cluster_scores[row.qid].get(self.clusters_test[row.docno], 0)
                norm_cluster_score = cluster_score / sum(self.cluster_scores[row.qid].values())
                boost_factor = norm_cluster_score * self.max_boost_factor

                scores.append(row.score * (1 + boost_factor))
            else:
                scores.append(row.score)

        result = inp.copy()
        result['score'] = scores

        return pt.model.add_ranks(result).sort_values(["qid", "rank"])

    def get_clusters(self):
        cluster_file = self.cluster_dir / "clusters.json.gz"

        if cluster_file.exists():
            with gzip.open(cluster_file, "rt") as f:
                clusters = json.load(f)
        else:
            with tracking(export_file_path=self.cluster_dir / "cluster-ir-metadata.yml"):
                query_collection = ir_datasets.load("msmarco-document/orcas")
                queries = (q.default_text() for q in query_collection.queries_iter())

                num_docs = sum(1 for _ in self._iter_docids(*self.dataset.get_prior_datasets()))
                num_clusters = num_docs // 100_000

                print(f"Clustering {num_docs} docs into {num_clusters} clusters")

                clustering = Clustering(num_clusters, tokenizer_kwargs=dict(min_df=200), merge_small_shards=False,
                                        glove_vectors='glove.6B.100d.txt', verbose=True, tol=1e-7)

                train_mapping = clustering.fit_transform(self._iter_docs(*self.dataset.get_prior_datasets()), queries)
                train_clusters = dict(zip(self._iter_docids(*self.dataset.get_prior_datasets()), train_mapping.tolist()))

                test_mapping = clustering.transform(self._iter_docs(self.dataset))
                test_clusters = dict(zip(self._iter_docids(self.dataset), test_mapping.tolist()))

                clusters = {"train": train_clusters, "test": test_clusters}

                with gzip.open(cluster_file, "wt") as f:
                    json.dump(clusters, f)

        return clusters["train"], clusters["test"]

    def get_cluster_scores(self):
        scores = defaultdict(lambda: defaultdict(int))

        for dataset in self.dataset.get_prior_datasets():
            for qrel in dataset.qrels_iter():
                if qrel.relevance > 0:
                    scores[qrel.query_id][self.clusters_train[qrel.doc_id]] += qrel.relevance

        return {k: dict(v) for k, v in scores.items()}

    def _iter_docs(self, *collections: LongEvalSciDataset):
        return (d.default_text() for collection in collections for d in collection.docs_iter())

    def _iter_docids(self, *collections: LongEvalSciDataset):
        return (d.doc_id for collection in collections for d in collection.docs_iter())




def get_index(ir_dataset, index_directory):
    # PyTerrier needs an absolute path
    index_directory = index_directory.resolve().absolute()

    if (
        not index_directory.exists()
        or not (index_directory / "data.properties").exists()
    ):
        with tracking(export_file_path=index_directory / "index-ir-metadata.yml"):
            indexer = pt.IterDictIndexer(
                str(index_directory), overwrite=True, meta={"docno": 100, "text": 20480}, threads=12
            )

            docs = (
                {"docno": i.doc_id, "text": i.default_text()}
                for i in ir_dataset.docs_iter()
            )
            indexer.index(docs)

    return pt.IndexFactory.of(str(index_directory))


def process_dataset(dataset, index_directory, cluster_directory, output_directory):
    if (output_directory / "run.txt.gz").exists():
        return

    index = get_index(dataset, index_directory)
    cluster_booster = ClusterBooster(dataset, cluster_directory)

    with tracking(export_file_path=output_directory / "retrieval-ir-metadata.yml"):
        retriever = pt.terrier.Retriever(index, wmodel="BM25") >> cluster_booster

        topics = pd.DataFrame(
            [
                {"qid": i.query_id, "query": i.default_text()}
                for i in dataset.queries_iter()
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
        copy(cluster_directory / "cluster-ir-metadata.yml", output_directory / "cluster-ir-metadata.yml")


@click.command()
@click.option("--dataset", type=str, help="The dataset id or a local directory.")
@click.option("--output", type=Path, required=True, help="The output directory.")
@click.option("--index", type=Path, required=True, help="The index directory.")
@click.option("--cluster", type=Path, required=True, help="The cluster directory.")
def main(dataset, output, index, cluster):
    ir_dataset = load(dataset)
    sub_collections = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()

    for snapshot in sub_collections:
        if not snapshot.get_prior_datasets():
            print(f'Snapshot {snapshot.get_snapshot()} does not have prior datasets, skipping...')

        print(snapshot.get_snapshot())
        print(snapshot.get_prior_datasets())

        process_dataset(snapshot, index / snapshot.get_snapshot(), cluster / snapshot.get_snapshot(), output / snapshot.get_snapshot())

    # The ir-metadata description of your approach
    ir_metadata = Path(__file__).parent / "ir-metadata.yml"

    copy(ir_metadata, output / "ir-metadata.yml")


if __name__ == "__main__":
    main()
