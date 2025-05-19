#!/usr/bin/env python3
from pathlib import Path
import click
from ir_datasets_longeval import load
import gzip
import json
from tirex_tracker import tracking, ExportFormat

def predict_intent(query_text, relevant_docs):
    return 1

def add_query_to_relevant_docs(dataset, ret):
    print(dataset)
    if not dataset.has_qrels():
        return ret

    docs_store = dataset.docs_store()
    query_id_to_relevant_documents = {}

    for qrel in dataset.qrels_iter():
        if qrel.query_id not in query_id_to_relevant_documents:
            query_id_to_relevant_documents[qrel.query_id] = {}
        query_id_to_relevant_documents[qrel.query_id][qrel.doc_id] = qrel.relevance

    for query in dataset.queries_iter():
        query_text = query.default_text().lower().strip()
        if query_text not in ret:
            ret[query_text] = {}
        for doc in query_id_to_relevant_documents.get(query.query_id, []):
            ret[query_text][doc] = docs_store.get(doc)
    return ret

def query_to_relevant_documents_from_prior_datasets(datasets):
    ret = {}

    for dataset in datasets:
        ret = add_query_to_relevant_docs(dataset, ret)
        for p in dataset.get_prior_datasets():
            ret = add_query_to_relevant_docs(p, ret)

    return ret

def process_dataset(ir_dataset, output_directory, query_to_relevant_documents):
    if (output_directory / "queries.jsonl.gz").exists():
        return

    with tracking(export_file_path=output_directory / "intent-prediction-metadata.yml", export_format=ExportFormat.IR_METADATA):
        query_intents = {}
        for query_text in query_to_relevant_documents.keys():
            if len(query_to_relevant_documents[query_text]) == 0:
                continue
            query_intents[query_text] = predict_intent(query_text, query_to_relevant_documents[query_text])

    with gzip.open(output_directory / "queries.jsonl.gz", "wt") as f:
        for query in ir_dataset.queries_iter():
            query_text = query.default_text().lower().strip()
            f.write(json.dumps({"qid": query.query_id, "intent": query_intents.get(query_text, None)}) + "\n")

@click.command()
@click.option("--predict", type=str, help="The dataset id or a local directory on which the predictions should be made.")
@click.option("--prior-datasets", type=str, multiple=True, help="The dataset id or a local directory on which the predictions should be made.")
@click.option("--output", type=Path, required=True, help="The output directory.")
def main(predict, prior_datasets, output):
    ir_dataset = load(predict)
    datasets = [ir_dataset] if not ir_dataset.get_datasets() else ir_dataset.get_datasets()
    query_to_relevant_documents = query_to_relevant_documents_from_prior_datasets(datasets + [load(i) for i in prior_datasets])

    for d in datasets:
        process_dataset(d, output / d.get_snapshot(), query_to_relevant_documents)

if __name__ == "__main__":
    main()

