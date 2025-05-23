import pandas as pd

from sklearn.model_selection import train_test_split
from ir_datasets_longeval import load


files = {}

train_collection = load('longeval-sci/2024-11/train')

queries = list(train_collection.queries_iter())
documents = [d.doc_id for d in train_collection.docs_iter()]
qrels = list(train_collection.qrels_iter())

d_both, d_remaining = train_test_split(documents, train_size=0.25)
d_first, d_second = train_test_split(d_remaining, train_size=0.5)

files['docs_train'] = pd.DataFrame({'doc_id': d_both + d_first})
files['docs_test'] = pd.DataFrame({'doc_id': d_both + d_second})

q_both, q_remaining = train_test_split(queries, train_size=0.25)
q_first, q_second = train_test_split(q_remaining, train_size=0.5)

files['queries_train'] = pd.DataFrame(q_both + q_first)
files['queries_test'] = pd.DataFrame(q_both + q_second)

docids_train = set(files['docs_train']['doc_id'])
docids_test = set(files['docs_test']['doc_id'])

qids_train = set(files['queries_train']['query_id'])
qids_test = set(files['queries_test']['query_id'])


files['qrels_train'] = pd.DataFrame([qrel for qrel in qrels if qrel.doc_id in docids_train and qrel.query_id in qids_train])
files['qrels_test'] = pd.DataFrame([qrel for qrel in qrels if qrel.doc_id in docids_test and qrel.query_id in qids_test])

for filename, df in files.items():
    df.to_csv(f'{filename}.csv.gz', index=False, compression='gzip')
