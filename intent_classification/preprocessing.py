from ir_datasets_longeval import load
import pandas as pd


def read_tsv(filename):
    df = pd.read_csv(filename, sep='\t')
    return df


def write_tsv(df,filename):
    df.to_csv(filename, sep='\t',index=False)


def get_queries(ds_longeval):
    queries = [(q.query_id, q.text) for q in ds_longeval.queries_iter()]
    df_queries = pd.DataFrame(queries, columns=["query_id", "query_text"])
    return df_queries


def get_qrels(ds_longeval):
    qrels = [(qrel.query_id, qrel.doc_id, qrel.relevance) for qrel in ds_longeval.qrels_iter()]
    # Convert to DataFrame for convenience
    df_qrels = pd.DataFrame(qrels, columns=["query_id", "doc_id", "relevance"])
    return df_qrels


def get_docs(ds_longeval):
    # Iterate through documents
    docs = [(doc.doc_id, doc.abstract, doc.links, doc.title, doc.authors) for doc in  ds_longeval.docs_iter()]
    # Convert to DataFrame
    df_docs = pd.DataFrame(docs, columns=["doc_id", "doc_text", "doc_links", "doc_title", "doc_authors"])
    return df_docs


def merge_all(queries,qrels,docs):
    """
    merge queries, qrels and docs (abstracts)
    """
    df_queries_rel = queries.merge(qrels, on="query_id", how="left")
    df_queries_rel_docs = df_queries_rel.merge(docs, on="doc_id", how="left")
    df_queries_rel_docs = df_queries_rel_docs[df_queries_rel_docs['relevance'] != 0]
    return df_queries_rel_docs


def group_by_query(df_queries_rel_docs):
    """
    group by query so all the authors, titles and abstracts are grouped together per query
    """
    grouped_df = df_queries_rel_docs.groupby(["query_id", "query_text"]).agg({
    "doc_text": lambda texts: " ".join(str(t) for t in texts if pd.notnull(t)),
    "doc_id": lambda ids: list(ids),
    "doc_authors": lambda authors: list(authors),
    "doc_title": lambda titles: list(titles),
    }).reset_index()
    grouped_df["nb_clicks"] = grouped_df["doc_id"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return grouped_df


def nb_of_clicks(grouped_df):
    """
    add number of clicks
    """
    grouped_df["nb_clicks"] = grouped_df["doc_id"].apply(lambda x: len(x) if isinstance(x, list) else 0)
    return grouped_df


if __name__ == '__main__':
    dataset = load("longeval-sci/2024-11/train")
    le_queries = get_queries(dataset)
    le_qrels = get_qrels(dataset)
    le_docs = get_docs(dataset)
    df_merged = merge_all(le_queries,le_qrels,le_docs)
    df_grouped_by_query = group_by_query(df_merged)
    df_and_nb_of_clicks = nb_of_clicks(df_grouped_by_query)
    write_tsv(df_and_nb_of_clicks,"data/output/docs_grouped_by_queries.tsv")