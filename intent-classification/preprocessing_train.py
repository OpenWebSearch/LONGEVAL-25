from ir_datasets_longeval import load
import pandas as pd
import ast
import json


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

def list_to_string(df):
    df['doc_authors'] = df['doc_authors'].apply(json.dumps)
    df['doc_title'] = df['doc_title'].apply(json.dumps)
    return df

def remove_duplicate_authors(df):
    """
    If the paper's name is already there and the names of the authors of the same paper are repeated twice
    """
    def process_row(row):
        try:
            titles = ast.literal_eval(row['doc_title'])
            authors = ast.literal_eval(row['doc_authors'])
        except Exception:
            return row  # Skip malformed entries

        seen = set()
        filtered_titles = []
        filtered_authors = []

        for title, author_group in zip(titles, authors):
            if title not in seen:
                seen.add(title)
                filtered_titles.append(title)
                filtered_authors.append(author_group)

        row['doc_title'] = str(filtered_titles)
        row['doc_authors'] = str(filtered_authors)
        return row
    return df.apply(process_row, axis=1)