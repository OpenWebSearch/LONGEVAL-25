import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel
import ast
from preprocessing import read_tsv, write_tsv


def list_to_string(df):
    df['doc_title'] = df['doc_title'].apply(lambda x: ' '.join(x) if isinstance(x, list) else x)
    df['doc_authors'] = df['doc_authors'].apply(lambda x: ', '.join(x) if isinstance(x, list) else x)
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


def snorkel_to_intent_category(df):
    df['snorkel_label'] = df['snorkel_label'].replace({
        -1: 'exploratory',
        1: 'navigational',
    })
    return df


def snorkel_apply(df):
    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100)
    preds = label_model.predict(L=L_train)
    df['snorkel_label'] = preds
    probs = label_model.predict_proba(L=L_train)
    df['snorkel_prob_navigational'] = probs[:, 1]
    return df



NAVIGATIONAL = 1
EXPLORATORY = -1

@labeling_function()
def lf_query_in_title_if_long(x):
    query = x.query_text.lower()
    title = x.doc_title.lower()
    if len(query.split()) > 2 and query in title:
        return NAVIGATIONAL
    return EXPLORATORY


@labeling_function()
def lf_query_in_authors(x):
    """
    If the query is the name of an author and only one paper by this author was clicked, the user
    was searching for a specific paper by this author. Consequently, it is a navigational query.
    """
    query = x.query_text.lower().strip()
    try:
        author_groups = ast.literal_eval(x.doc_authors)
    except (ValueError, SyntaxError, TypeError):
        return EXPLORATORY
    if not isinstance(author_groups, list):
        return EXPLORATORY
    matched_groups = 0
    for group in author_groups:
        if not isinstance(group, list):
            continue
        for author in group:
            if not isinstance(author, dict):
                continue
            author_name = author.get('name', '').lower()
            if query == author_name or query in author_name:
                matched_groups += 1
                break  # Only count once per group
    if matched_groups == 1:
        return NAVIGATIONAL
    else:
        return EXPLORATORY


@labeling_function()
def lf_low_clicks_navigational(x):
    if x.nb_clicks <= 3:
        return NAVIGATIONAL
    return EXPLORATORY


if __name__ == '__main__':
    df_queries = read_tsv("data/output/docs_grouped_by_queries.tsv")
    df_queries_auth_title_string = list_to_string(df_queries)
    df_unique_authors_same_paper = remove_duplicate_authors(df_queries_auth_title_string)
    lfs = [lf_query_in_title_if_long, lf_query_in_authors, lf_low_clicks_navigational]
    df_snorkel_pred = snorkel_apply(df_unique_authors_same_paper)
    df_final_intent = snorkel_to_intent_category(df_snorkel_pred)
    print(df_final_intent['snorkel_label'])
    write_tsv(df_final_intent,"data/output/longeval25_sci_train_intent.tsv")

    navigational_count = df_final_intent['snorkel_label'].value_counts().get('navigational', 0)
    print(f"Number of navigational labels: {navigational_count}")

    percentage_navigational = (df_final_intent['snorkel_label'] == "navigational").mean() * 100
    print(f"Percentage of navigational queries: {percentage_navigational:.2f}%")