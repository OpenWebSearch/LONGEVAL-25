import pandas as pd
from snorkel.labeling import labeling_function, PandasLFApplier
from snorkel.labeling.model.label_model import LabelModel
import ast
from preprocessing_train import write_tsv, get_queries, get_qrels, get_docs, merge_all, group_by_query, nb_of_clicks,list_to_string,remove_duplicate_authors
from ir_datasets_longeval import load




def snorkel_to_intent_category(df):
    df['snorkel_intent'] = df['snorkel_intent'].replace({
        -1: 'exploratory',
        1: 'navigational',
    })
    return df


def snorkel_apply(df,lfs):
    """
    Applying Snorkel to the queries for which we have information about previously relevant documents
    (queries that match on query_id with train)
    """

    applier = PandasLFApplier(lfs=lfs)
    L_train = applier.apply(df=df)
    label_model = LabelModel(cardinality=2, verbose=True)
    label_model.fit(L_train=L_train, n_epochs=500, log_freq=100)
    preds = label_model.predict(L=L_train)
    df['snorkel_intent'] = preds
    probs = label_model.predict_proba(L=L_train)
    df['snorkel_prob_navigational'] = probs[:, 1]
    return df

def add_columns(df):
    """
    Adding columns to fill in with the information from previously rel. documents
    """
    columns_to_add = [
    'doc_text', 'doc_id', 'doc_authors', 'doc_title',
    'nb_clicks']
    for col in columns_to_add:
        df[col] = None
    return df

def update_columns_from_train(df_target, df_source, columns_to_update, key='query_id'):
    """
    Update the test df with information about previously relevant documents for the queries
    that match on query_id with train
    """
    # Select only relevant columns + key
    df_source_subset = df_source[[key] + columns_to_update]

    # Merge on key, using left join to preserve df_target structure
    df_merged = df_target.merge(df_source_subset, on=key, how='left', suffixes=('', '_new'))

    # Fill target columns with new values (if present)
    for col in columns_to_update:
        new_col = f"{col}_new"
        if new_col in df_merged.columns:
            df_merged[col] = df_merged[new_col].combine_first(df_merged[col])
            df_merged.drop(columns=[new_col], inplace=True)

    return df_merged

def merge_final_results(df,df_intent):
    """
    Merge Snorkel results into a test dataframe. Intent by default for the queries: exploratory,
    probability navigational by default: 0.5
    """
    data_test = df.merge(
        df_intent[['query_id', 'snorkel_intent','snorkel_prob_navigational']],
        on='query_id',
        how='left'
    )
    data_test['snorkel_intent'] = data_test['snorkel_intent'].apply(
        lambda x: 'exploratory' if pd.isna(x) or x == 0 or x == '0' else x
    )
    # Step 2: Set snorkel_prob_navigational to 0.5 if intent is 'exploratory'
    data_test.loc[data_test['snorkel_intent'] == 'exploratory', 'snorkel_prob_navigational'] = 0.5

    #data_test['snorkel_prob_navigational'] = data_test['snorkel_prob_navigational'].fillna(0.5)
    return data_test

#Labels

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
    #train data
    dataset = load("longeval-sci/2024-11/train")
    le_queries = get_queries(dataset)
    le_qrels = get_qrels(dataset)
    le_docs = get_docs(dataset)

    #Preprocessing
    df_merged = merge_all(le_queries, le_qrels, le_docs)
    # preprocessing for Snorkel
    df_grouped_by_query = group_by_query(df_merged)
    df_queries_auth_title_string = list_to_string(df_grouped_by_query)
    df_unique_authors_same_paper = remove_duplicate_authors(df_queries_auth_title_string)
    data_train = nb_of_clicks(df_unique_authors_same_paper)
    print("Preprocessing done")

    #put test data here
    dataset_test = load("longeval-sci/2025-01")

    # only need queries here
    data_test = get_queries(dataset_test)
    data_test_ac = add_columns(data_test)
    #update overlapping queries with information about previously relevant documents from train.
    data_test_exp = update_columns_from_train(data_test_ac, data_train, ['doc_text', 'doc_id', 'doc_authors', 'doc_title',
    'nb_clicks'])
    data_test_overlap = data_test_exp[data_test_exp['doc_text'].notnull()].copy()
    
    #apply Snorkel to overlapping queries
    lfs = [lf_query_in_title_if_long, lf_query_in_authors, lf_low_clicks_navigational]
    df_snorkel_pred = snorkel_apply(data_test_overlap, lfs)
    df_final_intent = snorkel_to_intent_category(df_snorkel_pred)
    data_test_final = merge_final_results(data_test_exp, df_final_intent)

    output_directory = "data/output/longeval25_sci_test_2025_01_intent.tsv"
    write_tsv(data_test_final, output_directory)

    percentage_navigational = (data_test_final['snorkel_intent'] == "navigational").mean() * 100
    print(f"Percentage of navigational queries: {percentage_navigational:.2f}%")

    overlapping_queries = data_test_final[data_test_final['doc_text'].notnull()]
    percentage_navigational = (overlapping_queries['snorkel_intent'] == "navigational").mean() * 100
    print(f"Percentage of navigational queries in overlapping queries: {percentage_navigational:.2f}%")
