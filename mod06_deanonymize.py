
import pandas as pd

def load_data(anonymized_path, auxiliary_path):
    """
    Load anonymized and auxiliary datasets.
    """
    anon = pd.read_csv(anonymized_path)
    aux = pd.read_csv(auxiliary_path)
    return anon, aux


def link_records(anon_df, aux_df):
    """
    Attempt to link anonymized records to auxiliary records
    using exact matching on quasi-identifiers.

    Returns a DataFrame with columns:
      anon_id, matched_name
    containing ONLY uniquely matched records.
    """
    columns = []

    for col in anon_df.columns:
        if col in aux_df.columns and col not in ["anon_id", "name"]:
            columns.append(col)

    if len(columns) == 0:
        return pd.DataFrame(columns=["anon_id", "matched_name"])

    matches = pd.merge(anon_df, aux_df, on=columns, how="inner")

    counts = matches.groupby("anon_id").size()
    unique_ids = counts[counts == 1].index

    matches = matches[matches["anon_id"].isin(unique_ids)]

    result = matches[["anon_id", "name"]].copy()
    result = result.rename(columns={"name": "matched_name"})

    return result


def deanonymization_rate(matches_df, anon_df):
    """
    Compute the fraction of anonymized records
    that were uniquely re-identified.
    """
    total = len(anon_df)

    if total == 0:
        return 0.0

    matched = matches_df["anon_id"].nunique()
    rate = matched / total

    return rate
