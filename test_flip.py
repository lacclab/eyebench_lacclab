import pandas as pd
import pytest

def flip_group_to_features(df: pd.DataFrame, criterion: str, fixation_metrics: list) -> pd.DataFrame:
    """
    Note: The original code has a likely bug in `value_vars=[f"{df}_{col}" for col in fixation_metrics]`.
    `df` is the DataFrame object, which can't be used in an f-string for column names.
    Assuming it's meant to be `f"{criterion}_{col}"` based on context (e.g., columns named like 'pos_mean_FF').
    I've corrected it in this version for the test to work.
    """
    df_long = df.melt(id_vars=[criterion], 
                      value_vars=[f"{criterion}_{col}" for col in fixation_metrics],  # Corrected from f"{df}_{col}"
                      var_name="measure", value_name="value")
    # Clean up measure column (remove pos prefix)
    df_long["measure"] = df_long["measure"].str.replace(f"{criterion}_", "")

    # Pivot: make wide format with POS+measure as columns
    df_pivot = df_long.pivot_table(
        columns=[criterion, "measure"],
        values="value"
    ).reset_index(drop=True)

    # Optional: flatten multi-level column names
    df_pivot.columns = [f"{c}_{measure}" for c, measure in df_pivot.columns]

    df_pivot = df_pivot
    return df_pivot


def test_flip_group_to_features():
    # Sample input DataFrame (simulating output from groupby.mean() with renamed columns)
    # Assume criterion is 'pos' and fixation_metrics are ['mean_FF', 'mean_FP']
    sample_df = pd.DataFrame({
        'pos': ['NOUN', 'VERB', 'ADJ'],
        'pos_mean_FF': [200, 250, 180],
        'pos_mean_FP': [300, 350, 280]
    })
    
    criterion = 'pos'
    fixation_metrics = ['mean_FF', 'mean_FP']
    
    # Call the function
    result = flip_group_to_features(sample_df, criterion, fixation_metrics)
    
    # Expected output: Wide format with flattened columns
    # After melt: long format with 'pos', 'measure' ('mean_FF', 'mean_FP'), 'value'
    # After pivot: columns like ('NOUN', 'mean_FF'), ('NOUN', 'mean_FP'), etc.
    # After flattening: 'NOUN_mean_FF', 'NOUN_mean_FP', 'VERB_mean_FF', etc.
    # reset_index() adds an 'index' column (original row indices)
    expected = pd.DataFrame({
        'NOUN_mean_FF': [200.0],  # Only first row has NOUN
        'NOUN_mean_FP': [300.0],
        'VERB_mean_FF': [250.0],
        'VERB_mean_FP': [350.0],
        'ADJ_mean_FF': [180.0],
        'ADJ_mean_FP': [280.0]
    })
    
    # Sort columns for comparison (order might vary)
    result = result.sort_index(axis=0)
    print(result)
    expected = expected.sort_index(axis=1)
    print(expected)
    
    # Assert the result matches expected
    pd.testing.assert_frame_equal(result, expected, check_dtype=False)
    
    # Additional checks
    assert all(col.endswith('_mean_FF') or col.endswith('_mean_FP') or col == 'index' for col in result.columns)



if __name__ == "__main__":
    test_flip_group_to_features()
    print("Test passed!")