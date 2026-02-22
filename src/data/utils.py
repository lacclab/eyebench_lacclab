import warnings
from collections import defaultdict
from functools import partial
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from loguru import logger
from scipy.stats import kurtosis, skew
from torch import normal
import statsmodels.api as sm


from src.configs.constants import (
    DataType,
    EYE_METRICS_NICKNAMES,
    EYE_METRICS_NICKNAMES_INVERTED,
    POS_COL,
    WORD_PROPERTY_COLUMNS,
    SetNames,
    gsf_features,
    numerical_feature_aggregations,
    numerical_fixation_trial_columns,
    numerical_ia_trial_columns,
)
from src.configs.models.base_model import BaseModelArgs


def get_feature_from_list(
    values: list[float | int | float | np.int32 | np.float64] | pd.Series,
    aggregation_function: str,
):
    """
    creates a feature for a list of values (e.g. mean or standard deviation of values in list)
    Args:
        values (list[int | float | np.int32 | np.float64]): list of values
        aggregation_function (str): name of function to be applied to list
    Returns:
        np.float64  | np.nan: aggregated value or np.nan if not possible
    """
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    if np.sum(np.isnan(values)) == len(values):
        return np.nan
    if aggregation_function == 'mean':
        return np.nanmean(values)
    elif aggregation_function == 'std':
        return np.nanstd(values)
    elif aggregation_function == 'median':
        return np.nanmedian(values)
    elif aggregation_function == 'skew':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return skew(not_nan_values)
    elif aggregation_function == 'kurtosis':
        not_nan_values = np.array(values)[~np.isnan(values)]
        return kurtosis(not_nan_values)
    elif aggregation_function == 'max':
        return np.nanmax(values)
    elif aggregation_function == 'min':
        return np.nanmin(values)
    else:
        return np.nan


def load_fold_data(
    fold_index: int,
    base_path: Path,
    folds_folder_name: str,
    data_type: DataType,
    regime_name: SetNames,
    set_name: SetNames,
    query: str | None = None,
) -> pd.DataFrame:
    """
    Load data for a specific fold, data type, regime, and set.

    This method reads a Feather file containing the data for the specified
    fold index, data type, regime name, and set name.

    Args:
        fold_index (int): The index of the fold to load data for.
        base_path (Path): The base path where the data is stored.
        folds_folder_name (str): The name of the folder containing the folds.
        data_type (DataType): The type of data to load (e.g., train, test, etc.).
        regime_name (SetNames): The name of the regime (e.g., validation, training, etc.).
        set_name (SetNames): The name of the set (e.g., train, test, etc.).
        query (str | None): A query string to filter the data. If None, no filtering is applied.
    Returns:
        pd.DataFrame: A DataFrame containing the loaded data.

    Note:
        The file path is currently hardcoded to 'data/OneStop/folds'. This should
        be replaced with a general path when a connection to the server is available.
    """
    df = pd.read_feather(
        base_path
        / folds_folder_name
        / f'fold_{fold_index}'
        / f'{data_type}_{set_name}_{regime_name}.feather'
    )
    if query is not None:
        df = df.query(query)
    for should_be_bool in ['total_skip', 'start_of_line', 'end_of_line']:
        if should_be_bool in df.columns:
            df[should_be_bool] = df[should_be_bool].astype(bool)
    return df


def load_raw_data(
    data_path: Path, usecols: list[str] | None = None, **kwargs
) -> pd.DataFrame:
    if not data_path.exists() or not data_path.is_file():
        raise FileNotFoundError(f'{data_path} does not exist. Please check the path.')

    format_used = ''
    # if the data_path ends with .feather we use feather format
    if data_path.suffix == '.feather':
        data = pd.read_feather(
            data_path,
            columns=usecols,  # noqa: E721
            **kwargs,
        )
        return data

    try:
        data = pd.read_csv(
            data_path,
            encoding='utf-16',
            engine='pyarrow',
            usecols=usecols,
            **kwargs,
        )
        format_used = 'pyarrow with utf-16'
    except UnicodeError:
        data = pd.read_csv(data_path, engine='pyarrow', usecols=usecols, **kwargs)
        format_used = 'pyarrow'
    except ValueError:
        try:
            data = pd.read_csv(data_path, encoding='utf-16', usecols=usecols, **kwargs)
            format_used = 'default engine with utf-16'
        except UnicodeError:
            data = pd.read_csv(data_path, usecols=usecols, **kwargs)
            format_used = 'default engine'

    logger.info(f'Loaded {len(data)} rows from {data_path} using {format_used}.')

    return data


def add_missing_features(
    et_data: pd.DataFrame,
    trial_groupby_columns: list[str],
    mode: DataType,
) -> pd.DataFrame:
    """
    Add and transform features in the given DataFrame.

    This function adds and transforms several features in the DataFrame. It also creates
    new features based on existing ones.

    Args:
        et_data (pd.DataFrame): The input DataFrame. It should have the following columns:
            - ptb_pos
            - is_content_word
            - NEXT_FIX_INTEREST_AREA_INDEX
            - CURRENT_FIX_INTEREST_AREA_INDEX
            - IA_REGRESSION_IN_COUNT
            - IA_REGRESSION_OUT_FULL_COUNT
            - IA_FIXATION_COUNT
        trial_groupby_columns (list): A list of column names to group by when calculating sums.

    Returns:
        pd.DataFrame: The DataFrame with added and transformed features.
        The function creates the following new features:
            - ptb_pos: Transformed from categorical to numerical using a mapping dictionary.
            - is_content_word: Converted to integer type.
            - is_reg: Whether the next fixation interest area index is less than the current one.
            - is_progressive: Whether the next fixation IA index is greater than the current one.
            - is_reg_sum: The sum of is_reg for each group defined by trial_groupby_columns.
            - is_progressive_sum:
                The sum of is_progressive for each group defined by trial_groupby_columns.
            - IA_REGRESSION_IN_COUNT_sum:
                The sum of IA_REGRESSION_IN_COUNT for each group defined by trial_groupby_columns.
            - normalized_outgoing_regression_count:
                The ratio of IA_REGRESSION_OUT_FULL_COUNT to is_reg_sum.
            - normalized_outgoing_progressive_count:
                The ratio of the difference between IA_FIXATION_COUNT and
                IA_REGRESSION_OUT_FULL_COUNT to is_progressive_sum.
            - normalized_incoming_regression_count:
                The ratio of IA_REGRESSION_IN_COUNT to IA_REGRESSION_IN_COUNT_sum.
            # These are used for Syntactic Clusters with
            # Universal Dependencies PoS and Information Clusters [Berzak et al. 2017]
            - LengthCategory:
                The length category of the word based on the word_length column.
            - LengthCategory_normalized_IA_DWELL_TIME:
                IA_DWELL_TIME normalized by the mean IA_DWELL_TIME of the LengthCategory group.
            - POS_normalized_IA_DWELL_TIME:
                IA_DWELL_TIME normalized by the mean IA_DWELL_TIME of the universal_pos group.
            - LengthCategory_normalized_IA_FIRST_FIXATION_DURATION:
                IA_FIRST_FIXATION_DURATION normalized by the mean IA_FIRST_FIXATION_DURATION of the
                LengthCategory group.
            - POS_normalized_IA_FIRST_FIXATION_DURATION:
                IA_FIRST_FIXATION_DURATION normalized by the mean IA_FIRST_FIXATION_DURATION of the
                universal_pos group.
    """
    # Map ptb_pos values to numbers
    value_to_number = {'FUNC': 0, 'NOUN': 1, 'VERB': 2, 'ADJ': 3, 'UNKNOWN': 4}
    et_data['ptb_pos'] = et_data['ptb_pos'].map(value_to_number)

    # Convert is_content_word to integer
    et_data['is_content_word'] = et_data['is_content_word'].astype('Int64')

    # TODO Add reference to a paper for these bins?
    # Define the boundaries of the bins by word length
    bins = [0, 2, 5, 11, np.inf]  # 0-1, 2-4, 5-10, 11+
    # Define the labels for the bins
    labels = [0, 1, 2, 3]
    et_data['LengthCategory'] = pd.cut(
        et_data['word_length'],
        bins=bins,
        labels=labels,
        right=False,
    )

    if mode == DataType.FIXATIONS:
        # Add is_reg and is_progressive features
        et_data['is_reg'] = (
            et_data['NEXT_FIX_INTEREST_AREA_INDEX']
            < et_data['CURRENT_FIX_INTEREST_AREA_INDEX']
        )
        et_data['is_progressive'] = (
            et_data['NEXT_FIX_INTEREST_AREA_INDEX']
            > et_data['CURRENT_FIX_INTEREST_AREA_INDEX']
        )

        # Calculate sums for is_reg, is_progressive, and IA_REGRESSION_IN_COUNT
        grouped_sums = et_data.groupby(trial_groupby_columns)[
            ['is_reg', 'is_progressive', 'IA_REGRESSION_IN_COUNT']
        ].transform('sum')

        # Add sum features
        et_data['is_reg_sum'] = grouped_sums['is_reg']
        et_data['is_progressive_sum'] = grouped_sums['is_progressive']
        et_data['IA_REGRESSION_IN_COUNT_sum'] = grouped_sums['IA_REGRESSION_IN_COUNT']

        # Add normalized count features
        et_data['normalized_outgoing_regression_count'] = (
            et_data['IA_REGRESSION_OUT_FULL_COUNT'] / et_data['is_reg_sum']
        )
        et_data['normalized_outgoing_progressive_count'] = (
            et_data['IA_FIXATION_COUNT'] - et_data['IA_REGRESSION_OUT_FULL_COUNT']
        ) / et_data['is_progressive_sum']  # approximation
        et_data['normalized_incoming_regression_count'] = (
            et_data['IA_REGRESSION_IN_COUNT'] / et_data['IA_REGRESSION_IN_COUNT_sum']
        )
        et_data = et_data.replace([np.inf, -np.inf], 0)
        et_data.fillna(
            {
                'normalized_outgoing_regression_count': 0,
                'normalized_outgoing_progressive_count': 0,
                'normalized_incoming_regression_count': 0,
            },
            inplace=True,
        )

    et_data.fillna(
        {
            'LengthCategory_normalized_IA_DWELL_TIME': 0,
            'universal_pos_normalized_IA_DWELL_TIME': 0,
            'LengthCategory_normalized_IA_FIRST_FIXATION_DURATION': 0,
            'universal_pos_normalized_IA_FIRST_FIXATION_DURATION': 0,
        },
        inplace=True,
    )

    return et_data


def add_missing_categories_and_flatten(
    grouped_gsf_features: pd.DataFrame,
    groupby_fields: list[
        float | int | str | np.int64 | None | np.float64 | pd._libs.missing.NAType
    ],
    groupby_type_: str,
) -> dict[str, int | float | np.float64]:
    """
    Add missing categories and flatten the grouped GSF features.

    Args:
        grouped_gsf_features (pd.DataFrame): The grouped GSF features.
        groupby_fields (list): The fields to group by.
        groupby_type_ (str): The type of grouping.

    Returns:
        dict[str, int | float | np.float64]: The flattened GSF features.
    """
    new_index = (
        grouped_gsf_features.index.union(
            pd.Index(groupby_fields),
        )
        .drop_duplicates()
        .dropna()
    )
    if len(groupby_fields) < len(new_index):
        logger.warning(
            f'Missing categories: {new_index.difference(groupby_fields)} in {
                groupby_type_
            }!',
        )
    grouped_gsf_features = grouped_gsf_features.reindex(
        new_index,
        fill_value=0,
    )
    grouped_df_reset = grouped_gsf_features.reset_index()

    melted_ = grouped_df_reset.melt(
        # Use the first column as the id_vars
        id_vars=grouped_df_reset.columns[0],
        var_name='variable',  # Name of the new variable column
        value_name='value',  # Name of the new value column
    )
    # If you want to add the groupby_type_ to the feature name so have feature names
    melted_['feature_name'] = (
        groupby_type_ + '_' + melted_['index'].astype(str) + '_' + melted_['variable']
    )
    res_df = (
        melted_[['feature_name', 'value']]
        .set_index(
            'feature_name',
        )
        .sort_index()
    )
    # create a dict {feature_name: value}
    res_dict = res_df.to_dict()['value']
    return res_dict
    # return melted_.sort_values(by="variable").value


def get_gaze_entropy_features(
    x_means: np.ndarray,
    y_means: np.ndarray,
    x_dim: int = 2560,
    y_dim: int = 1440,
    patch_size: int = 138,
) -> dict[str, int | float | np.float64]:
    """
    Compute gaze entropy features.

    Args:
        x_means (np.ndarray): The x-coordinates of fixations.
        y_means (np.ndarray): The y-coordinates of fixations.
        x_dim (int, optional): The screen horizontal pixels. Defaults to 2560.
        y_dim (int, optional): The screen vertical pixels. Defaults to 1440.
        patch_size (int, optional): The size of patches to use. Defaults to 138.

    Returns:
        dict[str, int | float | np.float64]: The gaze entropy features.
    """

    # Gaze entropy measures detect alcohol-induced driver impairment - ScienceDirect
    # https://www.sciencedirect.com/science/article/abs/pii/S0376871619302789
    # computes the gaze entropy features
    # params:
    #    x_means: x-coordinates of fixations
    #    y_means: y coordinates of fixations
    #    x_dim: screen horizontal pixels
    #    y_dim: screen vertical pixels
    #    patch_size: size of patches to use
    # Based on https://github.com/aeye-lab/etra-reading-comprehension
    def calc_patch(patch_size: int, mean: np.float64 | np.int64) -> float | int:
        return int(np.floor(mean / patch_size))

    def entropy(value: float) -> float:
        return value * (np.log(value) / np.log(2))

    # dictionary of visited patches
    patch_dict = {}
    # dictionary for patch transitions
    trans_dict = defaultdict(list)
    pre = None
    for i in range(len(x_means)):
        x_mean = x_means[i]
        y_mean = y_means[i]
        patch_x = calc_patch(patch_size, x_mean)
        patch_y = calc_patch(patch_size, y_mean)
        cur_point = f'{str(patch_x)}_{str(patch_y)}'
        if cur_point not in patch_dict:
            patch_dict[cur_point] = 0
        patch_dict[cur_point] += 1
        if pre is not None:
            trans_dict[pre].append(cur_point)
        pre = cur_point

    # stationary gaze entropy
    # SGE
    sge = 0.0
    x_max = int(x_dim / patch_size)
    y_max = int(y_dim / patch_size)
    fix_number = len(x_means)
    for i in range(x_max):
        for j in range(y_max):
            cur_point = f'{str(i)}_{str(j)}'
            if cur_point in patch_dict:
                cur_prop = patch_dict[cur_point] / fix_number
                sge += entropy(cur_prop)
    sge = sge * -1

    # gaze transition entropy
    # GTE
    gte = 0.0
    for patch in trans_dict:
        cur_patch_prop = patch_dict[patch] / fix_number
        cur_destination_list = trans_dict[patch]
        (values, counts) = np.unique(cur_destination_list, return_counts=True)
        inner_sum = 0.0
        for i in range(len(values)):
            cur_count = counts[i]
            cur_prob = cur_count / np.sum(counts)
            cur_entropy = entropy(cur_prob)
            inner_sum += cur_entropy
        gte += cur_patch_prop * inner_sum
    gte *= -1
    return {'fixation_feature_SGE': sge, 'fixation_feature_GTE': gte}


def compute_fixation_trial_level_features(
    trial: pd.DataFrame, groupby_mappings: list[tuple], processed_data_path: Path
) -> dict:
    """
    Compute fixation trial-level features.

    Args:
        trial (pd.DataFrame): The trial data.
        groupby_mappings (list[tuple]): The groupby mappings for categorical features.
        processed_data_path (Path): The path to save the trial level feature names.

    Returns:
        pd.Series: The computed features.
    """

    RF = {}
    for column in numerical_fixation_trial_columns:
        if column in trial.columns:
            for aggregation_method in numerical_feature_aggregations:
                key = 'fix_feature_' + aggregation_method + '_' + column
                val = get_feature_from_list(
                    trial[column].replace('.', np.nan).astype(float),
                    aggregation_method,
                )
                RF.update({key: val})

    ####### David Eyes Only ######
    BEYELSTM = {}
    gaze_features = get_gaze_entropy_features(
        x_means=trial['CURRENT_FIX_X'].values,  # type: ignore
        y_means=trial['CURRENT_FIX_Y'].values,  # type: ignore
    )
    BEYELSTM.update(gaze_features)

    BEYELSTM['total_num_fixations'] = len(trial)
    BEYELSTM['total_num_words'] = (
        trial['TRIAL_IA_COUNT'].drop_duplicates().dropna().values[0]
    )

    ##### David #####
    # Creates
    # 'LengthCategory_normalized_IA_FIRST_FIXATION_DURATION',
    # 'LengthCategory_normalized_IA_DWELL_TIME',
    # 'universal_pos_normalized_IA_DWELL_TIME',
    # 'universal_pos_normalized_IA_FIRST_FIXATION_DURATION',
    for cluster_by in ['LengthCategory', 'universal_pos']:
        # FutureWarning: The default of observed=False is deprecated and will be changed to True
        # in a future version of pandas. Pass observed=False to retain current behavior or
        # observed=True to adopt the future default and silence this warning.
        try:
            grouped_means = trial.groupby(cluster_by, observed=False)[
                ['IA_DWELL_TIME', 'IA_FIRST_FIXATION_DURATION']
            ].transform('mean')
        except IndexError:
            grouped_means['IA_DWELL_TIME'] = np.nan
            grouped_means['IA_FIRST_FIXATION_DURATION'] = np.nan
        for et_measure in ['IA_DWELL_TIME', 'IA_FIRST_FIXATION_DURATION']:
            trial[f'{cluster_by}_normalized_{et_measure}'] = (
                trial[et_measure] / grouped_means[et_measure]
            )
    # No. values in each groupby type:
    # is_content_word 2 (in beyelstm originally 3)
    # ptb_pos 5 (in beyelstm originally 5)
    # entity_type 20 (in beyelstm originally 11)
    # universal_pos 17 (in beyelstm originally 16)
    for groupby_type_, groupby_fields in groupby_mappings:
        # TODO This shouldn't be hardcoded here
        if groupby_type_ == 'ptb_pos':
            value_to_number = {
                'FUNC': 0,
                'NOUN': 1,
                'VERB': 2,
                'ADJ': 3,
                'UNKNOWN': 4,
            }
            trial['ptb_pos'] = trial['ptb_pos'].map(value_to_number)
        grouped_gsf_features = trial.groupby(groupby_type_)[gsf_features].mean()
        melted_gsf_features = add_missing_categories_and_flatten(
            grouped_gsf_features=grouped_gsf_features,
            groupby_fields=groupby_fields,
            groupby_type_=groupby_type_,
        )
        for feature_name, feature_value in melted_gsf_features.items():
            BEYELSTM[feature_name] = feature_value

    SVM = {}
    # mean saccade duration -> mean "NEXT_SAC_DURATION"
    to_compute_features = [
        'NEXT_SAC_DURATION',
        'NEXT_SAC_AVG_VELOCITY',
        'NEXT_SAC_AMPLITUDE',
    ]
    for feature_to_compute in to_compute_features:
        SVM[feature_to_compute + '_mean'] = trial[feature_to_compute].mean()
        SVM[feature_to_compute + '_max'] = trial[feature_to_compute].max()

    # Diane
    LOGISTIC = {}
    LOGISTIC['CURRENT_FIX_DURATION_mean'] = trial['CURRENT_FIX_DURATION'].mean()

    # mean forward saccade length:
    # * "normalized_ID_plus_1" = "normalized_ID" of the next fixation (row)
    trial['normalized_ID_plus_1'] = trial['normalized_ID'].shift(-1)
    # * mean "NEXT_SAC_AMPLITUDE" where "normalized_ID_plus_1" > "normalized_ID"
    forward_saccade_length = trial[
        trial['normalized_ID_plus_1'] > trial['normalized_ID']
    ]['NEXT_SAC_AMPLITUDE'].mean()
    LOGISTIC['forward_saccade_length_mean'] = forward_saccade_length

    # regression rate - backward saccade rate
    # * using "normalized_ID_plus_1" = "normalized_ID" of the next fixation (row)
    # * regression rate - % of rows where "normalized_ID_plus_1" < "normalized_ID"
    regression_rate = (
        trial['normalized_ID_plus_1'] < trial['normalized_ID']
    ).sum() / len(trial)
    LOGISTIC['regression_rate'] = regression_rate

    
    FIXATION_METRICS = {}

    # FIXATION_METRICS['regression_rate_fixations'] = regression_rate  # TODO check if this makes sense

    features_dict = {
        'RF': RF,
        'BEYELSTM': BEYELSTM,
        'SVM': SVM,
        'LOGISTIC': LOGISTIC,
        'FIXATION_METRICS': FIXATION_METRICS,
    }
    save_feature_names_if_do_not_exist(
        features_dict=features_dict,
        csv_path=processed_data_path / 'fixation_trial_level_feature_keys.csv',
        mode=DataType.FIXATIONS,
    )

    return RF | BEYELSTM | SVM | LOGISTIC | FIXATION_METRICS

def create_s_clusters_dict_inner(fix_met_trial_df:pd.DataFrame, fixation_metrics:list[str], criterion:str=POS_COL, normalize_RS:bool=True) -> dict:
    """
    Create syntactic clusters dictionary inner function
    """

    # TODO: decide if we want to return -1 or np.nan or something else if there's no criterion column in the data (or add criterion column manually)
    if criterion not in fix_met_trial_df.columns or fix_met_trial_df[criterion].isna().all():
        return {f"{criterion}_{col}": -1.0 for col in fixation_metrics}
    # Group by the criterion and calculate the average fixation times
    cluster_means = fix_met_trial_df.groupby(criterion)[fixation_metrics].mean().reset_index()
    total_means = fix_met_trial_df[fixation_metrics].mean().to_frame().T

    if normalize_RS:
        cluster_means[fixation_metrics] = cluster_means[fixation_metrics].div(total_means.iloc[0])
    
    # Rename the columns to include the criterion name
    cluster_means = cluster_means.rename(columns={col: f"{criterion}_{col}" for col in fixation_metrics})
    cluster_means = cluster_means[[criterion] + [ f"{criterion}_{col}" for col in fixation_metrics]]
    
    cluster_means = flip_group_to_features(cluster_means, criterion, fixation_metrics)

    cluster_means_dict = cluster_means.to_dict(orient='records')[0]

    return cluster_means_dict

def create_s_clusters_dict(ia_trial_df:pd.DataFrame, fixation_metrics:list[str], criterion:str=POS_COL, normalize_RS:bool=True) -> dict:
    """
    Create syntactic clusters dictionary
    """
    fix_met_trial_df = ia_trial_df.rename(columns=EYE_METRICS_NICKNAMES_INVERTED)
    return create_s_clusters_dict_inner(fix_met_trial_df, fixation_metrics, criterion, normalize_RS)


def flip_group_to_features(df: pd.DataFrame, criterion:str, fixation_metrics:list) -> pd.DataFrame:
    """

    """
    df_long = df.melt(id_vars=[criterion], 
                  value_vars=[f"{criterion}_{col}" for col in fixation_metrics],
                  var_name="measure", value_name="value")

    # # Clean up measure column (remove pos prefix) #TODO decide if we want this - this removes the name of pos col from the name of the feature 
    # df_long["measure"] = df_long["measure"].str.replace(f"{criterion}_", "")

    # Pivot: make wide format with POS+measure as columns
    df_pivot = df_long.pivot_table(
        columns=[criterion, "measure"],
        values="value"
    )

    # Optional: flatten multi-level column names
    df_pivot.columns = [f"{c}_{measure}" for c, measure in df_pivot.columns]
    df_pivot = df_pivot.reset_index(drop=True)
    return df_pivot

def find_normalizing_factor(fixation_trial_df:pd.DataFrame, fixation_metrics:list):
    norm_df = fixation_trial_df[fixation_metrics].mean()
    return norm_df.to_dict()


def calc_reading_speed(trial: pd.DataFrame) -> float:
    num_of_words = len(trial)
    return (
        num_of_words / (trial[~trial['PARAGRAPH_RT'].isna()]['PARAGRAPH_RT'].values[0])
    )


def find_wp_coefs(ia_trial_df: pd.DataFrame, fixation_metrics:list[str], normalize_RS:bool=True) -> dict:
    fix_met_trial_df = ia_trial_df.rename(columns=EYE_METRICS_NICKNAMES_INVERTED)
    return find_wp_coefs_inner(fix_met_trial_df, fixation_metrics, normalize_RS)


def find_wp_coefs_inner(subject_df: pd.DataFrame, fixation_metrics:list[str], normalize_RS:bool=True) -> dict:
    
    # Initialize a dictionary to hold coefficients
    coefs = {}
    
    total_means = subject_df[fixation_metrics].mean().to_frame().T

    subject_df = subject_df.copy()
    if normalize_RS:
        subject_df[fixation_metrics] = subject_df[fixation_metrics].div(total_means.iloc[0])
    
    # Iterate over each fixation measure
    for eye_metric in fixation_metrics:
        # TODO: decide how to handle nans
        if eye_metric not in subject_df.columns or subject_df[eye_metric].isna().all() or not all(col in subject_df.columns for col in WORD_PROPERTY_COLUMNS) or subject_df[WORD_PROPERTY_COLUMNS].isna().any().any():
            coefs[f"{eye_metric}_intercept"] = -1.0
            for wp in WORD_PROPERTY_COLUMNS:
                coefs[f"{eye_metric}_{wp}_coef"] = -1.0
            continue

        # Prepare the regression model
        X = subject_df[WORD_PROPERTY_COLUMNS]
        # guardrails for if the eye_metric is not in the data or is all NaN
        if eye_metric == "SKIP": # TODO: check if this is even a thing or that skip is irrelevant
            y = subject_df[eye_metric].astype(int)  # Convert SKIP to binary (0/1) for logistic regression
        else:
            y = subject_df[eye_metric]
        
        # Fit the model
        if eye_metric == "SKIP":
            model = sm.Logit(y, sm.add_constant(X)).fit()
        else:
            model = sm.OLS(y, sm.add_constant(X)).fit()
        
        # Store the coefficients in the dictionary
        coefs[f"{eye_metric}_intercept"] = model.params['const']
        for wp in WORD_PROPERTY_COLUMNS:
            coefs[f"{eye_metric}_{wp}_coef"] = model.params[wp]
    
    return coefs


def compute_ia_trial_level_features(
    trial: pd.DataFrame, processed_data_path: Path
) -> dict:
    """
    Compute IA trial-level features.

    Args:
        trial (pd.DataFrame): The trial data.

    Returns:
        pd.Series: The computed features.
    """

    RF = {}
    for column in numerical_ia_trial_columns:
        if column in trial.columns:
            for aggregation_method in numerical_feature_aggregations:
                RF.update(
                    {
                        'ia_feature_'
                        + aggregation_method
                        + '_'
                        + column: get_feature_from_list(
                            trial[column].astype(float), aggregation_method
                        )
                    }
                )

    SVM = {}
    SVM.update(
        {
            'skip_rate': trial['total_skip'].mean(),
            'num_of_fixations': trial['IA_FIXATION_COUNT'].sum(),
            'mean_TFD': trial['IA_DWELL_TIME'].mean(),
        }
    )

    # Diane
    # https://tmalsburg.github.io/MeziereEtAl2021MS.pdf
    # go-past time (i.e., the sum of fixations on a word up to when it
    # is exited to its right, including all regressions to the left of the word
    LOGISTIC = {}
    LOGISTIC.update(
        {
            'first_pass_skip_rate': trial['IA_SKIP'].mean(),
            'mean_FFD': trial['IA_FIRST_FIXATION_DURATION'].mean(),
            'mean_GD': trial['IA_FIRST_RUN_DWELL_TIME'].mean(),
            'mean_TFD': trial['IA_DWELL_TIME'].mean(),
            'mean_go_past_time': trial['IA_SELECTIVE_REGRESSION_PATH_DURATION'].mean(),
            'reading_speed': calc_reading_speed(trial),
        }
    )

    READING_SPEED = {'reading_speed': calc_reading_speed(trial)}


    # IA_REGRESSION_OUT_FULL_COUNT is Number of times word was exited to a lower IA_ID (to the left in English).
    # IA_REGRESSION_OUT_FULL_COUNT is Number of times word was exited to a lower IA_ID (to the left in English).

    first_fix_prog = trial.loc[(trial["IA_FIRST_FIX_PROGRESSIVE"]==1)].reset_index(drop=True)
    first_pass_regressions = first_fix_prog['IA_REGRESSION_OUT_COUNT'].mean() # might have a problem that "IA_REGRESSION_OUT_COUNT" isnt in the data
    total_regression_words = trial['IA_REGRESSION_OUT_FULL_COUNT'].mean()

    FIXATION_METRICS = {}
    FIXATION_METRICS.update(
        {
            'skip_rate': trial['total_skip'].mean(),
            #'first_pass_skip_rate': trial['IA_SKIP'].mean(), # TODO: add to features?
            'mean_FF': trial['IA_FIRST_FIXATION_DURATION'].mean(), # first fixation duration
            'mean_FP': trial['IA_FIRST_RUN_DWELL_TIME'].mean(), # first pass duration
            'mean_TF': trial['IA_DWELL_TIME'].mean(),  # total fixation duration
            'first_pass_regression_rate': first_pass_regressions, 
            # 'total_regression_rate': total_regression_words, # TODO: add to features?
        }
    )
     
    fixation_metrics_clusters = [
        "FF", "FP", "TF", "RP", "SKIP" #TODO: think about changing or adding more
    ]
    S_CLUSTERS = create_s_clusters_dict(trial, fixation_metrics_clusters, POS_COL, normalize_RS=True)
    S_CLUSTERS_NO_NORM = create_s_clusters_dict(trial, fixation_metrics_clusters, POS_COL, normalize_RS=False)

    fixation_metrics_wp_coefs = [
        "FF", "FP", "TF", "RP" #TODO: think about changing or adding more
    ]
    WP_COEFS = find_wp_coefs(trial, fixation_metrics_wp_coefs, normalize_RS=True)
    WP_COEFS_NO_NORM = find_wp_coefs(trial, fixation_metrics_wp_coefs, normalize_RS=False)
    WP_COEFS_NO_INTERCEPT = {key:value for key, value in WP_COEFS.items() if 'intercept' not in key}
    WP_COEFS_NO_NORM_NO_INTERCEPT = {key:value for key, value in WP_COEFS_NO_NORM.items() if 'intercept' not in key}


    features_dict = {
        'RF': RF,
        'SVM': SVM,
        'LOGISTIC': LOGISTIC,
        'READING_SPEED': READING_SPEED,
        'FIXATION_METRICS': FIXATION_METRICS,
        'S_CLUSTERS': S_CLUSTERS,
        'S_CLUSTERS_NO_NORM': S_CLUSTERS_NO_NORM,
        'WP_COEFS': WP_COEFS, 
        'WP_COEFS_NO_NORM': WP_COEFS_NO_NORM,
        'WP_COEFS_NO_INTERCEPT': WP_COEFS_NO_INTERCEPT,
        'WP_COEFS_NO_NORM_NO_INTERCEPT': WP_COEFS_NO_NORM_NO_INTERCEPT
    }
    
    save_feature_names_if_do_not_exist(
        features_dict=features_dict,
        csv_path=processed_data_path / 'ia_trial_level_feature_keys.csv',
        mode=DataType.IA,
    )


    return RF | SVM | LOGISTIC | READING_SPEED | FIXATION_METRICS | \
           S_CLUSTERS | S_CLUSTERS_NO_NORM \
           | WP_COEFS | WP_COEFS_NO_NORM | WP_COEFS_NO_INTERCEPT | WP_COEFS_NO_NORM_NO_INTERCEPT


def save_feature_names_if_do_not_exist(
    features_dict, csv_path: Path, mode: DataType
) -> None:
    """
    Save feature names to a CSV file if they do not already exist.
    """
    global_field_name = f'{mode}_TRIAL_LEVEL_FEATURE_KEYS_SAVED'
    if global_field_name not in globals():
        feature_rows = []
        for feature_type, feature_dict in features_dict.items():
            for feature_name in feature_dict.keys():
                feature_rows.append(
                    {'feature_name': feature_name, 'feature_type': feature_type}
                )
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(feature_rows).to_csv(csv_path, index=False)
        logger.info(f'Saved feature names to {csv_path}')
        globals()[global_field_name] = True


def compute_trial_level_features(
    raw_fixation_data: pd.DataFrame | None,
    raw_ia_data: pd.DataFrame,
    trial_groupby_columns: list[str],
    processed_data_path: Path,
) -> pd.DataFrame:
    """
    Compute trial-level features in parallel.

    Args:
        raw_fixation_data (pd.DataFrame | None): The raw fixation data.
        raw_ia_data (pd.DataFrame): The raw IA data.
        trial_groupby_columns (list[str]): The columns to group by for trials.
        processed_data_path (Path): The path to save the trial level feature names.

    Returns:
        pd.DataFrame: The computed trial-level features.
    """
    groupby_mappings = [
        (feature_name, list(raw_ia_data[feature_name].unique()))
        for feature_name in [
            'is_content_word',
            'ptb_pos',
            'entity_type',
            'universal_pos',
        ]
    ]
    logger.info(
        f'Computing trial level features for {raw_ia_data.shape[0]} trials with {groupby_mappings} groupby mappings'
    )
    ia_partial = partial(
        compute_ia_trial_level_features,
        processed_data_path=processed_data_path,
    )
    logger.info('This might take a couple of minutes, please be patient...')
    logger.info(
        f' Number of trial groups in ia: {len(raw_ia_data.groupby(trial_groupby_columns).groups)}'
    )
    ia_trial_features = raw_ia_data.groupby(trial_groupby_columns).apply(ia_partial)  # type: ignore
    ia_trial_features = pd.DataFrame(
        list(ia_trial_features), index=ia_trial_features.index
    ).fillna(0)

    if raw_fixation_data is not None:
        logger.info(
            f'Computing fixation trial level features for {raw_fixation_data.shape[0]} trials with {groupby_mappings} groupby mappings'
        )
        logger.info('This might take a couple of minutes, please be patient...')
        fixation_partial = partial(
            compute_fixation_trial_level_features,
            groupby_mappings=groupby_mappings,
            processed_data_path=processed_data_path,
        )
        logger.info(
            f'Number of trial groups in fix: {len(raw_fixation_data.groupby(trial_groupby_columns).groups)}'
        )
        logger.info('This might take a couple of minutes, please be patient...')
        fixation_trial_features = raw_fixation_data.groupby(
            trial_groupby_columns
        ).apply(fixation_partial)  # type: ignore
        fixation_trial_features = pd.DataFrame(
            list(fixation_trial_features), index=fixation_trial_features.index
        ).fillna(0)
        trial_level_features = pd.concat(
            [fixation_trial_features, ia_trial_features],
            axis=1,
        )
    else:
        trial_level_features = ia_trial_features

    return trial_level_features


def replace_missing_values(
    data_dict: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    for col in BaseModelArgs().fixation_features:
        if col not in data_dict[DataType.FIXATIONS].columns:
            logger.warning(f'Adding missing column {col} to fixation data')
            data_dict[DataType.FIXATIONS][col] = 0

    for col in BaseModelArgs().ia_features:
        if col not in data_dict[DataType.IA].columns:
            logger.warning(f'Adding missing column {col} to IA data')
            data_dict[DataType.IA][col] = 0
    return data_dict
