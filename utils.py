"""Quick utility functions."""

from IPython import embed
import numpy as np


def apply_across_df_level(df, levels, func=np.mean):
    """Average out a level from a dataframe.

    This is a convenience function to quickly average away a set
    of levels in a dataframe. The dataframe will be grouped by all
    other levels, and then averaged.

    Parameters
    ----------
    df : DataFrame
        The dataframe in question
    levels : list of strings
        The names of the levels to average out

    Returns
    -------
    df : DataFrame
        The input dataframe with levels averaged out
    """
    if not isinstance(levels, (list, tuple)):
        raise ValueError('level names must be in a list')
    count_levels = filter(lambda a: a not in levels, df.index.names)
    print('Retaining levels: {0}'.format(count_levels))
    df = df.groupby(level=count_levels)
    df = df.agg(func)
    return df


def query_from_dicts(dict_list):
    '''
    Create a series of string queries from a dictionary.

    This allows you to create a query to be used with pandas .query or .eval
    using dictionaries for quick slicing.

    Parameters
    ----------
    dict_list : list of dict
        The dictionaries to use to build queries. Queries will be constructed
        using key
    '''
    if not isinstance(dict_list, (list, np.ndarray)):
        dict_list = [dict_list]
    qu_list = []
    for item in dict_list:
        join_list = []
        for key, val in item.iteritems():
            if isinstance(val, str):
                val = [val]
            join_str = key + ' in {0}'.format(list(val))
            join_list.append(join_str)
        qu_list.append(join_list)
    queries = [' and '.join(items) for items in qu_list]
    if len(queries) == 1:
        queries = queries[0]
    return queries
