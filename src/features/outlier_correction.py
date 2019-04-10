import pandas as pd
import numpy as np

def flag_normal_outliers(series, n_sd):
    """ Flags entries within a series as an outlier based
        on their position in a normal distribution.

        Note: this function assumes a normal distribution.

        :param series:      Series to flag outliers with.
        :param n_sd:        Number of standard deviations
                                accepted within the distribution.

        :return: boolean series flagging outliers.
    """

    m, sd = series.agg(['mean', 'std']).values

    outliers = (series < m - n_sd * sd) | (series > m + n_sd * sd)

    return outliers