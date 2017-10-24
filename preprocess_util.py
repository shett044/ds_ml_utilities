__author__ = 'sshetty'
import pandas as pd
from sklearn import preprocessing
import copy
import numpy as np


def norm_range_0_1(x):
    return (x - x.min()) / (x.max() - x.min())


def fill_missing_timeseries(df, minDate, maxDate, fillSales=1, columns=['part', 'plant', 'date', 'sales'], freq='M', formatDate='%Y%m'):
    """
    Make sure the columns are in this range data.columns = ['part', 'plant', 'date','sales']
    This formats only monthly data.
    Fills missing months.
    """
    print "Data Records for each row \n", df.describe(include="all").ix["count"].T
    df.columns = columns
    time_df = pd.DataFrame()
    time_df["date"] = pd.date_range(minDate, (maxDate), freq=freq).map(
        lambda x: int(x.strftime(formatDate)))
    timeseries_pergrain = len(time_df["date"])
    print "Number of Time series records per grain should be ", timeseries_pergrain
    time_df['key'] = 0
    df["key"] = 0
    time_df = pd.merge(df.loc[:, ['part', 'plant', 'key']].drop_duplicates(
    ), time_df, on='key', how='outer').drop('key', 1)
    tmp = pd.merge(df.drop('key', 1), time_df, how='right')
    tmp["sales"] = tmp["sales"].fillna(fillSales)
    assert len(data.part.unique()) * timeseries_pergrain == tmp.shape[0]
    return tmp

def add_lag_items(df, col, lags=[1], dropna=False):
    """
    Add Lagged items
    """
    for i in lags:
        df['lag_%d' % i] = df[col].shift(i)
    if dropna:
        return df.loc[df['lag_%d' % i].notnull()]
    return df


def add_mov_avg_items(ts_df, col, window=[1], dropna=False):
    """
    Add Moving average items
    """
    for i in window:
        # Center of weight for i weeks
        ma_com = 0.5 * (i-1)
        ts_df['ma_%d' % i] = pd.ewma(ts_df[col], com=ma_com, ignore_na=True).shift(i) #
    if dropna:
        return ts_df.loc[ts_df['ma_%d' % i].notnull()]
    return ts_df


def over_sample(train, target, os_model):
    """
    Over samples data according to the os_model
    Creates a ratio with one maj_class and other minority classes.
    :param train: Data frame/ np Array
    :param target: Target column
    :param os_model: Check UnbalancedSampling package
    :return train, target
    """
    import copy
    tmp_model = copy.copy(os_model)
    ratio = {}
    from scipy import stats
    maj_class = stats.mode(pd.Series(target)).mode[0]
    train = np.array(train)
    target = np.array(target)
    final_x = np.array(train[target == maj_class])
    final_y = np.array(target[target == maj_class])
    print final_x.shape, final_y.shape
    for min_class in np.delete(np.unique(target), maj_class):
        os_model = copy.copy(tmp_model)
        ratio[min_class] = float(np.count_nonzero(
            target == maj_class)) / float(np.count_nonzero(target == min_class))
        os_model.ratio = ratio[min_class] / 2
        class_filter = (target == min_class) | (target == maj_class)
        print train[class_filter, ].shape, target[class_filter].shape, pd.Series(target[class_filter]).value_counts(), ratio
        print "Head"
        print train[class_filter, ], target[class_filter]
        osx, osy = os_model.fit_transform(
            train[class_filter, ], target[class_filter])
        print "Ratio for {} and {} is {}".format(min_class, maj_class, ratio[min_class])
        print "Oversampling {} and {}".format(min_class, maj_class)
        print pd.Series(osy).value_counts()
        print osx.shape, final_x.shape
        final_x = np.concatenate([final_x, osx[osy == min_class, ]], axis=0)
        final_y = np.concatenate([final_y, osy[osy == min_class]], axis=0)
    print "Train shape is {}".format(final_x.shape)
    print "Target Distribution"
    print pd.Series(final_y).value_counts()
    print "--------------------------------------"
    print "Actual  -----{}".format(list(pd.Series(target).value_counts() / pd.Series(target).value_counts().sum()))
    print "Oversampling------{}".format(list(pd.Series(final_y).value_counts() / pd.Series(final_y).value_counts().sum()))
    print final_x.shape
    return pd.DataFrame(final_x), pd.Series(final_y)


def create_qbins(ser, bins, labels=None):
    """
    Creates binning on a series depeding on the number of quantile of bins. Also gives you the bins along with it.
    output: (bins_result, bin_lable)
    """
    min_ser = np.min(ser) - 1
    max_ser = np.max(ser)
    return pd.qcut(ser, np.linspace(0, 1, bins), labels=labels, retbins=True)


def create_bins(ser, bins, labels=None):
    """
    Creates binning on a series depeding on the number of bins. Also gives you the bins along with it.
    output: (bins_result, bin_lable)
    """
    min_ser = np.min(ser) - 1
    max_ser = np.max(ser)
    return pd.cut(ser, np.linspace(min_ser, max_ser, bins), labels=None, retbins=True)


def count_nan(ser):
    """
    Count number of NaN in the column/List
    :param ser:
    :return:
    """
    if isinstance(ser, list):
        return sum(ser is None)

    return sum(ser.isnull())


def create_dummies(df, columns_factor=None):
    """
    Factorizes the df
    :param df
    :param columns
    :return: Factorized object columns
    """
    if columns_factor is None:
        columns_factor = df.select_dtypes(include=['object']).columns
    return pd.get_dummies(df, columns=columns_factor)


def normalize(df, cols=None):
    """
    Normalize the dataframe and as per columns
    """
    from sklearn.preprocessing import normalize
    if cols is None:
        cols = df.columns
    # Normalize numeric object
    num_cols = df[cols].select_dtypes(exclude=['object', 'boolean']).columns
    df[num_cols] = df[num_cols].pipe(normalize)
    # Normalize object
    return create_dummies(df)


def label_to_numeric(series):
    """
    Converts label to numeric and gives model to retrieve it back.

    :param series:
    :return: (model, transformed number)
    """
    le = preprocessing.LabelEncoder()
    le.fit(series.unique())
    return copy.copy(le), le.transform(series)


def kFold_validate(train, target, folds, model, err_func, cv=None):
    """
    Perform K fold cross validations
    :param train: Training data
    :param target: Predictor column
    :param folds: K fold value
    :param model: Model to train on
    :param cv: cv is already present. Default is None
    :param err_func: Error function
    :return: List of results from the error function
    """

    from sklearn import cross_validation
    # Simple K-Fold cross validation. 10 folds.
    if cv is None:
        cv = cross_validation.KFold(len(train), n_folds=folds, indices=False)
    results = []
    # "Error_function" can be replaced by the error function of your analysis
    for traincv, testcv in cv:
        #             probas = model.fit(train[traincv], target[traincv]).predict_proba(train[testcv])
        tmp = model.fit(train[traincv], target[traincv])
        prediction = tmp.predict(train[testcv])
        results.append(err_func(target[testcv], prediction))
    return results, model
