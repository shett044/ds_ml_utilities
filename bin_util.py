def generate_bin_interval(df, bin_col, count_col = None, bin_interval=5, bin_threshold=100):
    """
    series: Generate bins of that series
    """
    # Generating equal bin_interval sized bins
    df['bins'] = df[bin_col].apply(lambda x: bin_interval*(int(x/bin_interval)+1))
    # Adding bin_threshold as a stopping criteria
    stopping_val = bin_threshold + bin_interval
    df.loc[df[bin_col]>bin_threshold, "bins"] = stopping_val
    # Creating bin count based on count criteria
    if count_col:
        tmp = (
            df
            .groupby("bins")[count_col]
            .apply(pd.Series.nunique)
            .sort_index().reset_index(name='%s_Count' % count_col)
        )
    else:
        tmp = (
            df
            .groupby("bins").size()
            .sort_index().reset_index(name='Count')
        )
    # Generating intervals
    tmp['startbin'] = tmp['bins'] - bin_interval
    tmp['bins'] = pd.Series(tmp[['startbin','bins']].astype(str).values.tolist()).str.join('-')
    # Replacing last value with "threshold +" value
    tmp.loc[tmp["bins"] == tmp.iloc[-1]['bins'], "bins"] = "%.2f+"%(bin_threshold)
    return tmp.drop('startbin',1).set_index('bins')