import requests, io
import pandas as pd
from datetime import datetime


def get_data(url):
    """Use the request module to download raw data from Github and load it into a pandas DataFrame.

    Args:
        url (str): Url to the raw data on GitHub designed to be either confirmed  cases or deaths.

    Returns:
        DataFrame: Resulting pandas dataframe of cases/deaths data.
    """
    r = requests.get(url).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8'))).dropna().set_index(
        'UID')  # drop missing values as they are insignificant in the scope of the project
    df.index.name = None
    return df


def first_date(df):
    """Gets the column index of the first date (which uses the format 'm/d/y').

    Args:
        df (DataFram): Pandas dataframe containing the data of US.

    Returns:
        [int]: The index of the first date column.
    """
    for idx, header in enumerate(df):
        try:
            datetime.strptime(header, "%m/%d/%y")
            break
        except:
            pass
    return idx


def get_state(df, state, combine=False):
    """Slice the DataFrame to only include data from a specified state.

    Args:
        df (DataFrame): Pandas dataframe containing US cases/deaths data.
        state (str, list): The desired state(s) to slice the  data to, a list can be input if multiple state's data is desired.
        combine (bool, optional): Group by state and sum the results for summary or use in plotting.

    Returns:
        DataFrame: Resulting dataframe containing the sliced data.
    """
    if isinstance(state, list):
        df = df[df['Province_State'].isin(state)]
    else:
        df = df[df['Province_State'] == state]
    if combine:
        df = df.groupby('Province_State').sum()
    return df


def select_dates(df, start=None, end=None):
    """Allows the user to specify a start and/or end date to select data from.

    Args:
        df (DataFrame): dataframe containing the desired US cases/deaths data.
        start (str, optional): Remove all data for dates before this (use 'd/m/y' format). Defaults to None.
        end (str, optional): Remove all data for dates after this (use 'd/m/y' format). Defaults to None.

    Returns:
        [DataFrame]: Resulting dataframe containing the sliced data.
    """
    first = first_date(df)

    headers = {head: idx for idx, head in enumerate(df.columns)}
    if start is not None and end is not None:
        df = df.drop(columns=df.columns[first:headers[start]]).drop(columns=df.columns[headers[end]:])
    elif start is not None:
        df = df.drop(columns=df.columns[first:headers[start]])
    elif end is not None:
        df = df.drop(columns=df.columns[headers[end]:])
    return df


def daily_cases(df, period=1):
    """Takes the difference between columns to calculate the number of daily cases/deaths.

    Args: df (DataFrame): Pandas dataframe congaing US cases/deaths the total cases/deaths at a given date. period (
    int, optional): The number of days to calculate the rolling average over. Use default of one for calculating
    daily cases.

    Returns:
        DataFrame: Resulting dataframe containing the number of daily cases/deaths.
    """
    first = first_date(df)

    daily = df.iloc[:, 11:].diff(axis=1, periods=period).dropna(
        axis=1) / period  # drop first column as the diff yields NaN
    df.update(daily)
    return df.drop(columns=df.columns[first])  # drop first date column since it does not have sufficient data


def split_data(df):
    """Splits the generated dataframe into two, the first with the info on state and county and the second with the
    numerical case/death data. UID index is consistent across both to relate the two.

    Args:
        df (DataFrame): Pandas dataframe containing the US.

    Returns:
        tup: Tuple containing the text info in index 0 and date info at index 1.
    """
    first = first_date(df)
    text_data = df.iloc[:, :first]
    date_data = df.iloc[:, first:]
    date_data.columns = pd.to_datetime(date_data.columns)
    return text_data, date_data


from matplotlib import pyplot as plt


def plot_bar(df, num_days='1W'):
    """Plot data in a bar chart using various Matplotlib and Pandas functionality.

    Args:
        df (DataFrame): Pandas dataframe containing US case date.
        num_days (str, optional): Length of time to use in the resampling. Use number followed by letter corresponding
        to date part (D, W, M, or Y). Defaults to '1W'.

    Raises:
        RuntimeError: raised if the initial dataframe contains too many rows of data likely resulting from not grouping
        data. The result will be overly computationally intensive and not plot well.

    Returns:
        plot: Returns a Matplotlib plot of the selected data.
    """
    if max(df.nunique()) > 10:
        raise RuntimeError('Attempting to plot a DataFrame with more than 10 rows is not recommended')
    text, date = split_data(df)
    date = date.T.resample(num_days).sum()

    fig, ax = plt.subplots(figsize=(10, 10))
    date.plot.bar(ax=ax)
    labels = [dt.strftime('%b %d') for dt in date.index]
    ax.set_xticklabels(labels, rotation=-45)
    plt.close()
    return fig


def plot_line(df):
    """Plot data in a line chart.

    Args:
        df (DataFrame): Pandas dataframe containing US case data.

    Raises:
        RuntimeError: raised if the initial dataframe contains too many rows of data likely resulting from not grouping
         data. The result will be overly computationally intensive and not plot well.

    Returns:
        plot: Returns a Matplotlib plot of the selected data.
    """
    if max(df.nunique()) > 10:
        raise RuntimeError('Attempting to plot a DataFrame with more than 10 rows is not recommended')
    idx = first_date(df)
    plot = df.iloc[:, idx:].T.plot()
    return plot
