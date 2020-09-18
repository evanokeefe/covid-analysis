import requests, io
import pandas as pd

def get_data(url):
    """Use the request module to download raw data from Github and load it into a pandas DataFrame

    Args:
        url (str): Url to the raw data on GitHub designed to be either confirmed  cases or deaths

    Returns:
        DataFrame: Resulting pandas dataframe of cases/deaths data
    """
    r = requests.get(url).content
    df = pd.read_csv(io.StringIO(r.decode('utf-8'))).dropna() #drop missing values as the scope of the project does not depend on indivdual cases
    return df

def get_state(df, state):
    """Slice the DataFrame to only include data from a specified state

    Args:
        df (DataFrame): Pandas datafram containing US cases/deaths data
        state (str, list): The desired state(s) to slice the  data to, a list can be input if multiple state's data is desired

    Returns:
        DataFrame: Resulting dataframe containing the sliced data
    """
    if isinstance(state, list):
        df = df[df['Province_State'].isin(state)]
    else:
        df = df[df['Province_State'] == state]
    return df

def select_dates(df, start=None, end=None):
    """Allows the user to specify a start and/or end date to select data from

    Args:
        df (DataFrame): dataframe containing the desired US cases/deaths data
        start (str, optional): Remove all data for dates before this (use 'd/m/y' format). Defaults to None.
        end (str, optional): Remove all data for dates after this (use 'd/m/y' format). Defaults to None.

    Returns:
        [DataFrame]: Resulting dataframe containing the sliced data
    """
    headers = {head: idx for idx, head in enumerate(df.columns)}
    if start != None:
        df.drop(columns=df.columns[11:headers[start]], inplace=True)
    if end != None:
        df.drop(columns=df.columns[headers[end]:], inplace=True)
    return df
