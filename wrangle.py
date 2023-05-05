### IMPORTS 

# Imports
from env import host, user, password
import os

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer

np.random.seed(42)

### FUNCTIONS 

def get_db_url(db_name, user=user, host=host, password=password):
    '''
    get_db_url accepts a database name, username, hostname, password 
    and returns a url connection string formatted to work with codeup's 
    sql database.
    Default values from env.py are provided for user, host, and password.
    '''
    return f'mysql+pymysql://{user}:{password}@{host}/{db_name}'

# generic function to get a sql pull into a dataframe
def get_mysql_data(sql_query, database):
    """
    This function will:
    - take in a sql query and a database (both strings)
    - create a connection url to mySQL database
    - return a df of the given query, connection_url combo
    """
    url = get_db_url(database)
    return pd.read_sql(sql_query, url)    

def get_csv_export_url(g_sheet_url):
    '''
    This function will
    - take in a string that is a url of a google sheet
      of the form "https://docs.google.com ... /edit#gid=12345..."
    - return a string that can be used with pd.read_csv
    '''
    csv_url = g_sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
    return csv_url

# defining a function to get zillow data from either a cached csv or the Codeup MySQL server
def get_zillow_data(sql_query= """
                    SELECT bedroomcnt, bathroomcnt, calculatedfinishedsquarefeet, taxvaluedollarcnt, yearbuilt, taxamount, fips
                    FROM properties_2017
                    JOIN propertylandusetype USING (propertylandusetypeid)
                    WHERE propertylandusedesc = 'Single Family Residential'
                    """
                    , filename="zillow.csv"):
    
    """
    This function will:
    -input 2 strings: sql_query, filename 
        default query selects specific columns from zillow database per acq/prep exercise instructions
        default filename "zillow.csv"
    - check the current directory for filename (csv) existence
      - return df from that filename if it exists
    - If csv doesn't exist:
      - create a df of the sql_query
      - write df to csv
      - return that df
    """
    if os.path.isfile(filename):
        df = pd.read_csv(filename)
        print ("csv file found and read")
        return df
    else:
        url = get_db_url('zillow')
        df = pd.read_sql(sql_query, url)
        df.to_csv(filename, index=False)
        print ("csv file not found, data read from sql query, csv created")
        return df

# defining overall function to acquire and clean up zillow data
def wrangle_zillow():
    """
    This function will acquire and preps specific column data from zillow database. 
    Specifically it will:
    - 1. get data from database
    - 2. rename the columns to something more useful
    - 3. drops the nulls
    - 4. changes the datatype to int for all but one column
    
    - returns a df that looks like this:
        - bedroom_cnt - int
        - bathroom_cnt - float
        - square_feet - int
        - tax_value_cnt - int
        - year_built - int
        - tax_amount - int
        - fips - int
    """

    # first get the data from csv or sql
    df = get_zillow_data()
    
    #rename columns to something less unwieldy
    df.columns = ['bedroom_cnt', 'bathroom_cnt', 'square_feet', 'tax_value_cnt', 'year_built', 'tax_amount', 'fips']
    
    # decided to drop all nulls since it was < 1% of data
    df = df.dropna()
    
    # most columns can/should be integers; exception was bathroom_cnt which I left as a float
    for col in df.columns [df.columns != 'bathroom_cnt']:
        df[col] = df[col].astype(int)
    
    # fips really should be a categorical, like a string
    df.fips = df.fips.astype(str)
    
    return df


def split_function(df, target_var):
    """
    This function will
    - take in a dataframe (df) and a string (target_var)
    - split the dataframe into 3 data frames: train (60%), validate (20%), test (20%)
    -   while stratifying on the target_var
    - And finally return the three dataframes in order: train, validate, test
    """
    train, test = train_test_split(df, random_state=42, test_size=.2, stratify=df[target_var])
    
    train, validate = train_test_split(train, random_state=42, test_size=.25, stratify=train[target_var])

    print(f'Prepared df: {df.shape}')
    print()
    print(f'Train: {train.shape}')
    print(f'Validate: {validate.shape}')
    print(f'Test: {test.shape}')
    
    return train, validate, test
