# creates a basic columns cleaner tool for continual use
from functools import wraps
import pandas as pd

def pcard_columns_tweaker(method): 
    """
    decorator around a method that returns a dataframe with 
    clean column labels

    Parameters: 
        - method: the method to wrap. 
    Returns: 
        - a decorated method or function. 
    """

    @wraps(method) # keep original docstring for help()
    def method_wrapper(self, *args, **kwargs): 
        df = method(self, *args, **kwargs)

        #clean the columnn labels
        return (df
                .rename(columns=lambda col_names: col_names
                        .replace('.', ' ')
                        .replace('/', ' ')
                        .replace('-', ' ')
                        .replace('  ', '')
                        .replace(' ', '_')
                        .lower()
                        .strip()
                        )
                )
    return method_wrapper

@pcard_columns_tweaker 
def pcard_normal_tweaker(df):
    """
    cleans the column labels for the vendor file
    """
    
    return df

@pcard_columns_tweaker
def pcard_spend_tweaker(_df:pd.DataFrame) -> pd.DataFrame: 
    """
    cleans the column labels for the pcard spend file
    """

    return (_df
            .iloc[:, 2:10]
            .dropna(subset=['Employee ID'])
            .drop_duplicates(subset=['Employee ID'], keep='first')
            .astype({
                'Employee ID':int, 
                'Open Date':'datetime64[ns]', 
                'Credit Limit':int
                    })
            .assign(final_four = lambda df: df['Card Number'].str[-4:], 
                    f_name = lambda df: df['First Name'].str.lower(),
                    l_name = lambda df: df['Last Name'].str.lower(),
                    f_name_card = lambda df: df.f_name + df.final_four
                   )
           )

def pcard_trans_df(_df:pd.DataFrame):
    """
    manipulates the pcard transaction file
    """
    
    return (_df
            .assign(
                f_name_card = lambda df: 
                df.acc_account_name.str.split(expand=True)[0].str.lower() 
                + df.acc_account_number.str[-4:])
           )

def validate_df(columns, instance_method=True): 
    """
    Decorator that raises a `ValueError` if input isn't a pandas
    `DataFrame` or doesn't contain the proper columns. 

    *Note the `DataFrame`must be the first positional argument 
    passed to this method.*

    Parameters:
        - columns: A set of required column names.
                   For example, {'acc_account_name', 'acc_account_number'}.
        - instance_method: Whether or not the item being decorated is
                           an instance method. Pass `False` to decorate
                           static methods and functions.

    Returns:
        A decorated method or function.
    """

    def method_wrapper(method): 
        @wraps(method)
        def validate_wrapper(self, *args, **kwargs): 
            # functions and static methods don't pass self
            # so self is the first positional argument in that case
            df = (self, *args)[0 if not instance_method else 1]
            if not isinstance(df, pd.DataFrame):
                raise ValueError('Must pass in a pandas `DataFrame`')
            if columns.difference(df.columns):
                raise ValueError(
                    f'DataFrame must contain the following columns: {columns}'
                )
            return method(self, *args, **kwargs)
        return validate_wrapper
    return method_wrapper

@validate_df(columns={'acc_account_name', # pcard holder name 
                          'acc_account_number', # pcard holder card number
                          'fin_transaction_date', # financial transaction date - used to filter dates
                          'mch_merchant_name', # provides merchant name name in dirty form
                          'mch_city_name', # provides either a city or phone number for the vendor
                          'mch_state_province', # state of vendor location 
                          'mch_country_code', # country of vendor location
                          'mch_merchant_category_code_(mcc)', # purchases MCC code
                          'fin_transaction_amount', # purchases transaction amount
                          'gbl_item_description', # 1 of 2 item description columns
                          'pur_item_description' # 2 of 2 item description columns
                          }, instance_method=False) # columns=set() argument does not check for specfiic column names
def pcard_trans_dtypes_converter(_df:pd.DataFrame): 
    """
    Converts provided pandas dataframe columns from object to the appropriate data types. 

    Listed below are the labels: 
    'acc_account_name', # pcard holder name 
    'acc_account_number', # pcard holder card number
    'fin_transaction_date', # financial transaction date - used to filter dates
    'mch_merchant_name', # provides merchant name name in dirty form
    'mch_city_name', # provides either a city or phone number for the vendor
    'mch_state_province', # state of vendor location 
    'mch_country_code', # country of vendor location
    'mch_merchant_category_code_(mcc)', # purchases MCC code
    'fin_transaction_amount', # purchases transaction amount
    'gbl_item_description', # 1 of 2 item description columns
    'pur_item_description' # 2 of 2 item description columns

    Parameters: 
        - _df: the pcard transaction dataframe 

    Returns: 
        - pd.DataFrame
    """

    return (_df
            .assign(fin_transaction_amount = lambda _df: 
                    _df.fin_transaction_amount.str.replace(',', '').astype('float64')
            )
            .astype({'fin_transaction_date': 'datetime64[ns]'})
            )

@validate_df(columns={'fin_transaction_date'}, instance_method=False)
def report_pcard_trans_metrics(_df: pd.DataFrame): 
    """
    Prints out the following metrics based on pcard transaction file: 

        - Timeframe (start and end date) for the df_pcard_trans_revised dataframe based on fin_transaction_date.
        - Total number of pcard transactions by rows.
        - Total number of individual unique pcards.
        - Total number of unique vendors in original dirty file.
        - All unique MCC codes and total transactions and spend in dollars per MCC code.
        - Total number of transactions and spend in dollars.
        - Max, min and average spend in dollars.

    Parameters: 
        - _df: the pcard transaction dataframe used to create metrics

    Returns: 
        - None
    """

    temp_df = pcard_trans_dtypes_converter(_df.query('f_name_card != "metra1234"'))
    max_financial_transaction_date = temp_df.fin_transaction_date.max().date()
    min_financial_transaction_date = temp_df.fin_transaction_date.min().date()
    total_trans_rows = len(temp_df)
    total_unique_cards = len(temp_df.acc_account_number.unique())
    total_vendors = len(temp_df.mch_merchant_name.unique())
    total_spend_amount = temp_df.fin_transaction_amount.sum()
    max_dollar, min_dollar, average_dollar = (temp_df.fin_transaction_amount.max(), 
                                              temp_df.fin_transaction_amount.min(),
                                              temp_df.fin_transaction_amount.mean(),
                                              )

    print(f'Max date: \t\t{max_financial_transaction_date}')
    print(f'Min date: \t\t{min_financial_transaction_date}')
    print(f'Total rows: \t\t{total_trans_rows:,.0f}')
    print(f'Unique cards: \t\t{total_unique_cards}')
    print(f'Unique vendors: \t{total_vendors:,.0f}')
    print(f'Total amount: \t\t{total_spend_amount:,.2f}')
    print(f'Maximum amount: \t{max_dollar:,.2f}')
    print(f'Minimum amount: \t{min_dollar:,.2f}')
    print(f'Average amount: \t{average_dollar:,.2f}')

    return None 
    
    