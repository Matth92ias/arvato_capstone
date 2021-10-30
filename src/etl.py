import pandas as pd
import numpy as np
import os

def prepare_data_dict(column_names,data_dir):
    '''
    prepare data dictionary which is saved as Excel and used to determine how variables are treated

    input:
    column_names: column names in dataframe
    data_dir: name of directory to import given Excels

    returns a data_dictionary which is later saved as Excel

    '''

    # Import necessary Excel files
    attr_values_path = os.path.join(data_dir,'00_raw_data/','DIAS Attributes - Values 2017.xlsx')
    attr_values = pd.read_excel(attr_values_path, index_col=None,skiprows=1,usecols="B:E")
    attr_descrpt_path = os.path.join(data_dir,'00_raw_data/','DIAS Information Levels - Attributes 2017.xlsx')
    attr_descrpt = pd.read_excel(attr_descrpt_path, index_col=None,skiprows=1,usecols="B:E")

    # Rename some variables and create indicator to see directly if variable is found in Excel
    attr_descrpt['Indicator'] = 1
    attr_descrpt['Attribute'] = attr_descrpt['Attribute'].str.replace('_RZ','')
    attr_descrpt['Information level'] = attr_descrpt['Information level'].ffill()

    # create a dataframe with translation of missing values
    attr_values['Attribute'] = attr_values['Attribute'].str.replace('_RZ','')
    attr_values['Attribute'] = attr_values['Attribute'].ffill()
    # Filter only missing values and select necessary columns
    missing_descrpt = attr_values[attr_values['Meaning'].isin(['unknown','unknown / no main age detectable','no transactions known'])].reset_index(drop=True)
    missing_descrpt = missing_descrpt[['Attribute','Value']]
    missing_descrpt = missing_descrpt.rename(columns = {'Value':'Missing Values'})

    # join information to the columns of dataframe
    data_dict = pd.DataFrame({'Attribute' : column_names})
    data_dict = data_dict.merge(missing_descrpt,on= 'Attribute',how='left')
    data_dict = data_dict.merge(attr_descrpt,on= 'Attribute',how='left')

    return data_dict


def get_missing_df(data):
    '''
    get amount of missing values and percentage in dataframe

    input:
    data: dataframe either customers or azdiad
    data_dir: directory where the data is saved. DIAS Attributes - Values 2017.xlsx is supposed to be saved in subfolder 00_raw_data

    returns dataframe with missing statistics
    '''
    missing_df = pd.DataFrame({
     'Missing': data.isnull().sum(),
     'PercentMissing' : data.isnull().sum() / data.isnull().count()
    })

    missing_df = missing_df.reset_index().rename(columns={'index': 'Attribute'})
    return missing_df


def map_missing_dict(data,data_dict):
    '''
    function to map values coded with numeric values into the np.nan (as described in DIAS Attributes)

    input
    data: dataframe either customers or azdiad
    data_dir: directory where the data is saved. DIAS Attributes - Values 2017.xlsx is supposed to be saved in subfolder 00_raw_data

    dataframe where missing values are substiuted with np.nan
    '''
    missing_descrpt = data_dict[pd.notnull(data_dict['Missing Values'])]

    missing_dict = {}
    for r in range(missing_descrpt.shape[0]):
      attr_name = missing_descrpt.iloc[r]['Attribute']
      values = missing_descrpt.iloc[r]['Missing Values']

      # if several, then separate and convert to int
      if type(values) is str:
        values = values.replace(' ','').split(',')
        values = list(map(int,values))
      else:
        values = [values]
      missing_dict[attr_name] = values

    # Substitute values in df with missing values or print that value is not
    # in dataframe
    for key,value_list in missing_dict.items():
        if key in data.columns.values:
            data[key] = data[key].map(lambda x: np.nan if x in value_list else x)
        else:
            print("Attribute {} is not available in DataFrame.".format(key))

    return data

def engineer_cameo_intl(data):
    '''
    function to engineer CAMEO_INTL_2015

    input:
    data: the dataframe which is transformed

    returns data
    '''
    data['CAMEO_INTL_2015'] = data['CAMEO_INTL_2015'].map(lambda x: np.nan if x == -1 else x)
    data['CAMEO_INTL_Economic'] = pd.to_numeric(data.CAMEO_INTL_2015.map(lambda x: np.nan if pd.isnull(x) else str(x)[0]))
    data['CAMEO_INTL_Family'] = pd.to_numeric(data.CAMEO_INTL_2015.map(lambda x: np.nan if pd.isnull(x) else str(x)[1]))

    return data

def mapping_praegende_jj_mainstream(x):
    '''
    function to map PRAEGENDE_JUGENDJAHRE mainstream yes no to variable

    input:
    data: the dataframe which is transformed

    returns mapped values
    '''
    if pd.isnull(x):
        return np.nan
    if x in [1,3,5,8,10,12,14]:
        return 1
    else:
        return 0

def mapping_praegende_jj_years(x):
    '''
    function to map PRAEGENDE_JUGENDJAHRE years to values

    input:
    data: the dataframe which is transformed

    returns mapped values
    '''
    if pd.isnull(x):
        return np.nan
    if x in [1,2]:
        return 1
    if x in [3,4]:
        return 2
    if x in [5,6,7]:
        return 3
    if x in [8,9]:
        return 4
    if x in [11,12,13]:
        return 5
    if x in [14,15]:
        return 6
    else:
        return np.nan

def engineer_praegende_jj(data):
    '''
    function to engineer PRAEGENDE_JUGENDJAHRE

    input:
    data: the dataframe which is transformed

    returns data
    '''

    data['PRAEGENDE_JUGENDJAHRE_MAINSTREAM'] = data.PRAEGENDE_JUGENDJAHRE.map( lambda x : mapping_praegende_jj_mainstream(x) )
    data['PRAEGENDE_JUGENDJAHRE_YEARS'] = data.PRAEGENDE_JUGENDJAHRE.map( lambda x : mapping_praegende_jj_years(x) )

    return data


def complete_data_dict(data,data_dict):
    '''
    function to complete the data dictionary and calculate a treatement for every
    attribute

    input:
    data: the dataframe azdiad which is used to calculate statistics of the data
    data_dict: data_dict which was manually changed after creating template

    returns data_dict which is used to clean data in the future
    '''

    # drop cols more needed to find meaning of attributes
    data_dict = data_dict.drop(columns = ['Information level','Description','Additional notes','Indicator'])

    # create data dictionary for created variables
    new_vars = pd.DataFrame({
        'Attribute': ['PRAEGENDE_JUGENDJAHRE_MAINSTREAM','PRAEGENDE_JUGENDJAHRE_YEARS','CAMEO_INTL_Economic','CAMEO_INTL_Family'],
        'Type': ['cat','cat','cat','cat'],
        'Missing Values': [np.nan,np.nan,np.nan,np.nan],
        'Missing Values': [0,0,0,0],
        'PercentMissing': [0,0,0,0]
    })

    # add to data_dict and remove dropped variables
    data_dict = pd.concat([data_dict,new_vars],axis=0)

    # reset_index
    data_dict = data_dict.reset_index(drop=True)

    # For all the categorical variables calculate how many different levels (needed to be able to drop attributes
    # with too many missing values
    cat_vars = data_dict.loc[data_dict['Type'].isin(['cat','cat?'])]['Attribute'].values

    unique_values = []
    for col in cat_vars:
        unique_values.append(azdiad[col].nunique())

    cat_unique_values_df = pd.DataFrame({
        'Attribute': cat_vars ,
        'UniqueValueCount': unique_values
    })

    data_dict = data_dict.merge(cat_unique_values_df,on='Attribute',how='left')

    def calculate_treatement(row):
        if row['PercentMissing'] > .25:
            return 'Drop'

        if row['Type'] in ['year','unknown','date'] or row['Attribute'] in ['PRAEGENDE_JUGENDJAHRE','CAMEO_INTL_2015','D19_LETZTER_KAUF_BRANCHE']:
            return 'Drop'

        if row['UniqueValueCount'] > 15 and row['Type'] in ['cat','cat?']:
            return 'Drop'

        else:
            return 'Keep'

    data_dict['Treatement'] = data_dict.apply( lambda row : calculate_treatement(row), axis = 1)

    return data_dict


def clean_data(data,data_dir,keep_response=False):
    '''
    function to clean data and prepare for applying the algorithms

    input:
    data: dataframe to be cleaned (azdiad or customer)
    data_dir: directory where data is stored as data_dict is saved in 01_preprocessed
    and has to be imported for cleaning

    returns cleaned datasets

    '''

    # Import data dictionary
    print("The data dictionary is imported.")
    data_dict_path = os.path.join(data_dir,'01_preprocessed/','data_dictionary_full.xlsx')
    data_dict = pd.read_excel(data_dict_path, index_col=0)

    # Map missing values
    print("The data dictionary is used to map the missing values.")
    data = map_missing_dict(data,data_dict)

    # engineer some variables
    print("The functions engineer_cameo_intl and engineer_praegende_jj are used to engineer additional variables")
    data = engineer_cameo_intl(data)
    data = engineer_praegende_jj(data)

    # change Ost_West_KZ to numeric
    data['OST_WEST_KZ'] = data['OST_WEST_KZ'].map(lambda x: 1 if x == 'W' else 0)

    # drop columns which are not in dictionary
    not_in_dict = [col for col in data if col not in data_dict['Attribute'].values]

    if 'RESPONSE' in not_in_dict and keep_response == True:
        not_in_dict.remove('RESPONSE')
    print("The following attributes are dropped because they are not in the data dictionary: {}".format(not_in_dict))
    data = data.drop(columns = not_in_dict,axis=1)

    # drop if too many missing, too many levels or other reasons
    cols_to_drop = data_dict.loc[data_dict['Treatement'] == 'Drop','Attribute'].values
    print("The following attributes are dropped because they have too many missings or too many levels: {}".format(cols_to_drop))
    data = data.drop(columns=cols_to_drop,axis=1)

    return data
