import pandas as pd
import numpy as np

def import_raw_data(name,path):
    '''
    import data and clean some of the columns

    name: name of the file to be imported
    path: path to data directory
    '''

    spec_dtypes = {'CAMEO_DEU_2015': 'object','CAMEO_DEUG_2015': 'object','CAMEO_INTL_2015':'object'}
    data = pd.read_csv(os.path.join(path,'00_raw_data/',name), sep=';',dtype = spec_dtypes,error_bad_lines=False,quoting=3 )

    # object columns are imported with brackets which have to be removed
    object_cols = data.select_dtypes(include=['object']).columns.to_list()
    for col in object_cols:
        data[col] = data[col].str.strip('"')

    # convert numeric columns to numeric
    for col in ['CAMEO_DEUG_2015','CAMEO_INTL_2015']:
        data[col]= pd.to_numeric(np.where(data[col].isin(['X','XX']),-1, data[col]))
    data['CAMEO_DEU_2015'] = np.where(data['CAMEO_DEU_2015'].isin(['X','XX']),np.nan, data['CAMEO_DEU_2015'])

    return customers
