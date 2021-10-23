import os
import sys
import re

import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import MiniBatchKMeans


def get_attr_by_type(var_type,data_dict):
    '''
    function to get attribtues of certain type from data_dict

    input
    var_type: variable type which is supposed to be selected
    data_dict: data_dictionary which contains types and Attributes

    returns variables as array
    '''
    if var_type == 'categorical':
        cat_vars =  data_dict.loc[ (data_dict['Type'].isin(['cat','cat?'])) &  (data_dict['Treatement'] != 'Drop'),'Attribute'].values
        return cat_vars
    if var_type == 'numeric':
        num_vars = data_dict.loc[ (data_dict['Type'].isin(['num','num?'])) &  (data_dict['Treatement'] != 'Drop'),'Attribute'].values
        return num_vars
    if var_type == 'binary':
        bin_vars = data_dict.loc[ (data_dict['Type'].isin(['binary'])) &  (data_dict['Treatement'] != 'Drop'),'Attribute'].values
        return bin_vars
    else:
        return 'Type not defined'

def define_pipeline(data_dict):
    '''
    function to define variable transformation pipeline

    input
    data_dict: data_dictionary which contains types and Attributes

    returns pipeline object which has to be fitted
    '''

    categorical_transformer = Pipeline(steps = [
         ('imputer',SimpleImputer(strategy = 'constant',fill_value = 99)),
         ('one_hot',OneHotEncoder(handle_unknown='ignore'))
    ])

    numerical_transformer = Pipeline(steps = [
         ('imputer',SimpleImputer(strategy = 'median')),
         ('scale',StandardScaler())
    ])

    binary_transformer = Pipeline(steps = [
         ('imputer',SimpleImputer(strategy = 'most_frequent'))
    ])

    cat_vars = get_attr_by_type('categorical',data_dict)
    num_vars = get_attr_by_type('numeric',data_dict)
    bin_vars = get_attr_by_type('binary',data_dict)

    col_trans = ColumnTransformer([
            ('categorical', categorical_transformer,cat_vars),
            ('numeric', numerical_transformer,num_vars),
            ('binary', binary_transformer,bin_vars),
    ])

    return col_trans


# function returns WSS score for k values from 1 to kmax
def calculate_WSS(data, kmax,verbose=True):
    '''
    function to calculate WSS for Kmeans

    input:
    data: flatfile
    kmax: max number of clusters

    returns array with wss for different cluster numbers
    '''

    # as in https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb'
    sse = []
    for k in range(1, kmax+1):
        kmeans = MiniBatchKMeans(n_clusters = k).fit(data)
        centroids = kmeans.cluster_centers_
        pred_clusters = kmeans.predict(data)
        curr_sse = 0

        # calculate square of Euclidean distance of each point from its cluster center and add to current WSS
        for i in range(len(data)):
          curr_center = centroids[pred_clusters[i]]
          curr_sse += (data[i, 0] - curr_center[0]) ** 2 + (data[i, 1] - curr_center[1]) ** 2

        sse.append(curr_sse)
        if verbose == True:
            print('SSE for K = {} is {}'.format(k,curr_sse))

    return sse


def get_cluster_comp_df(cluster_pipeline,all_vars,num_vars):

    # get values of cluster_means, inverse_transform pca and save in dataframe
    cluster_centers = cluster_pipeline.named_steps['kmeans'].cluster_centers_
    cluster_centers = np.round(cluster_pipeline.named_steps['pca'].inverse_transform(cluster_centers),2)
    cl_comp_df = pd.DataFrame(cluster_centers)

    # rename attributes with real names
    cl_comp_df.columns = all_vars

    # for numeric variables also the standardization has to be inversed
    # separate numeric and not numeric
    cl_comp_df_numeric = cl_comp_df.loc[:,num_vars]
    cl_comp_df_not_numeric = cl_comp_df.loc[:,~cl_comp_df.columns.isin(num_vars)]

    # inverse transform numeric
    numeric_transformed = cluster_pipeline.named_steps['transform'].named_transformers_['numeric']['scale'].inverse_transform(cl_comp_df_numeric)

    # transform numeric back to dataframe
    numeric_transformed = pd.DataFrame(numeric_transformed)
    numeric_transformed.columns = num_vars

    # concat both numeric and nonnumeric together
    cl_comp_df = pd.concat([numeric_transformed,cl_comp_df_not_numeric],axis=1)

    # reshape such that every attribute is a row and all clusters are columns
    cl_comp_df = cl_comp_df.reset_index().rename(
        columns={'index':'Cluster'}).melt(
        id_vars = 'Cluster')

    cl_comp_df = cl_comp_df.rename(
        columns={'variable':'Attribute'}).pivot(
        index='Attribute', columns='Cluster', values='value')

    return cl_comp_df
