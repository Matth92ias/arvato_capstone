# arvato_capstone

Code to complete the Udacity Nanodegree Capstone project from Arvato. Block article
can be found here:

The structure of the project is the following:

notebooks: Contains the notebooks used to complete the tasks
  data_exploration: data exploration, cleaning and transforming
  customer_segmentation: contains the code to do the transformation of
  classification.ipynb

src: Contains all the functions written to complete the project
  import_module.py: functions used to import raw data
  etl.py: Functions to clean and transform data
  pipe.py: Functions to create pipeline to transform data used in customer_segmentation
            as well as in classification

data/
  00_raw_data: raw unprocessed data
  01_preprocessed: processed data
  02_models: models outputted and also predictions
  99_plots: plots saved for written blog

required packages:
- os
- sys
- re
- numpy >= 1.19.4
- pandas >= 1.1.4
- matplotlib >= 3.3.4
- seaborn >= 0.11.1
- sklearn >= 0.0
- joblib >= 1.0.1
