# arvato_capstone

Code to complete the Udacity Nanodegree Capstone project from Arvato. Block article can be found here: https://matzeknop92.medium.com/udacity-capstone-arvato-project-9e46919defb9

The structure of the project is the following:

notebooks: folder contains the notebooks used to complete the tasks
  - data_exploration: data exploration, cleaning and transforming
  - customer_segmentation: contains the code to do the transformation of classification.ipynb
  - classification: code to complete the classification part of the project
  - (experiment_with_pipeline: a little bit of experementing with the pipeline function of sklearn)

src: Contains all the functions written to complete the project
  - import_module.py: functions used to import raw data
  - etl.py: Functions to clean and transform data
  - pipe.py: Functions to create pipeline to transform data used in customer_segmentation as well as in classification

data/
  00_raw_data: raw unprocessed data
  01_preprocessed: processed data
  02_models: models outputted and also predictions
  99_plots: plots saved for written blog

The necessary packages can be found within the requirements.txt file

I want to thank Arvato for providing us with this nice dataset!
