# Sloth D3M Wrapper (Strategic Labeling Over Time Heuristics)
Wrapper of the KMeans clustering primitive into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base Sloth library can be found here: https://github.com/NewKnowledge/sloth

## Install

pip3 install -e git+https://github.com/NewKnowledge/TimeSeriesD3MWrappers.git#egg=TimeSeriesD3MWrapper --process-dependency-links

## Output
The output is a DataFrame containing a single column where each entry is the associated series' cluster number.

## Available Functions

#### set_training_data

Sets primitive's training data. The inputs are a numpy ndarray of size (number_of_time_series, time_series_length) containing training time series. There are no outputs.

#### fit

Fits clustering algorithm using training data from set_training_data and hyperparameters. There are no inputs or outputs.

#### produce
Produce primitive's best guess for the cluster number of each series. The input is a pandas frame where each row is a series. The output is a dataframe containing a single column where each entry is the associated series' cluster number.

# Shallot D3M Wrapper (SHApelet Learning Over Time-Series)
Wrapper of the Shallot Shapelet learning primitive into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base Sloth library (which contains the Shapelet class and other time series methods) can be found here: https://github.com/NewKnowledge/sloth

## Install

pip3 install -e git+https://github.com/NewKnowledge/TimeSeriesD3MWrappers.git#egg=TimeSeriesD3MWrapper --process-dependency-links

## Output
The output is a numpy ndarray containing a predicted class for each of the input time series.

## Available Functions

#### set_training_data

Sets primitive's training data. The inputs are a numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing training time series
and a numpy ndarray of size (number_time_series,) containing classes of training time series. There are no outputs.

#### fit

Fits Shapelet classifier using training data from set_training_data and hyperparameters. There are no inputs or outputs.

#### produce

Produce primitive's classifications for new time series data The input is a numpy ndarray of size (number_of_time_series, time_series_length, dimension) containing new time series. The output is a numpy ndarray containing a predicted class for each of the input time series.

# Parrot D3M Wrapper (Predictive AutoRegressive Results over Time)
Wrapper of the Parrot ARIMA primitive into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base Sloth library (which also contains other methods that can be called on time series data) can be found here: https://github.com/NewKnowledge/sloth

## Install

pip3 install -e git+https://github.com/NewKnowledge/TimeSeriesD3MWrappers.git#egg=TimeSeriesD3MWrapper --process-dependency-links

## Output
 The output is a list of length 'n_periods' that contains a prediction for each of 'n_periods' future time periods.

## Available Functions

#### set_training_data

Set's primitives training data. The input is a pandas data frame that contains training data in two columns. The first column contains time series indices (preferably in datetime format) and the second column contains time series values. There are no outputs. 

#### fit

Fits ARIMA model using trianing data from set_training_data and hyperparameters. There are no inputs or outputs. 

#### produce
Produce the primitive's prediction for future time series data. The output is a list of length 'n_periods' that contains a prediction for each of 'n_periods' future time periods. 'n_periods' is a hyperparameter that must be set before making the prediction.

# Kanine D3M Wrapper (K appointed nearest interesting neighbors)
Wrapper of the KNN time series classification primitive into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base Sloth library (which also contains other time series methods) can be found here: https://github.com/NewKnowledge/sloth

## Install

pip3 install -e git+https://github.com/NewKnowledge/TimeSeriesD3MWrappers.git#egg=TimeSeriesD3MWrapper --process-dependency-links

## Output
The output is a numpy ndarray containing a predicted class for each of the input time series

## Available Functions

#### set_training_data

Sets primitive's training data. The inputs are a numpy ndarray of size (number_of_time_series, time_series_length) containing training time series. The outputs are a numpy ndarray of size (number_time_series,) containing classes of training time series.

#### fit

Fits KNN model using training data from set_training_data and hyperparameters. There are no inputs or outputs.

#### produce

Produce primitive's classifications for new time series data The input is a numpy ndarray of size (number_of_time_series, time_series_length) containing new time series. The output is a numpy ndarray containing a predicted class for each of the input time series.

# Hdbscan D3M Wrapper (Hierarchical Density-Based Spatial Clustering of Applications with Noise)
Wrapper of the DBSCAN and HDBSCAN time series clustering primitives into D3M infrastructure. All code is written in Python 3.5 and must be run in 3.5 or greater. 

The base Sloth library (which also contains other time series methods) can be found here: https://github.com/NewKnowledge/sloth

## Install

pip3 install -e git+https://github.com/NewKnowledge/TimeSeriesD3MWrappers.git#egg=TimeSeriesD3MWrapper --process-dependency-links

## Output
The output is a numpy ndarray containing a cluster label for each of the input time series

## Available Functions

#### produce

Produce a cluster label for each of the input time series. The inputs are a numpy ndarray of size (number_of_time_series, time_series_length) containing new time series. The output is a numpy ndarray containing a cluster label for each of the input time series.