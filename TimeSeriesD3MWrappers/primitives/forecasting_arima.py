import sys
import os.path
import numpy as np
import pandas
import typing

import statsmodels.api as sm
#from Sloth.predict import Arima
from sklearn.preprocessing import OneHotEncoder

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame, List
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame 

import logging
__author__ = 'Distil'
__version__ = '1.0.3'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):
    datetime_index = hyperparams.Set(
        elements=hyperparams.Hyperparameter[int](-1),
        default=(),
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],  
        description='if multiple datetime indices exist, this HP specifies which to apply to training data. If \
            None, the primitive assumes there is only one datetime index. This HP can also specify multiple indices \
            which should be concatenated to form datetime_index')
    datetime_index_unit = hyperparams.Hyperparameter[typing.Union[str, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='unit of the datetime column if datetime column is integer or float')
    filter_index_one = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='top-level index of column in input dataset that contain unique identifiers of different time series')
    filter_index_two = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='second-level index of column in input dataset that contain unique identifiers of different time series')
    n_periods = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize, 
        default = 61, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
       description='number of periods to predict')
    interval = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='interval with which to sample future predictions')
    specific_intervals = hyperparams.Hyperparameter[typing.Union[typing.List[typing.List[int]], None]](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=None,
        description='defines specific prediction intervals if  different time series require different \
            intervals for output predictions')
    datetime_interval_exception = hyperparams.Hyperparameter[typing.Union[str, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='to handle different prediction intervals (stock market dataset). \
            If this HP is set, primitive will just make next forecast for this datetime value \
            (not multiple forecasts at multiple intervals')
    seasonal = hyperparams.UniformBool(default = True, semantic_types = [
       'https://metadata.datadrivendiscovery.org/types/ControlParameter'],
       description="whether to perform ARIMA prediction with seasonal component")
    seasonal_differencing = hyperparams.UniformInt(lower = 1, upper = 365, default = 1, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='period of seasonal differencing to use in ARIMA perdiction')
    pass

class Parrot(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    '''
        Primitive that applies an ARIMA forecasting model to time series data. The AR and MA terms
        of the ARIMA model are automatically selected and stationarity is induced before fitting 
        the model.
    
        Training inputs: D3M dataset with training time series observations and a time series index
                         column
        Outputs: D3M dataset with predicted observations for a length of 'n_periods' in the future
    '''
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "d473d487-2c32-49b2-98b5-a2b48571e07c",
        'version': __version__,
        'name': "parrot",
        # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
        'keywords': ['Time Series'],
        'source': {
            'name': __author__,
            'contact': __contact__,
            'uris': [
                # Unstructured URIs.
                "https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers",
            ],
        },
        # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
        # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
        # install a Python package first to be even able to run setup.py of another package. Or you have
        # a dependency which is not on PyPi.
        #  'installation': [
        #      {
        #     'type': metadata_base.PrimitiveInstallationType.PIP,
        #     'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
        #         git_commit=utils.current_git_commit(os.path.dirname(__file__)),
        #      ),
        # }],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_forecasting.arima.Parrot',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.AUTOREGRESSIVE_INTEGRATED_MOVING_AVERAGE,
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._params = {}
        self._X_train = None
        self._mins = None
        self._lag_order = None
        self._values = None 
        self._fits = None
        self._final_logs = None
        self._cat_indices = None
        self._encoders = None
        self._unique_index = True
        self.filter_idx_one = None
        self.filter_idx = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """
        Fits ARIMA model using training data from set_training_data and hyperparameters
        """

        # log transformation for standardization, difference, drop NAs
        self._mins = [year.values.min() if year.values.min() < 0 else 0 for year in self._X_train]
        self._values = [year.apply(lambda x: x - min + 1) for year, min in zip(self._X_train, self._mins)]
        self._values = [vals.values for vals in self._values]
        models = [[Arima(self.hyperparams['seasonal'], self.hyperparams['seasonal_differencing']) \
                    for i in range(vals.shape[1])] for vals in self._values]
        self._fits = []
        for vals, model_list, original in zip(self._values, models, self._X_train):
            fits = []
            for model, i in zip(model_list, range(len(model_list))):
                X_train = pandas.Series(data = vals[:,i].reshape((-1,)), index = original.index[:vals.shape[0]]) 
                model.fit(X_train)
                fits.append(model)
            self._fits.append(fits)
        return CallResult(None)

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params:Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """
        Set primitive's training data

        Parameters
        ----------
        inputs : pandas data frame containing training data where first column contains dates and second column contains values
        
        """

        # set datetime index
        times = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Time') + \
                inputs.metadata.get_columns_with_semantic_type('http://schema.org/DateTime')
        times = list(set(times))
        if len(self.hyperparams['datetime_index']) == 0:
            if len(times) == 0:
                raise ValueError("There are no indices marked as datetime values.")
            elif len(times) > 1:
                raise ValueError("There are multiple indices marked as datetime values. You must specify which index to use")
            else:
                time_index = inputs.iloc[:,times[0]]
        elif len(self.hyperparams['datetime_index']) > 1:
            time_index = ''
            for idx in self.hyperparams['datetime_index']:
                time_index = time_index + ' ' + inputs.iloc[:,idx].astype(str)
        else:
            if self.hyperparams['datetime_index'][0] not in times:
                raise ValueError("The index you provided is not marked as a datetime value.")
            else:
                time_index = inputs.iloc[:,self.hyperparams['datetime_index']]
        inputs['temp_time_index'] = pandas.to_datetime(time_index, unit = self.hyperparams['datetime_index_unit'])
        
        # mark key and categorical variables
        key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        cat = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/CategoricalData')
        categories = cat.copy()
        
        self.filter_idx_one = None
        self.filter_idx = None
        # convert categorical variables to 1-hot encoded
        if self.hyperparams['filter_index_one'] is not None:
            categories.remove(self.hyperparams['filter_index_one'])
            self.filter_idx_one = list(inputs)[self.hyperparams['filter_index_one']]
        if self.hyperparams['filter_index_two'] is not None:
            categories.remove(self.hyperparams['filter_index_two'])
            self.filter_idx = list(inputs)[self.hyperparams['filter_index_two']]
        self._cat_indices = []
        self._encoders = []
        for c in categories:
            encoder = OneHotEncoder(handle_unknown='ignore')
            self._encoders.append(encoder)
            encoder.fit(inputs.iloc[:,c].values.reshape(-1,1))
            inputs[list(inputs)[c] + '_' + encoder.categories_[0]] = pandas.DataFrame(encoder.transform(inputs.iloc[:,c].values.reshape(-1,1)).toarray())
            self._cat_indices.append(np.arange(inputs.shape[1] - len(encoder.categories_[0]), inputs.shape[1]))
        # create unique_index column if other indices
        unique_index = inputs['temp_time_index']
        if self.hyperparams['filter_index_one'] is not None:
            unique_index = unique_index.astype(str).str.cat(inputs.d3mIndex.astype(str))

        # drop original categorical variables, index key, and times
        inputs.set_index('temp_time_index', inplace=True)
        drop_idx = categories + times + key
        inputs.drop(columns = [list(inputs)[idx] for idx in drop_idx], inplace=True)
        self._cat_indices = [arr - len(drop_idx) - 1 for arr in self._cat_indices]
        
        # group data if datetime is not unique
        if not unique_index.is_unique:
            inputs = inputs.groupby(inputs.index).agg('sum')
            self._unique_index = False

        # for each filter value, reindex and interpolate daily values
        if self.filter_idx_one is not None:
            year_dfs = list(inputs.groupby([self.filter_idx_one]))
            year_dfs = [year[1].drop(columns = self.filter_idx_one) for year in year_dfs]
        else:
            year_dfs = [inputs]
        if self.filter_idx is not None:
            company_dfs = [list(year.groupby([self.filter_idx])) for year in year_dfs]
            company_dfs = [[company[1].drop(columns=self.filter_idx) for company in year] for year in company_dfs]
        else:
            company_dfs = [year_dfs]
        reind = [[company.reindex(pandas.date_range(min(year[0].index), max(year[0].index))) if min(year[0].index.year) == max(year[0].index.year)
                      else company for company in year] for year in company_dfs]
        interpolated = [[company.astype(float).interpolate(method='time', limit_direction = 'both') for company in year] for year in reind]
        self._target_lengths = [frame[0].shape[1] for frame in interpolated]
        vals = [pandas.concat(company, axis=1) for company in interpolated]
        self._X_train = vals

        # update hyperparams
        colnames = list(inputs)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce primitive's prediction for future time series data

        Parameters
        ----------
        None

        Returns
        ----------
        Outputs
            The output is a data frame containing the d3m index and a forecast for each of the 'n_periods' future time periods
        """

        # sort test dataset by filter_index_one and filter_index if they exist to get correct ordering of d3mIndex
        if self.filter_idx_one is not None and self.filter_idx is not None:
            inputs = inputs.sort_values(by = [self.filter_idx_one, self.filter_idx])
        elif self.hyperparams['filter_index_one']:
            inputs = inputs.sort_values(by = self.filter_idx_one)
        elif self.hyperparams['filter_index_two']:
            inputs = inputs.sort_values(by = self.filter_idx)

        # take d3m index from input test set
        index = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        output_df = pandas.DataFrame(inputs.iloc[:, index[0]].values)
        output_df.columns = [inputs.metadata.query_column(index[0])['name']]
        
        # produce future foecast using ARIMA models
        future_forecasts = [np.array([f.predict(self.hyperparams['n_periods']) for f in fit]).T for fit in self._fits]

        # undo differencing transformations 
        future_forecasts = [pandas.DataFrame(future_forecast) for future_forecast in future_forecasts]
        if self._lag_order == 1:
            future_forecasts = [future_forecast.apply(lambda x: x + min - 1) for future_forecast, min in zip(future_forecasts, self._mins)]

        # filter forecast according to interval, resahpe according to filter_name
        final_forecasts = []
        idx = None 
        if self.hyperparams['datetime_interval_exception']:
            idx = np.where(np.sort(inputs[self.filter_idx_one].astype(int).unique()) == int(self.hyperparams['datetime_interval_exception']))[0][0]
        if self.hyperparams['specific_intervals'] is None:
            specific_intervals = np.repeat(None, len(future_forecasts))
        else:
            specific_intervals = self.hyperparams['specific_intervals']
        for future_forecast, ind, specific_interval in zip(future_forecasts, range(len(future_forecasts)), specific_intervals):
            if specific_interval is not None:
                final_forecasts.append(future_forecast.iloc[specific_interval,:])
            if ind == idx:
                final_forecasts.append(future_forecast.iloc[0:1,:])
            elif self.hyperparams['interval']:
                final_forecasts.append(future_forecast.iloc[self.hyperparams['interval'] - 1::self.hyperparams['interval'],:])
            else:
                final_forecasts.append(future_forecast)
        
        # convert categorical columns back to categorical labels
        original_cat = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/CategoricalData')
        if self.hyperparams['filter_index_one'] is not None:
            original_cat.remove(self.hyperparams['filter_index_one'])
        if self.hyperparams['filter_index_two'] is not None:
            original_cat.remove(self.hyperparams['filter_index_two'])
        for forecast in final_forecasts:
            for one_hot_cat, original_cat, enc in zip(self._cat_indices, original_cat, self._encoders):
                if self._unique_index:
                    # round categoricals
                    forecast[one_hot_cat] = forecast[one_hot_cat].apply(lambda x: x >= 1.5).astype(int)
                    # convert to categorical labels
                    forecast[list(inputs)[original_cat]] = enc.inverse_transform(forecast[one_hot_cat].values)
                    # remove one-hot encoded columns
                    forecast.drop(columns = one_hot_cat, inplace = True)
                else:
                    # round categoricals to whole numbers
                    forecast[one_hot_cat] = forecast[one_hot_cat].astype(int)

        targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/TrueTarget')  
        if not len(targets):
            targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        
        # select desired columns to return
        if not self._unique_index:
            colnames = list(self._X_train[0])
            times = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Time') + \
                inputs.metadata.get_columns_with_semantic_type('http://schema.org/DateTime')
            times = list(set(times))

            # broadcast predictions
            pred_times = np.flip(inputs.iloc[:,times[0]].unique())
            unique_counts = [inputs.iloc[:,times[0]].value_counts()[p] for p in pred_times]
            final_forecasts = [f.loc[f.index.repeat(unique_counts)].reset_index(drop=True) for f in final_forecasts]
            target_names = [c for c in colnames for t in targets if inputs.metadata.query_column(t)['name'] in c]
        else:
            target_names = [inputs.metadata.query_column(target)['name'] for target in targets]
            if self.hyperparams['filter_index_one'] is not None or self.hyperparams['filter_index_two'] is not None:
                final_forecasts = [future_forecast.values.reshape((-1,len(targets)), order='F') for future_forecast in final_forecasts]
                colnames = list(set(self._X_train[0]))
            else:
                colnames = list(self._X_train[0])
        future_forecast = pandas.DataFrame(np.concatenate(final_forecasts))
        future_forecast.columns = colnames
        future_forecast = future_forecast[target_names]
        # combine d3mIndex and predictions
        output_df = pandas.concat([output_df, future_forecast], axis=1, join='inner')
        var_df = d3m_DataFrame(output_df)

        # first column ('d3mIndex')
        col_dict = dict(var_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        var_df.metadata = var_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        #('predictions')
        for index, name in zip(range(1, len(future_forecast.columns)), future_forecast.columns):
            col_dict = dict(var_df.metadata.query((metadata_base.ALL_ELEMENTS, index)))
            col_dict['structural_type'] = type("1")
            col_dict['name'] = name
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', \
                'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
            var_df.metadata = var_df.metadata.update((metadata_base.ALL_ELEMENTS, index), col_dict)
        
        return CallResult(var_df)



