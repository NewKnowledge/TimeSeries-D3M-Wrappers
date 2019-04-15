import sys
import os.path
import numpy as np
import pandas
import typing

from statsmodels.tsa.api import VAR as vector_ar
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA

from d3m.primitive_interfaces.base import PrimitiveBase, CallResult

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params
from common_primitives import utils as utils_cp, dataset_to_dataframe as DatasetToDataFrame, dataset_regex_filter

__author__ = 'Distil'
__version__ = '1.0.0'
__contact__ = 'mailto:nklabs@newknowledge.com'

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

class Params(params.Params):
    pass

class Hyperparams(hyperparams.Hyperparams):  
    n_periods = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize, 
        default = 30, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
       description='number of periods to predict')
    interval = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='interval with which to sample future predictions')
    datetime_interval_exception = hyperparams.Hyperparameter[typing.Union[str, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='to handle different prediction intervals (stock market dataset). \
            If this HP is set, primitive will just make next forecast for this datetime value \
            (not multiple forecasts at multiple intervals')
    max_lags = hyperparams.UniformInt(
        lower = 1, 
        upper = sys.maxsize, 
        default = 15, 
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='maximum lag order to evluate to find model - eval criterion = AIC')
    datetime_filter = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='index of column in input dataset that contain unique identifiers of \
            time series that have different datetime indices')
    filter_index = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = None,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'], 
        description='index of column in input dataset that contain unique identifiers of different time series')
    datetime_index = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = 0,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],  
        description='if multiple datetime indices exist, this HP specifies which to apply to training data')
    arma_p = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = 0,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],  
        description='The p order of the ARMA model in case some time series are univariate')
    arma_q = hyperparams.Hyperparameter[typing.Union[int, None]](
        default = 0,
        semantic_types = ['https://metadata.datadrivendiscovery.org/types/ControlParameter'],  
        description='The q order of the ARMA model in case some time series are univariate')
    pass

class VAR(PrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """
        Produce primitive's prediction for future time series data. The output is a data frame containing the d3m index and a 
        forecast for each of the 'n_periods' future time periods, modified if desired by the 'interval' HP. The default is a 
        future forecast for each of the selected input variables. This can be modified to just one output variable with 
        the associated HP
    """
    metadata = metadata_base.PrimitiveMetadata({
        # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
        'id': "76b5a479-c209-4d94-92b5-7eba7a4d4499",
        'version': __version__,
        'name': "VAR",
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
         'installation': [
            {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package': 'cython',
            'version': '0.28.5',
            },
            {
            'type': metadata_base.PrimitiveInstallationType.PIP,
            'package_uri': 'git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers'.format(
                git_commit=utils.current_git_commit(os.path.dirname(__file__)),)
            }
        ],
        # The same path the primitive is registered with entry points in setup.py.
        'python_path': 'd3m.primitives.time_series_forecasting.vector_autoregression.VAR',
        # Choose these from a controlled vocabulary in the schema. If anything is missing which would
        # best describe the primitive, make a merge request.
        'algorithm_types': [
            metadata_base.PrimitiveAlgorithmType.VECTOR_AUTOREGRESSION
        ],
        'primitive_family': metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
    })

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0)-> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        self._params = {}
        self._X_train = None
        self._values = None 
        self._fits = None
        self._final_logs = None

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        '''
        fits VAR model. Evaluates different lag orders up to maxlags, eval criterion = AIC
        '''
        
        # log transformation for standardization, difference, drop NAs
        self._values = [np.log(year.values) for year in self._X_train]
        self._final_logs = [year[-1:,] for year in self._values]
        self._values = [np.diff(year,axis=0) for year in self._values]

        # use ARMA model in case some data only has 1 variable
        ## TODO - calculate these params automatically / hyperparams
        arma_p = self.hyperparams['arma_p']
        arma_q = self.hyperparams['arma_q']

        models = [vector_ar(vals, dates = original.index) if vals.shape[1] > 1 \
            else ARMA(vals, order = (arma_p, arma_q), dates = original.index) for vals, original in zip(self._values, self._X_train)]
        self._fits = []
        for vals, model in zip(self._values, models):
            if vals.shape[1] > 1:
                lags = model.select_order(maxlags = self.hyperparams['max_lags']).aic
                lags = lags if lags > 1 else 1
                self._fits.append(model.fit(lags))
            else:
                self._fits.append(model.fit(disp = -1))
        return CallResult(None)

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        '''
        Sets primitive's training data

        Parameters
        ----------
        inputs: input d3m_dataframe containing n columns of features
        
        '''
        # set datetime index
        times = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/Time')
        time_index = times[self.hyperparams['datetime_index']]
        inputs.index = pandas.DatetimeIndex(inputs.iloc[:,time_index])

        # eliminate categorical variables, times, primary key
        cat = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/CategoricalData')
        key = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')

        # check that targets aren't categorical
        targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        if not len(targets):
            raise ValueError("All suggested targets are categorical variables. VAR cannot regress on categorical variables")

        # for each filter value, reindex and interpolate daily values
        if self.hyperparams['datetime_filter']:
            year_dfs = list(inputs.groupby(inputs.columns[self.hyperparams['datetime_filter']]))
        else:
            year_dfs = [inputs]
        if self.hyperparams['filter_index']:
            company_dfs = [list(year[1].groupby(year[1].columns[self.hyperparams['filter_index']])) for year in year_dfs]
        else:
            company_dfs = [year_dfs]
        reind = [[company[1].drop(company[1].columns[cat + key + times], axis = 1).reindex(pandas.date_range(min(year[0][1].iloc[:,time_index]), 
                max(year[0][1].iloc[:,time_index]))) for company in year] for year in company_dfs]
        interpolated = [[company.astype(float).interpolate(method='time', limit_direction = 'both') for company in year] for year in reind]
        self._target_lengths = [frame[0].shape[1] for frame in interpolated]
        vals = [pandas.concat(company, axis=1) for company in interpolated]
        self._X_train = vals

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:

        """
        Produce primitive's prediction for future time series data

        Parameters
        ----------
        None

        Returns
        ----------
        Outputs
            The output is a data frame containing the d3m index and a forecast for each of the 'n_periods' future time periods, 
            modified if desired by the 'interval' HP
            The default is a future forecast for each of the selected input variables. This can be modified to just one output 
                variable with the associated HP
        """

        # sort test dataset by datetime_filter and filter_index if they exist to get correct ordering of d3mIndex
        if self.hyperparams['datetime_filter'] and self.hyperparams['filter_index']:
            inputs = inputs.sort_values(by = [inputs.columns[self.hyperparams['datetime_filter']], inputs.columns[self.hyperparams['filter_index']]])
        elif self.hyperparams['datetime_filter']:
            inputs = inputs.sort_values(by = inputs.columns[self.hyperparams['datetime_filter']])
        elif self.hyperparams['filter_index']:
            inputs = inputs.sort_values(by = inputs.columns[self.hyperparams['filter_index']])

        # take d3m index from input test set
        index = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/PrimaryKey')
        output_df = pandas.DataFrame(inputs.iloc[:, index[0]].values)
        output_df.columns = [inputs.metadata.query_column(index[0])['name']]
        
        # produce future foecast using VAR / ARMA
        future_forecasts = [fit.forecast(vals[-fit.k_ar:], self.hyperparams['n_periods']) if vals.shape[1] > 1 \
            else fit.forecast(self.hyperparams['n_periods'])[0] for fit, vals in zip(self._fits, self._values)]
        
        # undo differencing transformations 
        future_forecasts = [np.exp(future_forecast.cumsum(axis=0) + final_logs).T if len(future_forecast.shape) is 1 \
            else np.exp(future_forecast.cumsum(axis=0) + final_logs) for future_forecast, final_logs in zip(future_forecasts, self._final_logs)]
        future_forecasts = [pandas.DataFrame(future_forecast) for future_forecast in future_forecasts]

        # filter forecast according to interval, resahpe according to filter_name
        final_forecasts = []
        idx = None 
        if self.hyperparams['datetime_interval_exception']:
            idx = np.where(np.sort(inputs[inputs.columns[self.hyperparams['datetime_filter']]].astype(int).unique()) == int(self.hyperparams['datetime_interval_exception']))[0][0]
        for future_forecast, ind in zip(future_forecasts, range(len(future_forecasts))):
            if ind == idx:
                final_forecasts.append(future_forecast.iloc[0:1,:])
            elif self.hyperparams['interval']:
                final_forecasts.append(future_forecast.iloc[self.hyperparams['interval'] - 1::self.hyperparams['interval'],:])
            else:
                final_forecasts.append(future_forecast)
        targets = inputs.metadata.get_columns_with_semantic_type('https://metadata.datadrivendiscovery.org/types/SuggestedTarget')
        final_forecasts = [future_forecast.values.reshape((-1,len(targets)), order='F') for future_forecast in final_forecasts]
        future_forecast = pandas.DataFrame(np.concatenate(final_forecasts))

        # select desired columns to return
        colnames = [inputs.metadata.query_column(target)['name'] for target in targets]
        future_forecast.columns = list(set(self._X_train[0]))
        future_forecast = future_forecast[colnames]
        
        output_df = pandas.concat([output_df, future_forecast], axis=1)
        var_df = d3m_DataFrame(output_df)
        
        # first column ('d3mIndex')
        col_dict = dict(var_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict['structural_type'] = type("1")
        col_dict['name'] = inputs.metadata.query_column(index[0])['name']
        col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/PrimaryKey',)
        var_df.metadata = var_df.metadata.update((metadata_base.ALL_ELEMENTS, 0), col_dict)

        #('predictions')
        for index, name in zip(range(1, len(colnames)), colnames):
            col_dict = dict(var_df.metadata.query((metadata_base.ALL_ELEMENTS, index)))
            col_dict['structural_type'] = type("1")
            col_dict['name'] = name
            col_dict['semantic_types'] = ('http://schema.org/Integer', 'https://metadata.datadrivendiscovery.org/types/SuggestedTarget', \
                'https://metadata.datadrivendiscovery.org/types/TrueTarget', 'https://metadata.datadrivendiscovery.org/types/Target')
            var_df.metadata = var_df.metadata.update((metadata_base.ALL_ELEMENTS, index), col_dict)

        return CallResult(var_df)

    
    def produce_weights(self, *, inputs: Inputs, filter_value: str = None, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        """
        Produce correlation coefficients (weights) for each of the terms used in the regression model

        Parameters
        ----------
        filter_value:   value to select a filter from column filter index for which to return correlation coefficient matrix. If None, 
                        method returns most recent filter

        Returns
        ----------
        Outputs
            The output is a data frame containing columns for each of the terms used in the regression model. Each row contains the 
            correlation coefficients for each term in the regression model. If there are multiple unique timeseries indices in the 
            dataset there can be multiple rows in this output dataset. Terms that aren't included in a specific timeseries index will 
            have a value of NA in the associated matrix entry.
        """
        if filter_value is None:
            filter_value = inputs.iloc[:, self.hyperparams['datetime_filter']].max()

        # get correlation coefficients 
        coef = [fit.coefs if vals.shape[1] > 1 else np.array([1]) for fit, vals in zip(self._fits, self._values)]
        idx = np.where(inputs.iloc[:, self.hyperparams['datetime_filter']].unique() == filter_value)[0][0]
        coef_df = pandas.DataFrame(coef[idx][0], columns= inputs.iloc[:, self.hyperparams['filter_index']].unique())
        return CallResult(coef_df)
    

if __name__ == '__main__':
    
    input_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/LL1_736_stock_market/TRAIN/dataset_TRAIN/datasetDoc.json')
    hyperparams_class = DatasetToDataFrame.DatasetToDataFramePrimitive.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    ds2df_client = DatasetToDataFrame.DatasetToDataFramePrimitive(hyperparams = hyperparams_class.defaults().replace({"dataframe_resource":"learningData"}))
    df = d3m_DataFrame(ds2df_client.produce(inputs = input_dataset).value)
    
    # VAR primitive
    var_hp = VAR.metadata.query()['primitive_code']['class_type_arguments']['Hyperparams']
    var = VAR(hyperparams = var_hp.defaults().replace({'filter_index':1, 'datetime_filter': 2, 'n_periods':52, 'interval':26, 'datetime_interval_exception':'2017'}))
    var.set_training_data(inputs = df, outputs = None)
    var.fit()
    test_dataset = container.Dataset.load('file:///datasets/seed_datasets_current/LL1_736_stock_market/TEST/dataset_TEST/datasetDoc.json')
    #results = var.produce(inputs = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value))
    results = var.produce_weights(inputs = d3m_DataFrame(ds2df_client.produce(inputs = test_dataset).value), filter_value='2017')
    print(results.value)
