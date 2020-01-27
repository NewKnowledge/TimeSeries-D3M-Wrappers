import sys
import os
import collections
import numpy as np
import pandas as pd
import typing
from sklearn.preprocessing import OneHotEncoder
from datetime import timedelta

from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.exceptions import PrimitiveNotFittedError

from d3m import container, utils
from d3m.container import DataFrame as d3m_DataFrame
from d3m.metadata import hyperparams, base as metadata_base, params

from statsmodels.tsa.api import VAR as vector_ar
import statsmodels.api as sm
import scipy.stats as stats

from TimeSeriesD3MWrappers.models.var_model_utils import Arima, calculate_time_frequency, discretize_time_difference
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__author__ = "Distil"
__version__ = "1.2.0"
__contact__ = "mailto:jeffrey.gleason@yonder.co"

Inputs = container.pandas.DataFrame
Outputs = container.pandas.DataFrame

MAX_INT = np.finfo('d').max - 1

class Params(params.Params):
    pass


class Hyperparams(hyperparams.Hyperparams):
    max_lag_order = hyperparams.Union[typing.Union[int, None]](
        configuration=collections.OrderedDict(
            user_selected=hyperparams.UniformInt(lower=0, upper=100, default=1),
            auto_selected=hyperparams.Hyperparameter[None](
                default=None,
                description="Lag order of regressions automatically selected",
            ),
        ),
        default="user_selected",
        description="The lag order to apply to regressions. If user-selected, the same lag will be " +
            "applied to all regressions. If auto-selected, different lags can be selected for different " +
            "regressions.",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
    )
    default_lag_order = hyperparams.UniformInt(
        lower=0,
        upper=100,
        default=1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="default lag order to use if matrix decomposition errors",
    )
    seasonal = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to perform ARIMA prediction with seasonal component",
    )
    seasonal_differencing = hyperparams.UniformInt(
        lower=1,
        upper=365,
        default=1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="period of seasonal differencing to use in ARIMA prediction",
    )
    dynamic = hyperparams.UniformBool(
        default=True,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="whether to perform dynamic in-sample prediction with ARIMA model",
    )
    interpret_value = hyperparams.Enumeration(
        default="lag_order",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["series", "lag_order"],
        description="whether to return weight coefficients for each series or each lag order " +
            "separately in the regression",
    )
    interpret_pooling = hyperparams.Enumeration(
        default="avg",
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        values=["avg", "max"],
        description="whether to pool weight coefficients via average or max",
    )
    confidence_interval_horizon = hyperparams.UniformInt(
        lower=1,
        upper=100,
        default=2,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="horizon for confidence interval forecasts. Exposed through auxiliary " +
            "'produce_confidence_intervals' method",
    )
    confidence_interval_alpha = hyperparams.Uniform(
        lower=0.01,
        upper=1,
        default=0.1,
        semantic_types=[
            "https://metadata.datadrivendiscovery.org/types/ControlParameter"
        ],
        description="significance level for confidence interval, i.e. alpha = 0.05 " +
            "returns a 95%% confdience interval from alpha / 2 to 1 - (alpha / 2) . " +
            "Exposed through auxiliary 'produce_confidence_intervals' method",
    )


class VAR(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
    """ Primitive that applies a VAR multivariate forecasting model to time series data. The VAR 
        implementation comes from the statsmodels library. It will default to an ARIMA model if
        timeseries is univariate. The lag order and AR, MA, and differencing terms for the VAR 
        and ARIMA models respectively are selected automatically and independently for each regression. 
        User can override automatic selection with 'max_lag_order' HP.
    
        Arguments:
            hyperparams {Hyperparams} -- D3M Hyperparameter object
        
        Keyword Arguments:
            random_seed {int} -- random seed (default: {0})
    """

    metadata = metadata_base.PrimitiveMetadata(
        {
            # Simply an UUID generated once and fixed forever. Generated using "uuid.uuid4()".
            "id": "76b5a479-c209-4d94-92b5-7eba7a4d4499",
            "version": __version__,
            "name": "VAR",
            # Keywords do not have a controlled vocabulary. Authors can put here whatever they find suitable.
            "keywords": ["Time Series"],
            "source": {
                "name": __author__,
                "contact": __contact__,
                "uris": [
                    # Unstructured URIs.
                    "https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers",
                ],
            },
            # A list of dependencies in order. These can be Python packages, system packages, or Docker images.
            # Of course Python packages can also have their own dependencies, but sometimes it is necessary to
            # install a Python package first to be even able to run setup.py of another package. Or you have
            # a dependency which is not on PyPi.
            "installation": [
                {"type": "PIP", "package": "cython", "version": "0.29.14"},
                {
                    "type": metadata_base.PrimitiveInstallationType.PIP,
                    "package_uri": "git+https://github.com/NewKnowledge/TimeSeries-D3M-Wrappers.git@{git_commit}#egg=TimeSeriesD3MWrappers".format(
                        git_commit=utils.current_git_commit(os.path.dirname(__file__)),
                    ),
                },
            ],
            # The same path the primitive is registered with entry points in setup.py.
            "python_path": "d3m.primitives.time_series_forecasting.vector_autoregression.VAR",
            # Choose these from a controlled vocabulary in the schema. If anything is missing which would
            # best describe the primitive, make a merge request.
            "algorithm_types": [
                metadata_base.PrimitiveAlgorithmType.VECTOR_AUTOREGRESSION
            ],
            "primitive_family": metadata_base.PrimitiveFamily.TIME_SERIES_FORECASTING,
        }
    )

    def __init__(self, *, hyperparams: Hyperparams, random_seed: int = 0) -> None:
        super().__init__(hyperparams=hyperparams, random_seed=random_seed)

        # track metadata about times, targets, indices, grouping keys
        self.filter_idxs = None
        self._target_types = None
        self._targets = None
        self.times = None
        self.key = None
        self.integer_time = False
        self.target_indices = None

        # encodings of categorical variables
        self._cat_indices = []
        self._encoders = []
        self.categories = None

        # information about interpolation
        self.freq = None
        self.interpolation_ranges = None

        # data needed to fit model and reconstruct predictions
        self._X_train_names = None
        self._X_train = None
        self._mins = None
        self._lag_order = []
        self._values = None
        self._fits = []
        self._is_fit = False

    def get_params(self) -> Params:
        return self._params

    def set_params(self, *, params: Params) -> None:
        self.params = params

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        """ Sets primitive's training data
        
            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
                outputs {Outputs} -- full D3M dataframe, containing attributes, key, and target
            
            Raises:
                ValueError: If multiple columns are annotated with 'Time' or 'DateTime' metadata
        """
        # make copy of input data!
        inputs_copy = inputs.copy()

        # mark datetime column
        times = inputs_copy.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/Time",
                "http://schema.org/DateTime",
            )
        )
        if len(times) != 1:
            raise ValueError(
                f"There are {len(times)} indices marked as datetime values. Please only specify one"
            )
        self.time_column = list(inputs_copy)[times[0]]

        # if datetime columns are integers, parse as # of days
        if (
            "http://schema.org/Integer"
            in inputs.metadata.query_column(times[0])["semantic_types"]
        ):
            self.integer_time = True
            inputs_copy[self.time_column] = pd.to_datetime(
                inputs_copy[self.time_column] - 1, unit="D"
            )
        else:
            inputs_copy[self.time_column] = pd.to_datetime(
                inputs_copy[self.time_column], unit="s"
            )

        # sort by time column
        inputs_copy = inputs_copy.sort_values(by = [self.time_column])

        # mark key and grp variables
        self.key = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey"
        )

        # mark target variables
        self._targets = inputs_copy.metadata.list_columns_with_semantic_types(
            (
                "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
                "https://metadata.datadrivendiscovery.org/types/TrueTarget",
                "https://metadata.datadrivendiscovery.org/types/Target",
            )
        )
        self._target_types = [
            "i"
            if "http://schema.org/Integer"
            in inputs_copy.metadata.query_column(t)["semantic_types"]
            else "c"
            if "https://metadata.datadrivendiscovery.org/types/CategoricalData"
            in inputs_copy.metadata.query_column(t)["semantic_types"]
            else "f"
            for t in self._targets
        ]
        self._targets = [list(inputs_copy)[t] for t in self._targets]

        # see if 'GroupingKey' has been marked
        # otherwise fall through to use 'SuggestedGroupingKey' to intelligently calculate grouping key order
        # we sort keys so that VAR can operate on as many series as possible simultaneously (reverse order)
        grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
            drop_list = []
        else:
            drop_list = suggested_grouping_keys

        grouping_keys_counts = [
            inputs_copy.iloc[:, key_idx].nunique() for key_idx in grouping_keys
        ]
        grouping_keys = [
            group_key
            for count, group_key in sorted(zip(grouping_keys_counts, grouping_keys))
        ]
        self.filter_idxs = [list(inputs_copy)[key] for key in grouping_keys]

        # drop index
        inputs_copy.drop(
            columns=[list(inputs_copy)[i] for i in drop_list + self.key], inplace=True
        )

        # check whether no grouping keys are labeled
        if len(grouping_keys) == 0:

            # avg across duplicated time indices if necessary and re-index
            if sum(inputs_copy[self.time_column].duplicated()) > 0:
                inputs_copy = inputs_copy.groupby(self.time_column).mean()
            else:
                inputs_copy = inputs_copy.set_index(self.time_column)

            # interpolate
            self.freq = calculate_time_frequency(inputs_copy.index[1] - inputs_copy.index[0])
            inputs_copy = inputs_copy.interpolate(method="time", limit_direction="both")

            # set X train and target idxs
            self.target_indices = [
                i
                for i, col_name in enumerate(list(inputs_copy))
                if col_name in self._targets
            ]
            self._X_train = [inputs_copy]
            self._X_train_names = [inputs_copy.columns]

        else:
            # find interpolation range from outermost grouping key
            if len(grouping_keys) == 1:
                date_ranges = inputs_copy.agg({self.time_column: ["min", "max"]})
                indices = inputs[self.filter_idxs[0]].unique()
                self.interpolation_ranges = pd.Series(
                    [date_ranges] * len(indices), index=indices
                )
                self._X_train = [None]
                self._X_train_names = [[]]
            else:
                self.interpolation_ranges = inputs_copy.groupby(
                    self.filter_idxs[:-1]
                ).agg({self.time_column: ["min", "max"]})
                self._X_train = [None for i in range(self.interpolation_ranges.shape[0])]
                self._X_train_names = [[] for i in range(self.interpolation_ranges.shape[0])]
            
            for name, group in inputs_copy.groupby(self.filter_idxs):
                if len(grouping_keys) > 2:
                    group_value = tuple([group[self.filter_idxs[i]].values[0] for i in range(len(self.filter_idxs) - 1)])
                else:
                    group_value = group[self.filter_idxs[0]].values[0]
                if len(grouping_keys) > 1:
                    training_idx = np.where(
                            self.interpolation_ranges.index.to_flat_index() == group_value
                        )[0][0]
                else:
                    training_idx = 0
                group = group.drop(columns=self.filter_idxs)

                # avg across duplicated time indices if necessary and re-index
                if sum(group[self.time_column].duplicated()) > 0:
                    group = group.groupby(self.time_column).mean()
                else:
                    group = group.set_index(self.time_column)

                # interpolate
                min_date = self.interpolation_ranges.loc[group_value][self.time_column][
                    "min"
                ]
                max_date = self.interpolation_ranges.loc[group_value][self.time_column][
                    "max"
                ]
                # assume frequency is the same across all time series
                if self.freq is None:
                    self.freq = calculate_time_frequency(group.index[1] - group.index[0])
                group = group.reindex(
                    pd.date_range(min_date, max_date, freq = self.freq), 
                    tolerance = '1' + self.freq, 
                    method = 'nearest')
                group = group.interpolate(method="time", limit_direction="both")

                # add to training data under appropriate top-level grouping key
                self.target_indices = [
                    i
                    for i, col_name in enumerate(list(group))
                    if col_name in self._targets
                ]
                if self._X_train[training_idx] is None:
                    self._X_train[training_idx] = group
                else:
                    self._X_train[training_idx] = pd.concat(
                        [self._X_train[training_idx], group], axis=1
                    )
                self._X_train_names[training_idx].append(name)

    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        """ If there are multiple endogenous series, primitive will fit VAR model. Otherwise it will fit an ARIMA 
            model. In the VAR case, the lag order will be automatically choosen based on BIC (unless user overrides). 
            In the ARIMA case, the lag order will be automatically chosen by differencing tests (again, unless user 
            overrides). 
        
            Keyword Arguments:
                timeout {float} -- timeout, not considered (default: {None})
                iterations {int} -- iterations, not considered (default: {None})
            
            Returns:
                CallResult[None]
        """

        # mark if data is exclusively positive
        self._values = [sequence.values for sequence in self._X_train]
        self._positive = [True if np.min(vals) < 0 else False for vals in self._values]

        # difference data - VAR assumes data is stationary
        self._values_diff = [np.diff(sequence,axis=0) for sequence in self._X_train]

        # TODO always handle potential redundant columns

        # define models
        if self.hyperparams["max_lag_order"] is None:
            arima_max_order = 5
        else:
            arima_max_order = self.hyperparams["max_lag_order"]
        self.models = [
            vector_ar(vals, dates=original.index)
            if vals.shape[1] > 1
            else Arima(
                seasonal=self.hyperparams["seasonal"],
                seasonal_differencing=self.hyperparams["seasonal_differencing"],
                max_order=arima_max_order,
                dynamic = self.hyperparams['dynamic']
            )
            for vals, original in zip(self._values_diff, self._X_train)
        ]

        # fit models
        for vals, model, original in zip(self._values_diff, self.models, self._X_train):

            # VAR
            if vals.shape[1] > 1:
                try:
                    lags = model.select_order(
                        maxlags=self.hyperparams["max_lag_order"]
                    ).bic
                    logger.info(
                        "Successfully performed model order selection. Optimal order = {} lags".format(
                            lags
                        )
                    )
                except np.linalg.LinAlgError:
                    lags = self.hyperparams["default_lag_order"]
                    logger.debug(
                        f"Matrix decomposition error. Using default lag order of {lags}"
                    )
                except ValueError as e:
                    lags = 0
                    logger.debug('ValueError: ' + str(e) + '. Using lag order of 0')
                self._lag_order.append(lags)
                self._fits.append(model.fit(maxlags=lags))

            # ARIMA
            else:
                X_train = pd.Series(
                    data=vals.reshape((-1,)), index=original.index[: vals.shape[0]]
                )
                model.fit(X_train)
                self._lag_order.append(None)
                self._fits.append(model)

        self._is_fit = True
        return CallResult(None, has_finished=self._is_fit)

    def _calculate_prediction_intervals(
        self, inputs: Inputs, grouping_key_ct: int
    ) -> typing.Tuple[
        typing.Sequence[int],
        typing.Sequence[typing.Sequence[int]],
        typing.Sequence[typing.Sequence[typing.Any]],
    ]:
        """ private util function that uses learned grouping keys to extract horizon, 
            horizon intervals and d3mIndex information

            Arguments:
                inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
                grouping_key_ct {int} -- number of grouping keys

            Returns:
                tuple(Sequence[int]) -- number of periods to predict (per forecast)
                Sequence[Sequence[int]] -- prediction slices (per forecast)
                Sequence(Sequence[Any]] -- indices of predictions (per forecast)
        """

        # check whether no grouping keys are labeled
        if grouping_key_ct == 0:
            group_tuple = ((None, inputs),)
        else:
            group_tuple = inputs.groupby(self.filter_idxs)

        # groupby learned filter_idxs and extract n_periods, interval and d3mIndex information
        n_periods = [1 for i in range(len(self._X_train))]
        intervals = [None for i in range(len(self._X_train))]
        d3m_indices = [None for i in range(len(self._X_train))]
        for _, group in group_tuple:
            if grouping_key_ct > 2:
                group_value = tuple([group[self.filter_idxs[i]].values[0] for i in range(len(self.filter_idxs) - 1)])
            elif grouping_key_ct > 1:
                group_value = group[self.filter_idxs[0]].values[0]
            if grouping_key_ct > 1:
                testing_idx = np.where(
                    self.interpolation_ranges.index.to_flat_index() == group_value
                )[0][0]
            else:
                testing_idx = 0
            min_train_idx = self._X_train[testing_idx].index[0]
            time_diff = (
                self._X_train[testing_idx].index[1] - min_train_idx
            ).total_seconds()
            local_intervals = discretize_time_difference(
                group[self.time_column], min_train_idx, self.freq
            )

            # save n_periods prediction information
            num_p = int(max(local_intervals) - self._X_train[testing_idx].shape[0] + 1)
            if n_periods[testing_idx] < num_p:
                n_periods[testing_idx] = num_p

            # save interval prediction information
            if intervals[testing_idx] is None:
                intervals[testing_idx] = [local_intervals]
            else:
                intervals[testing_idx].append(local_intervals)

            # save d3m indices prediction information
            idxs = group.iloc[:, self.key[0]].values
            if d3m_indices[testing_idx] is None:
                d3m_indices[testing_idx] = [idxs]
            else:
                d3m_indices[testing_idx].append(idxs)

        return n_periods, intervals, d3m_indices

    def produce(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ prediction for future time series data

        Arguments:
            inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
        
        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, not considered (default: {None})

        Raises:
            PrimitiveNotFittedError: if primitive not fit
        
        Returns:
            CallResult[Outputs] -- (N, 2) dataframe with d3m_index and value for each prediction slice requested.
                prediction slice = specific horizon idx for specific series in specific regression 
        """
        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        # make copy of input data!
        inputs_copy = inputs.copy()

        # if datetime columns are integers, parse as # of days
        if self.integer_time:
            inputs_copy[self.time_column] = pd.to_datetime(
                inputs_copy[self.time_column] - 1, unit="D"
            )
        else:
            inputs_copy[self.time_column] = pd.to_datetime(
                inputs_copy[self.time_column], unit="s"
            )

        # sort by time column
        inputs_copy = inputs_copy.sort_values(by = [self.time_column])

        # find marked 'GroupingKey' or 'SuggestedGroupingKey'
        grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/GroupingKey"
        )
        suggested_grouping_keys = inputs_copy.metadata.get_columns_with_semantic_type(
            "https://metadata.datadrivendiscovery.org/types/SuggestedGroupingKey"
        )
        if len(grouping_keys) == 0:
            grouping_keys = suggested_grouping_keys
        else:
            inputs_copy = inputs_copy.drop(columns=[list(inputs_copy)[i] for i in suggested_grouping_keys])

        # groupby learned filter_idxs and extract n_periods, interval and d3mIndex information
        n_periods, intervals, d3m_indices = self._calculate_prediction_intervals(
            inputs_copy, len(grouping_keys)
        )

        # produce future forecast using VAR / ARMA
        future_forecasts = [
            fit.forecast(y = vals[vals.shape[0]-fit.k_ar:], 
                steps = n)
            if lags is not None and lags > 0
            else np.repeat(fit.params, n, axis = 0)
            if lags == 0 
            else fit.predict(n_periods = n).reshape(-1,1)
            for fit, vals, lags, n in zip(
                self._fits, self._values_diff, self._lag_order, n_periods
            )
        ]

        # prepend in-sample predictions
        future_forecasts = [
            np.concatenate((vals[:1], vals_diff[:1], fit.predict_in_sample().reshape(-1,1), pred), axis = 0)
            if lags is None
            else np.concatenate((vals[:1], vals_diff[:lags], fit.fittedvalues, pred), axis = 0)
            for fit, pred, lags, vals, vals_diff
            in zip(self._fits, future_forecasts, self._lag_order, self._values, self._values_diff)
        ]

        # undo differencing transformation, convert to df
        future_forecasts = [pd.DataFrame(future_forecast.cumsum(axis=0)) for future_forecast in future_forecasts]

        # apply invariances (real, positive data AND rounding NA / INF values)
        future_forecasts = [
            f.clip(lower = 0).replace(np.inf, np.nan)
            if not positive 
            else f.replace(np.inf, np.nan)
            for f, positive in zip(future_forecasts, self._positive)
        ]
        future_forecasts = [f.fillna(f.mean()) for f in future_forecasts]

        # select predictions to return based on intervals
        key_names = [list(inputs)[k] for k in self.key]
        var_df = pd.DataFrame([], columns=key_names + self._targets)

        # TODO speed up nested for loops
        for forecast, interval, idxs in zip(future_forecasts, intervals, d3m_indices):
            if interval is not None:
                for row, col, d3m_idx in zip(interval, range(len(interval)), idxs):
                    # if new col in test (endogenous variable), average over all other cols
                    if col >= forecast.shape[1]:
                        logger.debug('Needed new forecast column!')
                        preds = forecast.mean(axis = 1).replace(np.inf, MAX_INT)
                        preds = pd.concat([preds] * len(self.target_indices), axis=1)
                    else:
                        cols = [col + t for t in self.target_indices]
                        preds = forecast[cols]
                    for r, i in zip(row, d3m_idx):
                        var_df.loc[var_df.shape[0]] = [i, *preds.iloc[r].values]
        var_df = d3m_DataFrame(var_df)
        var_df.iloc[:, 0] = var_df.iloc[:, 0].astype(int)
        logger.debug(var_df.head())

        # first column ('d3mIndex')
        col_dict = dict(var_df.metadata.query((metadata_base.ALL_ELEMENTS, 0)))
        col_dict["structural_type"] = type("1")
        col_dict["name"] = key_names[0]
        col_dict["semantic_types"] = (
            "http://schema.org/Integer",
            "https://metadata.datadrivendiscovery.org/types/PrimaryKey",
        )
        var_df.metadata = var_df.metadata.update(
            (metadata_base.ALL_ELEMENTS, 0), col_dict
        )

        # assign target metadata and round appropriately
        for (index, name), target_type in zip(
            enumerate(self._targets), self._target_types
        ):
            col_dict = dict(
                var_df.metadata.query((metadata_base.ALL_ELEMENTS, index + 1))
            )
            col_dict["structural_type"] = type("1")
            col_dict["name"] = name
            col_dict["semantic_types"] = (
                "https://metadata.datadrivendiscovery.org/types/PredictedTarget",
            )
            if target_type == "i":
                var_df[name] = var_df[name].astype(int)
                col_dict["semantic_types"] += ("http://schema.org/Integer",)
            elif target_type == "c":
                col_dict["semantic_types"] += (
                    "https://metadata.datadrivendiscovery.org/types/CategoricalData",
                )
            else:
                col_dict["semantic_types"] += ("http://schema.org/Float",)
            var_df.metadata = var_df.metadata.update(
                (metadata_base.ALL_ELEMENTS, index + 1), col_dict
            )
        return CallResult(var_df, has_finished=self._is_fit)

    def produce_weights(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ Produce absolute values of correlation coefficients (weights) for each of the terms used in each regression model. 
            Terms must be aggregated by series or by lag order (thus the need for absolute value). Pooling operation can be maximum 
            or average (controlled by 'interpret_pooling' HP).
        
        Arguments:
            inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
        
        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})

        Raises:
            PrimitiveNotFittedError: if primitive not fit
        
        Returns:
            CallResult[Outputs] -- pandas df where each row represents a unique series from one of the regressions that was fit. 
            The columns contain the coefficients for each term in the regression, potentially aggregated by series or lag order. 
            Column names will represent the lag order or series to which that column refers. 
            If the regression is an ARIMA model, the set of column names will also contain AR_i (autoregressive terms) and 
                MA_i (moving average terms)
            Columns that are not included in the regression for a specific series will have NaN values in those
                respective columns. 
        """

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        if self.hyperparams['interpret_value'] == 'series':
            logger.info("You should interpret a row of the returned matrix like this: " + 
                "Each row represents an endogeneous variable for which the VAR process learned an equation. " +
                "Each column represents all of the endogenous variables used in the regression equation. " + 
                "Each matrix entry represents the weight of the column endogeneous variable in the equation for the " + 
                "row endogenous variable.")

        # get correlation coefficients
        coefficients = [
            np.absolute(fit.coefs)
            if lags is not None
            else fit.get_absolute_value_params()
            for fit, lags in zip(self._fits, self._lag_order)
        ]
        trends = [
            np.absolute(fit.params[0, :].reshape(-1, 1)) if lags is not None else None
            for fit, lags in zip(self._fits, self._lag_order)
        ]

        # combine coeffcient vectors into single df
        coef_df = None
        for coef, trend, names in zip(coefficients, trends, self._X_train_names):
            # aggregate VAR coefficients based on HPs
            if trend is not None:
                if self.hyperparams["interpret_value"] == "series":
                    if self.hyperparams["interpret_pooling"] == "avg":
                        coef = np.mean(coef, axis=0)  # K x K
                    else:
                        coef = np.max(coef, axis=0)  # K x K
                    colnames = names
                else:
                    # or axis = 2, I believe symmetrical
                    if self.hyperparams["interpret_pooling"] == "avg":
                        coef = np.mean(coef, axis=1).T  # K x p + 1
                    else:
                        coef = np.max(coef, axis=1).T  # K x p + 1
                    coef = np.concatenate((trend, coef), axis=1)
                    colnames = ["trend_0"] + [
                        "ar_" + str(i + 1) for i in range(coef.shape[1] - 1)
                    ]
                new_df = pd.DataFrame(coef, columns=colnames, index=names)
                coef_df = pd.concat([coef_df, new_df], sort = True)

            # add index to ARIMA params
            else:
                coef.index = names
                if self.hyperparams["interpret_value"] == "lag_order":
                    coef_df = pd.concat([coef_df, coef], sort = True)

        if coef_df is not None:
            logger.info(f"There was no more than one variable in each grouping of time series, " +
                "therefore only ARIMA models were fit. Additionally, becasue the 'interpret_value' " +
                "hyperparameter is set to series, this will return an empty dataframe.")

        return CallResult(
            container.DataFrame(coef_df, generate_metadata=True),
            has_finished=self._is_fit,
        )

    def produce_confidence_intervals(
        self, *, inputs: Inputs, timeout: float = None, iterations: int = None
    ) -> CallResult[Outputs]:
        """ produce confidence intervals for each series 'confidence_interval_horizon' periods into 
                the future
        
        Arguments:
            inputs {Inputs} -- full D3M dataframe, containing attributes, key, and target
        
        Keyword Arguments:
            timeout {float} -- timeout, not considered (default: {None})
            iterations {int} -- iterations, considered (default: {None})
        
        Raises:
            PrimitiveNotFittedError: 
        
        Returns:
            CallResult[Outputs] -- 

            Ex. 
                series | timestep | mean | 0.05 | 0.95
                --------------------------------------
                a      |    0     |  5   |   3  |   7
                a      |    1     |  6   |   4  |   8
                b      |    0     |  5   |   3  |   7
                b      |    1     |  6   |   4  |   8
        """

        logger.info("This method exclusively produces confidence interval forecasts for future timesteps. It does " +
            "not produce confidence intervals for in-sample predictions. The length of the horizon " + 
            "can be controlled with the hyperparameter 'confidence_interval_horizon.'")

        if not self._is_fit:
            raise PrimitiveNotFittedError("Primitive not fitted.")

        horizon = self.hyperparams['confidence_interval_horizon']
        alpha = self.hyperparams['confidence_interval_alpha']

        # produce confidence interval forecasts using VAR / ARIMA
        confidence_intervals = []
        for fit, vals, vals_diff, lags in zip(self._fits, self._values, self._values_diff, self._lag_order):
            if lags is not None and lags > 0:
                forecast = fit.forecast_interval(y = vals_diff[-fit.k_ar:], 
                    steps = horizon, 
                    alpha = alpha)
                
                # undo differencing transforms
                forecast = [f.cumsum(axis = 0) for f in forecast]
                forecast += np.sum(np.concatenate((vals[:1], vals_diff[:lags], fit.fittedvalues), axis = 0), axis = 0)
                confidence_intervals.append(forecast)
            elif lags == 0:
                q = stats.norm.ppf(1 - alpha / 2)
                mean = np.repeat(fit.params, horizon, axis = 0)
                lower = np.repeat(fit.params - q * fit.stderr, horizon, axis = 0)
                upper = np.repeat(fit.params + q * fit.stderr, horizon, axis = 0)

                # undo differencing transforms
                offset = np.sum(np.concatenate((vals[:1], vals_diff[:lags], fit.fittedvalues), axis = 0), axis = 0)
                mean = mean.cumsum(axis = 0) + offset
                lower = lower.cumsum(axis = 0) + offset
                upper = upper.cumsum(axis = 0) + offset
                confidence_intervals.append(
                    (mean, lower, upper)
                )
            else:
                forecast = fit.predict(n_periods = horizon, 
                    return_conf_int = True, 
                    alpha = alpha)

                # undo differencing transforms
                forecast = [f.cumsum(axis = 0) for f in forecast]
                forecast += np.sum(np.concatenate((vals[:1], vals_diff[:1], fit.predict_in_sample().reshape(-1,1)), axis = 0), axis = 0)
                confidence_intervals.append(forecast)

        # combine into long form df
        series_names = [
            [[name] * horizon for name in name_list] 
            for name_list in self._X_train_names
        ]
        series_names = [
            [name for sub_name_list in name_list for name in sub_name_list] 
            for name_list in series_names
        ]        
        confidence_intervals = [
            pd.DataFrame(
                    np.concatenate(
                    [
                        point_estimate.flatten(order = 'F').reshape(-1, 1)    
                        for point_estimate in interval
                    ], 
                    axis = 1
                ), 
                index = names, 
                columns = ['mean', str(alpha / 2), str(1 - alpha / 2)]
            )
            for interval, names in zip(confidence_intervals, series_names)
        ]

        # apply invariances (real, positive data AND rounding NA / INF values)
        confidence_intervals = [
            ci.clip(lower = 0)#.replace(np.inf, np.nan)
            #if not positive 
            #else ci.replace(np.inf, np.nan)
            for ci, positive in zip(confidence_intervals, self._positive) if not positive 
        ]
        #confidence_intervals = [ci.fillna(ci.mean()) for ci in confidence_intervals]

        interval_df = pd.concat(confidence_intervals)

        # add index column
        interval_df['horizon_index'] = np.tile(np.arange(horizon), len(interval_df.index.unique()))
        return CallResult(
            container.DataFrame(interval_df, generate_metadata=True),
            has_finished=self._is_fit,
        )


