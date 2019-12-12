from pmdarima.arima import auto_arima
import pandas as pd
import numpy as np
import logging
import typing

logger = logging.getLogger(__name__)
# logger.setLevel(logging.DEBUG)


class Arima:
    def __init__(
        self, seasonal=True, seasonal_differencing=1, max_order=5, dynamic=True
    ):
        """initialize ARIMA class
        
        Keyword Arguments:
            seasonal {bool} -- whether time series has seasonal component (default: {True})
            seasonal_differencing {int} -- period for seasonal differencing (default: {1})
            max_order {int} -- maximum order of p and q terms on which to fit model (default: {5})
            dynamic {bool} -- whether in-sample lagged values should be used for in-sample prediction
        """

        self.seasonal = seasonal
        self.seasonal_differencing = seasonal_differencing
        self.max_order = max_order
        self.dynamic = dynamic

    def _transform(self, input):
        """ transforms data according to defined transformation 
        
        Arguments:
            input {np array} -- data pre-transform
        
        Returns:
            np array -- data post-transform
        """
        return np.log(input - self.min_train + 1)

    def _inverse_transform(self, input):
        """ inverse transform of data according to defined transformation 
        
        Arguments:
            input {np array} -- data pre-inverse-transform
        
        Returns:
            np array -- data post-inverse-transform
        """
        return np.exp(input) + self.min_train - 1

    def fit(self, train):
        """fit ARIMA model on training data, automatically selecting p (AR), q (MA), 
            P (AR - seasonal), Q (MA - seasonal), d, and D (differencing) amongst other parameters
            based on AIC
        
        Arguments:
            np array -- endogenous time series on which model should select parameters and fit
        """

        self.min_train = min(train)
        self.arima_model = auto_arima(
            train,
            # self._transform(train),
            start_p=1,
            start_q=1,
            max_p=self.max_order,
            max_q=self.max_order,
            m=self.seasonal_differencing,
            seasonal=self.seasonal,
            stepwise=True,
            suppress_warnings=True,
        )
        logger.info(f'Fit ARIMA model with {self.arima_model.df_model()} degrees of freedom')
        # self.arima_model.fit(self._transform(train))
        self.arima_model.fit(train)

    def predict(self, n_periods=1, return_conf_int=False, alpha=0.05):
        """forecasts the time series n_periods into the future
        
        Keyword Arguments:
            n_periods {int} -- number of periods to forecast into the future (default: {1})
            return_conf_int {bool} -- whether to return confidence intervals instead of 
                forecasts
            alpha {float} -- significance level for confidence interval, i.e. alpha = 0.05 
                returns a 95% confdience interval from alpha / 2 to 1 - (alpha / 2) 
                (default: {0.05})
        
        Returns:
            np array -- (n, 1) time series forecast n_periods into the future
                OR (n, 2) if returning confidence interval forecast
        """
        logger.info(f'Making predictions for ARIMA model with {self.arima_model.df_model()} degrees of freedom')
        if return_conf_int:
            forecast, interval = self.arima_model.predict(
                n_periods=n_periods, return_conf_int=True, alpha=alpha
            )
            return (
                forecast,
                interval[:, 0],
                interval[:, 1],
                # self._inverse_transform(forecast),
                # self._inverse_transform(interval[:, 0]),
                # self._inverse_transform(interval[:, 1]),
            )
        else:
            # return self._inverse_transform(
            #     self.arima_model.predict(n_periods=n_periods)
            # )
            return self.arima_model.predict(n_periods=n_periods)

    def predict_in_sample(self):
        """ thin wrapper for ARIMA predict_in_sample f(). always predicts all in-sample 
            points (except for first point). dynamic parameter controlled by instance variable
        """
        # return self._inverse_transform(
        #     self.arima_model.predict_in_sample(0, 1, dynamic=self.dynamic)
        # )
        return self.arima_model.predict_in_sample(0, 1, dynamic=self.dynamic)

    def get_absolute_value_params(self):
        """get absolute value of trend, AR, and MA parameters of 
            fit ARIMA model (no exogenous variables)
        
        Returns:
            pandas df -- df with column for each parameter
        """

        try:
            ar_count = self.arima_model.arparams().shape[0]
        except AttributeError:
            logger.debug("There are no ar parameters in this model")
            ar_count = 0
        try:
            ma_count = self.arima_model.maparams().shape[0]
        except AttributeError:
            logger.debug("There are no ma parameters in this model")
            ma_count = 0
        trend_count = self.arima_model.df_model() - ar_count - ma_count

        ar_cols = ["ar_" + str(i + 1) for i in range(ar_count)]
        ma_cols = ["ma_" + str(i + 1) for i in range(ma_count)]
        trend_cols = ["trend_" + str(i) for i in range(trend_count)]

        return pd.DataFrame(
            np.absolute(self.arima_model.params().reshape(1, -1)),
            columns=trend_cols + ar_cols + ma_cols,
        )


# define time constants
SECONDS_PER_MINUTE = 60
MINUTES_PER_HOUR = 60
HOURS_PER_DAY = 24
DAYS_PER_WEEK = 7
DAYS_PER_MONTH = [28, 30, 31]
DAYS_PER_YEAR = [365, 366]

S_PER_YEAR_0 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_YEAR[0]
S_PER_YEAR_1 = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_YEAR[1]
S_PER_MONTH_28 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH[0]
)
S_PER_MONTH_30 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH[1]
)
S_PER_MONTH_31 = (
    SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_MONTH[2]
)
S_PER_WEEK = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY * DAYS_PER_WEEK
S_PER_DAY = SECONDS_PER_MINUTE * MINUTES_PER_HOUR * HOURS_PER_DAY
S_PER_HR = SECONDS_PER_MINUTE * MINUTES_PER_HOUR


def calculate_time_frequency(time_diff):
    """method that calculates the frequency of a datetime difference (for prediction slices) 
    
        Arguments:
            time_diff {timedelta or float} -- difference between two instances
        
        Returns:
            str -- string alias representing granularity of pd.datetime object
    """

    # convert to seconds representation
    if type(time_diff) is pd._libs.tslibs.timedeltas.Timedelta:
        time_diff = time_diff.total_seconds()

    if time_diff % S_PER_YEAR_0 == 0:
        logger.debug("granularity is years")
        return "YS"
    elif time_diff % S_PER_YEAR_1 == 0:
        logger.debug("granularity is years")
        return "YS"
    ## TODO - currently hacky solution because PHEM monthly is end of month (want return 'M')
    #         sunspots monthly is beginning of month (want return 'MS')
    #         should have robust way of differentiating
    elif time_diff % S_PER_MONTH_31 == 0:
        # Sunspots monthly
        logger.debug("granularity is months 31")
        return "MS"
    elif time_diff % S_PER_MONTH_30 == 0:
        logger.debug("granularity is months 30")
        return "MS"
    elif time_diff % S_PER_MONTH_28 == 0:
        # PHEM monthly
        logger.debug("granularity is months 28")
        return "M"
    elif time_diff % S_PER_WEEK == 0:
        logger.debug("granularity is weeks")
        return "W"
    elif time_diff % S_PER_DAY == 0:
        logger.debug("granularity is days")
        return "D"
    elif time_diff % S_PER_HR == 0:
        logger.debug("granularity is hours")
        return "H"
    else:
        logger.debug("granularity is seconds")
        return "S"


def discretize_time_difference(
    times, initial_time, frequency, integer_timestamps=False
) -> typing.Sequence[int]:
    """method that discretizes sequence of datetimes (for prediction slices) 
    
        Arguments:
            times {Sequence[datetime] or Sequence[float]} -- sequence of datetime objects
            initial_time {datetime or float} -- last datetime instance from training set 
                (to offset test datetimes)
            frequency {str} -- string alias representing granularity of pd.datetime object
        
        Keyword Arguments:
            integer_timestamps {bool} -- whether timestamps are integers or datetime values

        Returns:
            typing.Sequence[int] -- prediction intervals expressed at specific time granularity

    """

    # take differences to convert to deltas
    time_differences = times - initial_time

    # edge case for integer timestamps
    if integer_timestamps:
        return time_differences.values.astype(int)

    # convert to seconds representation
    if type(time_differences.iloc[0]) is pd._libs.tslibs.timedeltas.Timedelta:
        time_differences = time_differences.apply(lambda t: t.total_seconds())

    if frequency == "YS":
        return [round(x / S_PER_YEAR_0) for x in time_differences]
    elif frequency == "MS" or frequency == 'M':
        return [round(x / S_PER_MONTH_31) for x in time_differences]
    elif frequency == "W":
        return [round(x / S_PER_WEEK) for x in time_differences]
    elif frequency == "D":
        return [round(x / S_PER_DAY) for x in time_differences]
    elif frequency == "H":
        return [round(x / S_PER_HR) for x in time_differences]
    else:
        return [round(x / SECONDS_PER_MINUTE) for x in time_differences]

