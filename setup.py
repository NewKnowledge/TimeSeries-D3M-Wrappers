from setuptools import setup

setup(
    name="TimeSeriesD3MWrappers",
    version="1.2.0",
    description="Two forecasting primitives: Vector Autoregression and Deep Autoregressive forecasting. Two classification primiives: KNN and LSTM FCN with attention",
    packages=["TimeSeriesD3MWrappers"],
    install_requires=[
        "numpy>=1.15.4,<=1.17.3",
        "scipy>=1.2.1,<=1.3.1",
        "scikit-learn[alldeps]>=0.20.3,<=0.21.3",
        "pandas>=0.23.4,<=0.25.2",
        "tensorflow-gpu == 2.0.0",
        "tslearn == 0.2.5",
        "statsmodels==0.10.2",
        "pmdarima==1.0.0",
        "deepar @ git+https://github.com/NewKnowledge/deepar@c801332d26742c17c4265d2155372ce7f1192bc4#egg=deepar-0.0.2",
    ],
    entry_points={
        "d3m.primitives": [
            "time_series_classification.k_neighbors.Kanine = TimeSeriesD3MWrappers.primitives.classification_knn:Kanine",
            "time_series_forecasting.vector_autoregression.VAR = TimeSeriesD3MWrappers.primitives.forecasting_var:VAR",
            "time_series_classification.convolutional_neural_net.LSTM_FCN = TimeSeriesD3MWrappers.primitives.classification_lstm:LSTM_FCN",
            "time_series_forecasting.lstm.DeepAR = TimeSeriesD3MWrappers.primitives.forecasting_deepar:DeepAR",
        ],
    },
)
