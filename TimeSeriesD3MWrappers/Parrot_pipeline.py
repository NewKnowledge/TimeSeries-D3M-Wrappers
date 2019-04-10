from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline(context=Context.TESTING)
pipeline_description.add_input(name='inputs')

# Step 1: dataset_to_dataframe
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.data_transformation.dataset_to_dataframe.Common'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_hyperparameter(name='dataframe_resource', argument_type= ArgumentType.VALUE, data='learningData')
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Step 2: DISTIL/NK Parrot primitive
step_1 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.time_series_forecasting.arima.Parrot'))
step_1.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='steps.0.produce')
step_1.add_hyperparameter(name='seasonal_differencing', argument_type= ArgumentType.VALUE, data=11)
step_1.add_hyperparameter(name='n_periods', argument_type= ArgumentType.VALUE, data=29)
step_1.add_output('produce')
pipeline_description.add_step(step_1)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.1.produce')

# Output to JSON
with open('pipeline.json', 'w') as outfile:
    outfile.write(pipeline_description.to_json())