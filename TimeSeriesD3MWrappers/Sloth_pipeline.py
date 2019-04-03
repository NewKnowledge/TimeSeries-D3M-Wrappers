from d3m import index
from d3m.metadata.base import ArgumentType, Context
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline(context=Context.TESTING)
pipeline_description.add_input(name='inputs')

# Step 2: DISTIL/NK Storc primitive
step_0 = PrimitiveStep(primitive=index.get_primitive('d3m.primitives.clustering.k_means.Sloth'))
step_0.add_argument(name='inputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_argument(name='outputs', argument_type=ArgumentType.CONTAINER, data_reference='inputs.0')
step_0.add_hyperparameter(name='nclusters', argument_type= ArgumentType.VALUE, data=4)
step_0.add_output('produce')
pipeline_description.add_step(step_0)

# Final Output
pipeline_description.add_output(name='output predictions', data_reference='steps.0.produce')

# Output to JSON
print(pipeline_description.to_json())