from d3m import index
from d3m.metadata.base import ArgumentType
from d3m.metadata.pipeline import Pipeline, PrimitiveStep

# Creating pipeline
pipeline_description = Pipeline()
pipeline_description.add_input(name="inputs")

# Step 1: Ts formatter
step_0 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_preprocessing.data_cleaning.DistilTimeSeriesFormatter"
    )
)
step_0.add_argument(
    name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
)
step_0.add_output("produce")
pipeline_description.add_step(step_0)

# Step 2: DS to DF on formatted ts DS
step_1 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
    )
)
step_1.add_argument(
    name="inputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.0.produce",
)
step_1.add_output("produce")
pipeline_description.add_step(step_1)

# Step 3: DS to DF on input DS
step_2 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_transformation.dataset_to_dataframe.Common"
    )
)
step_2.add_argument(
    name="inputs", argument_type=ArgumentType.CONTAINER, data_reference="inputs.0"
)
step_2.add_output("produce")
pipeline_description.add_step(step_2)

# step 3: column parser on input DF
step_3 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_transformation.column_parser.Common"
    )
)
step_3.add_argument(
    name="inputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.2.produce",
)
step_3.add_output("produce")
step_3.add_hyperparameter(
    name="parse_semantic_types",
    argument_type=ArgumentType.VALUE,
    data=[
        "http://schema.org/Boolean",
        "http://schema.org/Integer",
        "http://schema.org/Float",
        "https://metadata.datadrivendiscovery.org/types/FloatVector",
    ],
)
pipeline_description.add_step(step_3)

# Step 5: parse target semantic types
step_4 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_transformation.extract_columns_by_semantic_types.Common"
    )
)
step_4.add_argument(
    name="inputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.3.produce",
)
step_4.add_hyperparameter(
    name="semantic_types",
    argument_type=ArgumentType.VALUE,
    data=[
        "https://metadata.datadrivendiscovery.org/types/Target",
        "https://metadata.datadrivendiscovery.org/types/TrueTarget",
        "https://metadata.datadrivendiscovery.org/types/SuggestedTarget",
    ],
)
step_4.add_output("produce")
pipeline_description.add_step(step_4)

# Step 6: LSTM_FCN
step_5 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.time_series_classification.convolutional_neural_net.LSTM_FCN"
    )
)
step_5.add_argument(
    name="inputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.1.produce",
)
step_5.add_argument(
    name="outputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.4.produce",
)
step_5.add_output("produce")
pipeline_description.add_step(step_5)

# Step 7: construct predictions
step_6 = PrimitiveStep(
    primitive=index.get_primitive(
        "d3m.primitives.data_transformation.construct_predictions.Common"
    )
)
step_6.add_argument(
    name="inputs",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.5.produce",
)
step_6.add_argument(
    name="reference",
    argument_type=ArgumentType.CONTAINER,
    data_reference="steps.2.produce",
)
step_6.add_output("produce")
pipeline_description.add_step(step_6)

# Final Output
pipeline_description.add_output(
    name="output predictions", data_reference="steps.6.produce"
)

# Output json pipeline
blob = pipeline_description.to_json()
filename = blob[8:44] + ".json"
# filename = 'pipeline.json'
with open(filename, "w") as outfile:
    outfile.write(blob)
