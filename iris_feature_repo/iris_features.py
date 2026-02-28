from datetime import timedelta
from feast import Entity, FeatureView, Field
from feast.infra.offline_stores.bigquery_source import BigQuerySource
from feast.types import Float32, Int64

iris_entity = Entity(
    name="iris_id",
    description="Unique ID for each iris sample",
)

iris_source = BigQuerySource(
    table="meta-territory-488805-q1.feast_iris.iris_data",
    timestamp_field="event_timestamp",
)

iris_feature_view = FeatureView(
    name="iris_features",
    entities=[iris_entity],
    ttl=timedelta(days=365),
    schema=[
        Field(name="sepal_length", dtype=Float32),
        Field(name="sepal_width",  dtype=Float32),
        Field(name="petal_length", dtype=Float32),
        Field(name="petal_width",  dtype=Float32),
        Field(name="species",      dtype=Int64),
    ],
    source=iris_source,
)
