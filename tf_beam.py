
import apache_beam as beam
import tensorflow as tf
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam
import tensorflow.keras.layers as layers
import os
import tempfile
import time


FEATURE_KEYS = ['C'+str(i) for i in range(10)]
print(f'Feature keys: {FEATURE_KEYS}')

RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.float32))
    for name in FEATURE_KEYS]
)
print(f'Raw Feature Spec: {RAW_DATA_FEATURE_SPEC}')

RAW_DATA_METADATA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec(
        RAW_DATA_FEATURE_SPEC)
)
#print(f'Raw Data Metadata: {RAW_DATA_METADATA}')

TRAIN_NUM_EPOCHS = 1
NUM_TRAIN_INSTANCES = 1
TRAIN_BATCH_SIZE = 1
NUM_TEST_INSTANCES = 1

TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'


class AnomalyDetector(tf.keras.Model):
  def __init__(self, features):
    super(AnomalyDetector, self).__init__()
    h_outer = features // 2
    h_inner = features // 3
    latent = features // 5
    self.encoder = tf.keras.Sequential([
      layers.Dense(h_outer, activation="relu"),
      layers.Dense(h_inner, activation="relu"),
      layers.Dense(latent, activation="relu")])
    
    self.decoder = tf.keras.Sequential([
      layers.Dense(h_inner, activation="relu"),
      layers.Dense(h_outer, activation="relu"),
      layers.Dense(features, activation="sigmoid")])
    
  def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded

class MapAndFilterErrors(beam.PTransform):
    """Like beam.Map but filters out errors in the map_fn."""

    class _MapAndFilterErrorsDoFn(beam.DoFn):
        """Count the bad examples using a beam metric."""

        def __init__(self, fn):
            self._fn = fn
            # Create a counter to measure number of bad elements.
            self._bad_elements_counter = beam.metrics.Metrics.counter(
                'census_example', 'bad_elements')

        def process(self, element):
            try:
                yield self._fn(element)
            except Exception as e:  # pylint: disable=broad-except
                # Catch any exception the above call.
                print(e)
                self._bad_elements_counter.inc(1)

    def __init__(self, fn):
        self._fn = fn

    def expand(self, pcoll):
        return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))



def preprocessing_fn(inputs):
    outputs = dict().fromkeys(FEATURE_KEYS)
    for key in FEATURE_KEYS:
        outputs[key] = tft.scale_to_z_score(inputs[key])
    return outputs


def transform_data(train_data_file, working_dir):
    with beam.Pipeline() as pipeline:
        with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
            ordered_columns = ['C'+str(i) for i in range(10)]
            print(ordered_columns)
            converter = tft.coders.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

            raw_data = (
                pipeline
                | 'Read Train Data' >> beam.io.ReadFromText(train_data_file, skip_header_lines=1)
                | 'Fix Commas in Train Data' >> beam.Map(lambda line: line.replace(', ', ','))
                | 'Decode Train Data' >> MapAndFilterErrors(converter.decode))
            print("\n\n\n", raw_data.__dict__)
            print("\n\n\n", raw_data.producer)
            print("\n\n\n", raw_data.producer.__dict__)
            raw_dataset = (raw_data, RAW_DATA_METADATA)
            transformed_dataset, transform_fn = (
                raw_dataset | tft.beam.AnalyzeAndTransformDataset(preprocessing_fn))

            transformed_data, transformed_metadata = transformed_dataset

            transformed_data_coder = tft.coders.ExampleProtoCoder(
                transformed_metadata.schema)
            
            _ = (
                transformed_data
                | 'Encode Train Data' >> beam.Map(transformed_data_coder.encode)
                | 'Write Train Data' >> beam.io.WriteToTFRecord(
                    os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))
           
            _ = (
                transform_fn
                | 'Write TransformFn' >> tft.beam.WriteTransformFn(working_dir))
            print("YOOHOO\n")


def _make_training_input_fn(tf_transform_output, transformed_examples, batch_size):
    def input_fn():
        dataset = tf.data.experimental.make_batched_features_dataset(
            file_pattern=transformed_examples,
            batch_size=batch_size,
            features=tf_transform_output.transformed_feature_spec(),
            reader=tf.data.TFRecordDataset,
            shuffle=True)
        dataset = dataset.map(lambda *x: (tf.stack(x), tf.stack(x)))

        transformed_features = tf.compat.v1.data.make_one_shot_iterator(
            dataset).get_next()
        print("\ndada", type(transformed_features))
        #### !!!!!!!!
        #return (transformed_features, transformed_features)
        return transform_features
    return input_fn()


def _make_serving_input_fn(tf_transform_output):
    def serving_input_fn():
        raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
            RAW_DATA_FEATURE_SPEC, default_batch_size=None)
        seving_input_receiver = raw_input_fn()

        raw_features = seving_input_receiver.features
        transformed_features = tf_transform_output.transform_raw_features(
            raw_features)
        
        return tf.estimator.export.ServingInputReceiver(
            transformed_features, seving_input_receiver.receiver_tensors)
    return serving_input_fn

def get_feature_columns(tf_transform_output):
    real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                            for key in FEATURE_KEYS]
    return real_valued_columns

#def _build_keras_model(hidden_units: List[int] = None) -> tf.keras.Model:


def train_and_evaluate(working_dir, num_train_instances=NUM_TRAIN_INSTANCES,
                        num_test_instances=NUM_TEST_INSTANCES):
    tf_transform_output = tft.TFTransformOutput(working_dir)
    
    n_feat = len(FEATURE_KEYS)
    print(type(n_feat))
    autoencoder = AnomalyDetector(
        features=n_feat
    )
    autoencoder.compile(optimizer='adam', loss='mae')

    #estimator = tf.estimator.LinearClassifier(
    #    feature_columns=get_feature_columns(tf_transform_output),
    #    config=run_config,
    #    loss_reduction=tf.losses.Reduction.SUM)

    print("asdasdsad")
    train_input_fn = _make_training_input_fn( 
        tf_transform_output,
        os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
        batch_size=1)
    print("Training±\n")

    #estimator.train(
    #    input_fn=train_input_fn,
    #    max_steps=TRAIN_NUM_EPOCHS * num_train_instances / TRAIN_BATCH_SIZE)


    autoencoder.fit(
        train_input_fn,
        epochs=TRAIN_NUM_EPOCHS)
    
    serving_input_fn = _make_serving_input_fn(tf_transform_output)
    exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
    estimator.export_saved_model(exported_model_dir, serving_input_fn)


temp = tempfile.gettempdir()
temp = 'yolodir/{}'.format(time.asctime().replace(' ', '_').replace(':', '-'))
transform_data('data/toy_data.csv', temp)
train_and_evaluate(temp)
