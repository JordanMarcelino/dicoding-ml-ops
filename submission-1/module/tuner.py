from typing import Any, Dict, NamedTuple, Text

import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from keras_tuner.engine import base_tuner
from tensorflow.keras import layers
from tfx.components.trainer.fn_args_utils import FnArgs

LABEL_KEY = "label"
FEATURE_KEY = "text"
NUM_EPOCHS = 5

# Define parameter tuning with early stopping callbacks
TunerResult = NamedTuple(
    "TunerFnResult", [("tuner", base_tuner.BaseTuner), ("fit_kwargs", Dict[Text, Any])]
)
es = tf.keras.callbacks.EarlyStopping(
    monitor="val_binary_accuracy",
    mode="max",
    verbose=1,
    patience=10,
    restore_best_weights=True,
)


# Create utilities function
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(
    file_pattern, tf_transform_output, num_epochs, batch_size=64
) -> tf.data.Dataset:
    """Get post_tranform feature & create batches of data"""

    # Get post_transform feature spec
    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    # create batches of data
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def model_builder(hp, vectorizer):
    """
    Builds the model and sets up the hyperparameters to tune.
    """

    ## Define parameter used for tuning model
    n_layers = hp.Int("n_layers", min_value=1, max_value=5, step=1)
    embedding_dim = hp.Int("embedding_dim", min_value=16, max_value=128, step=32)
    lstm_units = hp.Int("lstm_units", min_value=32, max_value=128, step=32)
    fc_units = hp.Int("fc_units", min_value=16, max_value=64, step=16)
    lr = hp.Choice("lr", values=[1e-2, 1e-3, 1e-4])

    inputs = tf.keras.Input(
        shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string
    )

    x = vectorizer(inputs)
    x = layers.Embedding(5_000, embedding_dim, name="embedding")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units))(x)

    for _ in range(n_layers):
        x = layers.Dense(fc_units, activation="relu")(x)

    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        loss=tf.keras.losses.BinaryCrossentropy(),
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=[tf.keras.metrics.BinaryAccuracy()],
    )

    # print(model)
    model.summary()
    return model


def tuner_fn(fn_args: FnArgs):
    tf_output = tft.TFTransformOutput(fn_args.transform_graph_path)
    train_set = input_fn(fn_args.train_files[0], tf_output, NUM_EPOCHS)
    eval_set = input_fn(fn_args.eval_files[0], tf_output, NUM_EPOCHS)

    vectorize_layer = layers.TextVectorization(
        max_tokens=5_000,
        output_mode="int",
        output_sequence_length=500,
    )
    vectorize_layer.adapt(train_set.map(lambda x, _: x[transformed_name(FEATURE_KEY)]))

    # Define tuner search strategy
    tuner = kt.Hyperband(
        hypermodel=lambda hp: model_builder(hp, vectorize_layer),
        objective=kt.Objective("binary_accuracy", direction="max"),
        max_epochs=NUM_EPOCHS,
        factor=3,
        directory=fn_args.working_dir,
        project_name="kt_hyperband",
    )

    return TunerResult(
        tuner=tuner,
        fit_kwargs={
            "callbacks": [es],
            "x": train_set,
            "validation_data": eval_set,
            "steps_per_epoch": fn_args.train_steps,
            "validation_steps": fn_args.eval_steps,
        },
    )
