import copy
import time

import tensorflow as tf

from .window_generator import WindowGenerator

MOTOR_TEMPERATURE = '_MOT_TEMP_'
CONV_WIDTH = 3


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class TimeHistory(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.times = []

    def on_epoch_begin(self, batch, logs={}):
        self.epoch_time_start = time.time()

    def on_epoch_end(self, batch, logs={}):
        self.times.append(time.time() - self.epoch_time_start)


def compile_and_fit(model, window, patience=2, max_epochs=20):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')

    time_callback = TimeHistory()
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])

    history = model.fit(window.train, epochs=max_epochs,
                        validation_data=window.val,
                        callbacks=[early_stopping, time_callback])
    # Время исполнения!
    times = time_callback.times
    itog = 0
    for time in times:
        itog += time
    print('Время --- ', itog)
    return history


def build_linear(single_step_window, val_performance, performance):
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(linear, single_step_window)

    val_performance['Linear'] = linear.evaluate(single_step_window.val)
    performance['Linear'] = linear.evaluate(single_step_window.test, verbose=0)

    return linear


def build_dense(single_step_window, val_performance, performance):
    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(dense, single_step_window)

    val_performance['Dense'] = dense.evaluate(single_step_window.val)
    performance['Dense'] = dense.evaluate(single_step_window.test, verbose=0)

    return dense


def build_convolution_neural_network(conv_window, val_performance, performance):
    conv_model = tf.keras.Sequential([
        tf.keras.layers.Conv1D(filters=32,
                               kernel_size=(CONV_WIDTH,),
                               activation='relu'),
        tf.keras.layers.Dense(units=32, activation='relu'),
        tf.keras.layers.Dense(units=1),
    ])

    history = compile_and_fit(conv_model, conv_window)

    val_performance['Conv'] = conv_model.evaluate(conv_window.val)
    performance['Conv'] = conv_model.evaluate(conv_window.test, verbose=0)
    return conv_model


def build_recurrent_neural_network(wide_window, val_performance, performance):
    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=1)
    ])

    history = compile_and_fit(lstm_model, wide_window)

    val_performance['LSTM'] = lstm_model.evaluate(wide_window.val)
    performance['LSTM'] = lstm_model.evaluate(wide_window.test, verbose=0)
    return lstm_model


def get_accuracy(model, inputs, train_std, train_mean):
    print(train_std[1])

    test_list_in = copy.deepcopy(inputs.numpy())

    for index, i in enumerate(test_list_in):
        for index2, j in enumerate(i):
            for index3, h in enumerate(j):
                test_list_in[index][index2][index3] = (h * train_std[2]) + train_mean[2]

    predictions = model(inputs)
    test_list = copy.deepcopy(predictions.numpy())
    for index, i in enumerate(test_list):
        for index2, j in enumerate(i):
            for index3, h in enumerate(j):
                test_list[index][index2][index3] = (h * train_std[2]) + train_mean[2]
    data_len = 0
    perc = 0
    for input, pred in zip(test_list_in, test_list):
        for input_1, pred_1 in zip(input, pred):
            for input_2, pred_2 in zip(input_1, pred_1):
                perc += (abs(input_2 - pred_2) / input_2) * 100
                data_len += 1

    print('Точность --- ', perc / data_len)


def build_single_step(n_train_df, n_val_df, n_test_df, column_indices, train_std, train_mean):
    single_step_window = WindowGenerator(
        train_df=n_train_df, val_df=n_val_df, test_df=n_test_df,
        input_width=1, label_width=1, shift=0,
        label_columns=[MOTOR_TEMPERATURE]
    )

    baseline = Baseline(label_index=column_indices[MOTOR_TEMPERATURE])

    baseline.compile(loss=tf.losses.MeanSquaredError(),
                     metrics=[tf.metrics.MeanAbsoluteError()])

    val_performance = {}
    performance = {}
    val_performance['Baseline'] = baseline.evaluate(single_step_window.val)
    performance['Baseline'] = baseline.evaluate(single_step_window.test, verbose=0)

    wide_window = WindowGenerator(
        train_df=n_train_df, val_df=n_val_df, test_df=n_test_df,
        input_width=500, label_width=500, shift=0,
        label_columns=[MOTOR_TEMPERATURE]
    )

    # wide_window.plot(baseline, plot_col=MOTOR_TEMPERATURE, train_std=train_std, train_mean=train_mean)

    # Линейная модель
    linear = build_linear(single_step_window, val_performance, performance)
    wide_window.plot(linear, plot_col=MOTOR_TEMPERATURE, name="Линейная модель", train_std=train_std,
                     train_mean=train_mean)
    get_accuracy(linear, wide_window.example[0], train_std=train_std,
                 train_mean=train_mean)

    # Полтная модель
    dense = build_dense(single_step_window, val_performance, performance)
    wide_window.plot(dense, plot_col=MOTOR_TEMPERATURE, name="Плотная модель", train_std=train_std,
                     train_mean=train_mean)
    get_accuracy(dense, wide_window.example[0], train_std=train_std,
                 train_mean=train_mean)

    conv_window = WindowGenerator(
        input_width=CONV_WIDTH,
        label_width=1,
        shift=0,
        label_columns=[MOTOR_TEMPERATURE],
        train_df=n_train_df, val_df=n_val_df, test_df=n_test_df
    )

    # Сверточная нейронная сеть
    conv_model = build_convolution_neural_network(conv_window, val_performance, performance)
    # print(conv_model.get_weights())
    LABEL_WIDTH = 1000
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH,
        label_width=LABEL_WIDTH,
        shift=1,
        label_columns=[MOTOR_TEMPERATURE],
        train_df=n_train_df, val_df=n_val_df, test_df=n_test_df
    )

    wide_conv_window.plot(conv_model, plot_col=MOTOR_TEMPERATURE, name="Сверточная нейронная сеть", train_std=train_std,
                          train_mean=train_mean)
    inputs, _ = wide_conv_window.example
    get_accuracy(conv_model, inputs, train_std=train_std,
                 train_mean=train_mean)

    reccur_model = build_recurrent_neural_network(wide_window, val_performance, performance)
    wide_window.plot(reccur_model, plot_col=MOTOR_TEMPERATURE, name="Рекурентная нейронная сеть", train_std=train_std,
                     train_mean=train_mean)
    get_accuracy(reccur_model, wide_window.example[0], train_std=train_std,
                 train_mean=train_mean)
