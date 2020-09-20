from .single_step_models import build_single_step

DATE = '_DATE_'
NUM_MOTOR = '_NumMotor_'
CURRENT_ACTION = '_CURR_ACT_'
VELOCITY_AXIS_ACTION = '_VEL_AXIS_ACT_'
MOTOR_TEMPERATURE = '_MOT_TEMP_'
TORQUE_AXIS_ACTION = '_TORQUE_AXIS_ACT_'
VELOCITY_AXIS = '_VEL_AXIS_'
TIMESTAMP = 'Timestamp'
TIME = 'Time'

FIRST = '_1_'
SECOND = '_2_'
THIRD = '_3_'
FOURTH = '_4_'
FIFTH = '_5_'
SIXTH = '_6_'


def normalization(train_df, val_df, test_df):
    train_mean = train_df.mean()
    train_std = train_df.std()

    n_train_df = (train_df - train_mean) / train_std
    n_val_df = (val_df - train_mean) / train_std
    n_test_df = (test_df - train_mean) / train_std

    return n_train_df, n_val_df, n_test_df, train_std, train_mean


def data_splitting(df):
    column_indices = {name: i for i, name in enumerate(df.columns)}

    n = len(df)
    train_df = df[0:int(n * 0.7)]
    val_df = df[int(n * 0.7):int(n * 0.9)]
    test_df = df[int(n * 0.9):]

    # ЗАДУМАЙСЯ
    num_features = df.shape[1]

    return train_df, val_df, test_df, column_indices


def remove_columns(df_data, columns_to_remove: list):
    # Удаление ненужных для обучения колонок
    df = df_data.drop(columns_to_remove, axis=1)

    # Удаление неименовоной колонки (которая может возникнуть случайным образом)
    for col in df.columns:
        if 'Unnamed' in col:
            df = df.drop(col, axis=1)

    return df


def do_learn(robot_data):
    df = robot_data

    # Подготовка данных
    df = remove_columns(df, [NUM_MOTOR, VELOCITY_AXIS, DATE])

    # Разбиение данных на тестовые выборки
    train_df, val_df, test_df, column_indices = data_splitting(df)

    # Нормализация данных
    n_train_df, n_val_df, n_test_df, train_std, train_mean = normalization(train_df, val_df, test_df)
    # Одношаговые модели
    build_single_step(n_train_df, n_val_df, n_test_df, column_indices, train_std, train_mean)
