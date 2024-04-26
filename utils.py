from datetime import datetime
import os


def get_train_session_name(board_size, game_name="hex", num=0):
    location_from_root = (
        f"./models/{game_name}/{board_size}x{board_size}/train_session{num}"
    )
    location = os.path.join(os.path.dirname(__file__), location_from_root)
    print(location)
    while os.path.exists(location):
        num += 1
        location = location.replace(f"train_session{num-1}", f"train_session{num}")
    return f"train_session{num}"


def get_model_location(board_size, train_session, game_name="hex"):
    num = 0
    location_from_root = (
        f"./models/{game_name}/{board_size}x{board_size}/{train_session}/model_{num}.h5"
    )
    location = os.path.join(os.path.dirname(__file__), location_from_root)

    while os.path.exists(location):
        num += 1
        location = location.replace(f"model_{num-1}", f"model_{num}")

    param_location = location[: location.rfind("/")] + "/params.py"
    return location, param_location


if __name__ == "__main__":
    print(get_train_session_name(4))
    # print(get_model_location(4, "tournament0"))
    import tensorflow as tf

    # Configuring TensorFlow to use all logical cores available
    num_threads = tf.config.threading.get_inter_op_parallelism_threads()
    print(f"Number of threads: {num_threads}")

    # tf.config.threading.set_intra_op_parallelism_threads(num_threads)
    # tf.config.threading.set_inter_op_parallelism_threads(num_threads)
    from tensorflow.python.client import device_lib

    print(device_lib.list_local_devices())
