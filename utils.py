from datetime import datetime
import os


def get_tournament_name(board_size, game_name="hex", num=0):
    date = datetime.now().strftime("%Y-%m-%d")
    location_from_root = (
        f"../models/{game_name}/{board_size}x{board_size}/{date}/tournament{num}"
    )
    location = os.path.join(os.path.dirname(__file__), location_from_root)

    while os.path.exists(location):
        num += 1
        location = location.replace(f"tournament{num-1}", f"tournament{num}")
    return f"tournament{num}"


def get_model_location(board_size, tournament_name, game_name="hex"):
    num = 0
    date = datetime.now().strftime("%Y-%m-%d")
    location_from_root = f"../models/{game_name}/{board_size}x{board_size}/{date}/{tournament_name}/model_{num}.h5"
    location = os.path.join(os.path.dirname(__file__), location_from_root)

    while os.path.exists(location):
        num += 1
        location = location.replace(f"model_{num-1}", f"model_{num}")

    # find index of last '/' and replace everything after it with 'params.py'
    param_location = location[: location.rfind("/")] + "/params.py"
    return location, param_location

    # params_file_location = os.path.join(
    #     os.path.dirname(__file__), "../config/params.py"
    # )
    # params_copy_location = os.path.join(
    #     os.path.dirname(__file__),
    #     f"../models/{game_name}/{board_size}x{board_size}/{date}/{tournament}/params.py",
    # )
