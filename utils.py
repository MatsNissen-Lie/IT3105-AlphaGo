from datetime import datetime
import os


def get_tournament_name(board_size, game_name="hex", num=0):
    date = datetime.now().strftime("%Y-%m-%d")
    location_from_root = (
        f"./models/{game_name}/{board_size}x{board_size}/{date}/tournament{num}"
    )
    location = os.path.join(os.path.dirname(__file__), location_from_root)
    print(location)
    while os.path.exists(location):
        num += 1
        location = location.replace(f"tournament{num-1}", f"tournament{num}")
    return f"tournament{num}"


def get_model_location(board_size, tournament_name, game_name="hex"):
    num = 0
    date = datetime.now().strftime("%Y-%m-%d")
    location_from_root = f"./models/{game_name}/{board_size}x{board_size}/{date}/{tournament_name}/model_{num}.h5"
    location = os.path.join(os.path.dirname(__file__), location_from_root)

    while os.path.exists(location):
        num += 1
        location = location.replace(f"model_{num-1}", f"model_{num}")

    param_location = location[: location.rfind("/")] + "/params.py"
    return location, param_location


if __name__ == "__main__":
    print(get_tournament_name(4))
    # print(get_model_location(4, "tournament0"))