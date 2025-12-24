import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def show_orbit_plot(data: pd.DataFrame) -> None:
    t = np.array(data["time"])

    fig, axs = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

    # 1. График высоты
    axs[0].plot(t, np.array(data["altitude"]) / 1000, color="purple", linestyle="-", label="Высота (км)")
    axs[0].plot(t, np.array(data["apoapsis"]) / 1000, "r--", label="Апоцентр (км)")
    axs[0].plot(t, np.array(data["periapsis"]) / 1000, "b--", label="Перицентр (км)")
    axs[0].axhline(y=70, color="gray", linestyle="-", alpha=0.5, label="Граница атмосферы")
    axs[0].set_ylabel("Высота (км)")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, linestyle="--")
    axs[0].set_title("Орбитальные параметры полета")
    axs[0].set_ylim(0, 100)

    # 2. График скорости
    axs[1].plot(t, data["speed"], label="Орбитальная скорость (м/с)", color="green")
    axs[1].set_ylabel("Скорость (м/с)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--")

    # 3. График массы
    axs[2].plot(t, data["mass"], label="Масса аппарата (кг)", color="purple")
    axs[2].set_ylabel("Масса (кг)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle="--")

    # 4. График углов (Фазовое окно)
    # axs[3].plot(t, data["phase_angle"], label="Фазовый угол (до Муны)", color="orange", linewidth=2)
    #
    # axs[3].set_ylabel("Угол (градусы)")
    # axs[3].set_xlabel("Время от старта (с)")
    # axs[3].set_ylim(0, 360)
    # axs[3].set_yticks([0, 90, 180, 270, 360])
    # axs[3].legend(loc="upper right")
    # axs[3].grid(True, linestyle="--")
    # axs[3].set_title("Навигация: Трансферное окно")

    plt.tight_layout()
    plt.show()


def normalize_data(data: dict) -> None:
    min_length = len(min(data.values(), key=len))

    for key in data.keys():
        data[key] = data[key][:min_length]


def data_to_csv(data: pd.DataFrame, filename: str) -> None:
    try:
        data_frame = pd.DataFrame(data)
    except Exception as e:
        print("Ошибка обработки данных. Попробуйте normalize_data(data) перед использованием")
        print(e)
        return

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    file_path = os.path.join(data_folder, filename)

    try:
        data_frame.to_csv(file_path, index=False)
    except Exception as e:
        print("Ошибка сохранения данных. Попробуйте normalize_data(data) перед использованием")
        print(e)


def csv_to_data(filename: str) -> pd.DataFrame | None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    file_path = os.path.join(data_folder, filename)

    if not os.path.exists(data_folder):
        print(f"Ошибка: папка data/ не найдена.")
        return None
    if not os.path.exists(os.path.join(data_folder, filename)):
        print(f"Ошибка: файл {file_path} не найден.")
        return None

    try:
        data = pd.read_csv(file_path)
        return data
    except Exception as e:
        print(f"Произошла ошибка при парсинге CSV: {e}")
        return None


def get_starting_data(filename: str) -> dict | None:
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    file_path = os.path.join(data_folder, filename)

    if not os.path.exists(data_folder):
        print(f"Ошибка: папка data/ не найдена.")
        return None
    if not os.path.exists(os.path.join(data_folder, filename)):
        print(f"Ошибка: файл {file_path} не найден.")
        return None

    try:
        df = pd.read_csv(file_path, nrows=1)

        print(f"Начальные данные успешно загружены из {file_path}.")
        return df.iloc[0].to_dict()
    except Exception as e:
        print(f"Произошла ошибка при парсинге CSV: {e}")
        return None


if __name__ == "__main__":
    show_orbit_plot(csv_to_data("ksp_data_orbit.csv"))
