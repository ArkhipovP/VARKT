import math
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def get_angle(reference, target):
    t_pos = target.position(reference)

    # Углы объекта в плоскости экватора
    t_angle = math.atan2(t_pos[2], t_pos[0]) * 180 / math.pi

    return t_angle


def get_phase_angle(reference, vessel, target):
    return (get_angle(reference, target) - get_angle(reference, vessel)) % 360


def show_orbit_plot(data):
    t = np.array(data["time"])

    # Увеличиваем количество графиков до 4
    fig, axs = plt.subplots(4, 1, figsize=(10, 15), sharex=True)

    # 1. График высоты
    axs[0].plot(t, np.array(data["altitude"]) / 1000, label="Высота (км)", color="blue", linewidth=2)
    axs[0].plot(t, np.array(data["apoapsis"]) / 1000, "--", label="Апоцентр (км)", color="cyan")
    axs[0].plot(t, np.array(data["periapsis"]) / 1000, ":", label="Перицентр (км)", color="navy")
    axs[0].axhline(y=70, color="red", linestyle="-", alpha=0.3, label="Граница атмосферы")
    axs[0].set_ylabel("Высота (км)")
    axs[0].legend(loc="upper right")
    axs[0].grid(True, linestyle="--")
    axs[0].set_title("Орбитальные параметры полета")

    # 2. График скорости
    axs[1].plot(t, data["velocity"], label="Орбитальная скорость (м/с)", color="green")
    axs[1].set_ylabel("Скорость (м/с)")
    axs[1].legend(loc="upper right")
    axs[1].grid(True, linestyle="--")

    # 3. График массы
    axs[2].plot(t, data["mass"], label="Масса аппарата (кг)", color="purple")
    axs[2].set_ylabel("Масса (кг)")
    axs[2].legend(loc="upper right")
    axs[2].grid(True, linestyle="--")

    # 4. График углов (Фазовое окно)
    axs[3].plot(t, data["phase_angle"], label="Фазовый угол (до Муны)", color="orange", linewidth=2)

    axs[3].set_ylabel("Угол (градусы)")
    axs[3].set_xlabel("Время от старта (с)")
    axs[3].set_ylim(0, 360)
    axs[3].set_yticks([0, 90, 180, 270, 360])
    axs[3].legend(loc="upper right")
    axs[3].grid(True, linestyle="--")
    axs[3].set_title("Навигация: Трансферное окно")

    plt.tight_layout()
    plt.show()


def normalize_data(data):
    min_length = len(min(data.values(), key=len))

    for key in data.keys():
        data[key] = data[key][:min_length]


def data_to_csv(data, filename):
    data_frame = pd.DataFrame(data)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    file_name = os.path.join(data_folder, filename)

    data_frame.to_csv(file_name, index=False)
