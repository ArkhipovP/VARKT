from enum import Enum, auto

import numpy as np
import matplotlib.pyplot as plt

# Константы Кербина
KERBIN_GRAV = 3.5316e12  # Гравитационный параметр (м^3/с^2)
KERBIN_R = 600000  # Радиус Кербина (м)
KERBIN_ATMOSPHERE_R = 70000  # Радиус атмосферы Кербина относительно уровня земли
KERBIN_DENSITY = 1.225  # Плотность воздуха на уровне моря (кг/м^3)
KERBIN_PRESSURE = 101325  # Давление на уровне моря (Па)
KERBIN_G = 9.80665
H_SCALE = 5600  # Шкала высот (м)

# Константы ракеты
S_REF = 12.56  # Примерная площадь для тяжелой ракеты (радиус ~2м)
C_D = 0.25     # Коэффициент лобового сопротивления

# Данные для гравитационного манёвра
GRAV_TURN_START = 500   # Высота для начала
GRAV_TURN_CEIL = 45000  # "Потолок" поворота гравитационного манёвра (подобран эмпирически на основе графика)

# Данные для трансфера к Муне
MUN_ORBIT_R = 12000000  # Радиус орбиты Муны (м)
KERBIN_ORBIT_R = 190000  # Наша высота парковочной орбиты (м)

# Ключи в словарях данных
DATA_KEYS = ["time", "altitude", "mass",
             "position_x", "position_y", "position_z",
             "velocity", "velocity_x", "velocity_y", "velocity_z",
             "apoapsis", "periapsis",
             "vessel_angle", "mun_angle", "phase_angle"]


# Состояния FSM
class FlightState(Enum):
    PRELAUNCH = auto()
    LAUNCH = auto()
    GRAVITY_TURN = auto()
    COASTING_TO_SPACE = auto()
    CIRCULARIZATION_WAITING = auto()
    CIRCULARIZATION = auto()
    ORBITING = auto()


def calculate_full_transfer():
    # 1. Радиусы от центра Кербина
    # r1 - радиус нашей круговой орбиты
    r1 = KERBIN_R + KERBIN_ORBIT_R
    # r2 - радиус орбиты Муны (уже от центра)
    r2 = MUN_ORBIT_R

    # 2. Большая полуось переходного эллипса
    a_trans = (r1 + r2) / 2

    # 3. Расчет скоростей (уравнение vis-viva)
    # Скорость на круговой орбите 190км
    v_circular = np.sqrt(KERBIN_GRAV / r1)

    # Скорость в перицентре перехода (точка старта к Муне)
    v_per = np.sqrt(KERBIN_GRAV * (2 / r1 - 1 / a_trans))

    # Необходимая Delta-V
    delta_v = v_per - v_circular

    # 4. Время полета (половина периода эллипса в секундах)
    t_transfer = np.pi * np.sqrt(a_trans ** 3 / KERBIN_GRAV)

    # 5. Угловой расчет
    # Угловая скорость Муны (рад/с)
    omega_mun = np.sqrt(KERBIN_GRAV / r2 ** 3)

    # Угол, который пройдет Муна за время полета корабля (в градусах)
    alpha_mun = (omega_mun * t_transfer) * (180 / np.pi)

    # Фазовый угол для старта
    phase_angle = 180 - alpha_mun

    return t_transfer, phase_angle, delta_v, r1, r2


def show_plots(phase_angle, r1, r2):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    # --- ГРАФИК 1: ФИЗИКА АТМОСФЕРЫ ---
    altitudes = np.linspace(0, 75000, 500)
    # Используем формулу давления P = P0 * exp(-h/H)
    pressures = KERBIN_PRESSURE * np.exp(-altitudes / H_SCALE)

    ax1.plot(pressures / 1000, altitudes / 1000, color="royalblue", linewidth=2)
    ax1.axhline(y=45, color="orange", linestyle="--", label="Лимит поворота (45 км)")
    ax1.axhline(y=70, color="red", linestyle="-", alpha=0.5, label="Граница атмосферы (70 км)")
    ax1.fill_betweenx(altitudes / 1000, 0, pressures / 1000, color="royalblue", alpha=0.1)

    ax1.set_title("Модель атмосферы Кербина")
    ax1.set_xlabel("Давление (кПа)")
    ax1.set_ylabel("Высота (км)")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    # --- ГРАФИК 2: ГЕОМЕТРИЯ ПЕРЕЛЕТА ---
    theta = np.linspace(0, 2 * np.pi, 200)
    # Масштабируем в миллионы метров (10^6) для красоты осей
    ax2.plot((r1 / 1e6) * np.cos(theta), (r1 / 1e6) * np.sin(theta), "b--", alpha=0.4, label="Орбита ожидания")
    ax2.plot((r2 / 1e6) * np.cos(theta), (r2 / 1e6) * np.sin(theta), "darkgrey", label="Орбита Муны")

    # Корабль в момент t=0
    ax2.plot([r1 / 1e6], [0], "ro", label="Точка импульса")

    # Цель (Муна) под углом
    phi_rad = np.radians(phase_angle)
    ax2.plot([(r2 / 1e6) * np.cos(phi_rad)], [(r2 / 1e6) * np.sin(phi_rad)], "go", markersize=10,
             label="Положение Муны")

    # Линии векторов фазового угла
    ax2.plot([0, r1 / 1e6], [0, 0], "r-", alpha=0.2)
    ax2.plot([0, (r2 / 1e6) * np.cos(phi_rad)], [0, (r2 / 1e6) * np.sin(phi_rad)], "g-", alpha=0.2)

    ax2.set_aspect("equal")
    ax2.set_title(f"Трансферное окно: {phase_angle:.2f}°")
    ax2.legend(loc="lower left")
    ax2.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    t_transfer, phase_angle, delta_v, r1, r2 = calculate_full_transfer()

    print(f"{" ТЕОРЕТИЧЕСКИЙ РАСЧЕТ ":#^40}")
    print(f"Требуемая Delta-V: {delta_v:.2f} м/с")
    print(f"Время полета (t_transfer): {t_transfer / 3600:.2f} часов")
    print(f"Угол для старта (phase_angle): {phase_angle:.2f} градусов")
    print(f"{"#" * 40}")

    show_plots(phase_angle, r1, r2)