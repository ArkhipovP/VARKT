import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from enum import Enum, auto


@dataclass
class Planet:
    name: str                        # Название
    grav_parameter: float            # Гравитационный параметр (м^3/с^2)
    radius: float                    # Радиус (м)
    atmosphere_height: float = 0.0   # Высота атмосферы (м), 0 для безвоздушных тел
    sea_level_density: float = 0.0   # Плотность на уровне моря (кг/м^3)
    sea_level_pressure: float = 0.0  # Давление на уровне моря (кг/м^3)
    scale_height: float = 0.0        # Шкала высот (м)


G0 = 9.80665


# Шкала высот для участка атмосферы
def scale_height_points(h1: float, h2: float, p1: float, p2: float) -> float:
    return (h2 - h1) / np.log(p1 / p2)


# Давление
def calculate_pressure(planet: Planet, altitude: float) -> float:
    # Выше границы атмосферы давления нет (в реализации KSP)
    if altitude >= planet.atmosphere_height:
        return 0.0

    # p = p0 * e^(-h / H)
    return planet.sea_level_pressure * np.exp(-altitude / planet.scale_height)


# Плотность
def calculate_density(planet: Planet, altitude: float) -> float:
    if altitude >= planet.atmosphere_height:
        return 0.0

    # ρ = ρ0 * e^(-h / H)
    return planet.sea_level_density * np.exp(-altitude / planet.scale_height)


# Удельный импульс
# def calculate_isp(planet: Planet, altitude: float, isp_vac: float, isp_sea: float):
#     if planet.sea_level_pressure > 0:
#         p_rel = min(calculate_pressure(planet, altitude) / planet.sea_level_pressure, 1.0)
#     else:
#         p_rel = 0
#     return isp_vac - (isp_vac - isp_sea) * p_rel


# Расход топлива (m')
def calculate_fuel_consumption(stage: dict[str, float], throttle: float, dt: float) -> float:
    if throttle <= 0:
        return 0.0

    # Формула: m' = F_vac / (Isp_vac * g0)
    # Это секундный расход массы при 100% тяге
    m_dot_max = stage["thrust_vac"] / (stage["isp_vac"] * G0)

    # Расход за конкретный промежуток времени с учетом дросселя
    dm = m_dot_max * throttle * dt

    return dm


# Массив давлений из массива высот
def calculate_pressures(planet: Planet, altitudes: np.ndarray) -> np.ndarray:
    altitudes = np.asarray(altitudes)

    return np.where(  # буквально как тернарный оператор ну типа
        altitudes >= planet.atmosphere_height,
        0.0,
        planet.sea_level_pressure * np.exp(-altitudes / planet.scale_height)
    )


# Расчёт сил (векторный)

# Сила тяжести
def calculate_gravity_force(planet: Planet, mass: float, pos_vector: np.ndarray) -> np.ndarray:
    r_mag = np.linalg.norm(pos_vector)

    # Закон всемирного тяготения: F = GM * m / r^2
    f_mag = (planet.grav_parameter * mass) / (r_mag ** 2)

    # Вектор направления (противоположно вектору положения)
    unit_vector = -pos_vector / r_mag

    return unit_vector * f_mag


# Сила сопротивления
def calculate_drag_force(planet: Planet, altitude: float, velocity_vector: np.ndarray,
                         drag_coefficient: float, reference_area: float) -> np.ndarray:

    if altitude >= planet.atmosphere_height or planet.scale_height == 0:
        return np.array([0.0, 0.0, 0.0])

    v_mag = np.linalg.norm(velocity_vector)
    if v_mag < 0.1:
        return np.array([0.0, 0.0, 0.0])

    density = calculate_density(planet, altitude)

    # Сила сопротивления: Fd = 0.5 * rho * v^2 * Cd * S
    f_mag = 0.5 * density * (v_mag ** 2) * drag_coefficient * reference_area

    # Вектор направления (против скорости)
    unit_velocity = velocity_vector / v_mag

    return -unit_velocity * f_mag


# Вектор силы тяги
def calculate_thrust_force(planet: Planet, altitude: float, throttle: float, pitch: float, heading: float,
                           stage: dict[str, float]) -> np.ndarray:

    if throttle <= 0:
        return np.array([0.0, 0.0, 0.0])

    pressure = calculate_pressure(planet, altitude)

    # Линейная интерполяция тяги между вакуумом и уровнем моря
    # В KSP: F_current = F_vac - (F_vac - F_sea) * P_atm
    p_atm = min(pressure / planet.sea_level_pressure, 1) if planet.sea_level_pressure > 0 else 0
    current_max_thrust = stage["thrust_vac"] - (stage["thrust_vac"] - stage["thrust_sea"]) * p_atm
    thrust_magnitude = current_max_thrust * throttle

    p_rad = np.radians(pitch)
    h_rad = np.radians(heading)

    fy = thrust_magnitude * np.sin(p_rad)
    h_mag = thrust_magnitude * np.cos(p_rad)
    fx = h_mag * np.sin(h_rad)
    fz = h_mag * np.cos(h_rad)

    return np.array([fx, fy, fz])


# "Потолок" гравитационного разворота
def calculate_grav_turn_ceil(planet: Planet, threshold=0.0001) -> float:
    if planet.scale_height == 0:
        return 0.0

    # "Потолок" гравитационного разворота - высота, на которой давление достигает определённого порога (0.01%)
    cutoff_height = -planet.scale_height * np.log(threshold)
    return min(cutoff_height, planet.atmosphere_height)


# Расчёт угла тангажа
def calculate_pitch(altitude: float, turn_start: float, turn_ceil: float, n: float = 0.5) -> float:
    if altitude < turn_start:
        return 90.0
    if altitude > turn_ceil:
        return 0.0

    fraction = (altitude - turn_start) / (turn_ceil - turn_start)
    return 90.0 * (1.0 - fraction ** n)


# Расчёт орбитальных характеристик

# Удельная орбитальная энергия (ε)
def calculate_specific_energy(planet: Planet, pos_vector: np.ndarray, velocity_vector: np.ndarray) -> float:
    r_mag = np.linalg.norm(pos_vector)
    v_mag = np.linalg.norm(velocity_vector)

    if r_mag == 0: return 0
    return (v_mag ** 2 / 2.0) - (planet.grav_parameter / r_mag)


# Эксцентриситет орбиты (e)
def calculate_eccentricity(planet: Planet, pos_vector: np.ndarray, velocity_vector: np.ndarray) -> float:
    energy = calculate_specific_energy(planet, pos_vector, velocity_vector)
    # Угловой момент (h) для 2D случая в плоскости XY
    h = pos_vector[0] * velocity_vector[1] - pos_vector[1] * velocity_vector[0]

    e_sq = 1 + (2.0 * energy * h ** 2) / (planet.grav_parameter ** 2)
    return np.sqrt(max(0, e_sq))


# Большая полуось орбиты
def calculate_semi_major_axis(start_r: float, end_r: float) -> float:
    return (start_r + end_r) / 2


# Большая полуось орбиты (через энергию)
def calculate_semi_major_axis_energy(planet: Planet, pos_vector: np.ndarray, velocity_vector:  np.ndarray) -> float:
    energy = calculate_specific_energy(planet, pos_vector, velocity_vector)
    if energy == 0:
        return float("inf")  # Гиперболическая траектория
    return -planet.grav_parameter / (2.0 * energy)


# Период орбиты
def calculate_orbit_period(planet: Planet, pos_vector: np.ndarray, velocity_vector: np.ndarray):
    a = calculate_semi_major_axis_energy(planet, pos_vector, velocity_vector)
    if a <= 0:
        return float("inf")  # Гиперболическая траектория

    # T = 2 * pi * sqrt(a^3 / mu)
    return 2 * np.pi * np.sqrt(a ** 3 / planet.grav_parameter)


# Высота апоцентра
def calculate_apoapsis_altitude(planet: Planet, pos_vector: np.ndarray, velocity_vector: np.ndarray) -> float:
    a = calculate_semi_major_axis_energy(planet, pos_vector, velocity_vector)
    e = calculate_eccentricity(planet, pos_vector, velocity_vector)
    return a * (1.0 + e) - planet.radius


# Высота перицентра
def calculate_periapsis_altitude(planet: Planet, pos_vector: np.ndarray, velocity_vector: np.ndarray) -> float:
    a = calculate_semi_major_axis_energy(planet, pos_vector, velocity_vector)
    e = calculate_eccentricity(planet, pos_vector, velocity_vector)
    return a * (1.0 - e) - planet.radius


# Время до апоцентра  TODO: some stuff
def calculate_time_to_apoapsis(planet: Planet, pos_vector: np.ndarray, velocity_vector: np.ndarray) -> float:
    a = calculate_semi_major_axis_energy(planet, pos_vector, velocity_vector)
    e = calculate_eccentricity(planet, pos_vector, velocity_vector)

    if e >= 1.0:
        return 0.0  # Для парабол/гипербол апоцентра нет

    r_mag = np.linalg.norm(pos_vector)
    v_dot_r = np.dot(pos_vector, velocity_vector)

    # Эксцентрическая аномалия (E)
    # cos(E) = (1 - r/a) / e
    cos_E = (1.0 - r_mag / a) / max(e, 1e-9)
    cos_E = np.clip(cos_E, -1.0, 1.0)
    E = np.arccos(cos_E)

    # Уточняем полусферу (если v_dot_r > 0, мы удаляемся от перицентра)
    if v_dot_r < 0:
        E = 2 * np.pi - E

    # 2. Средняя аномалия (M)
    M = E - e * np.sin(E)

    # 3. Время от перицентра (t_p)
    period = calculate_orbit_period(planet, pos_vector, velocity_vector)
    t_p = (M / (2 * np.pi)) * period

    # Время до апоцентра, апоцентр находится на половине периода (t = T/2)
    t_ap = (period / 2.0) - t_p

    # Если мы уже пролетели апоцентр в этом витке
    if t_ap < 0:
        t_ap += period

    return t_ap


KERBIN = Planet(
    name="Kerbin",
    grav_parameter=3.5316e12,
    radius=600000,
    atmosphere_height=70000,
    sea_level_density=1.225,
    sea_level_pressure=101325,
    scale_height=5600
)


MUN = Planet(
    name="Mun",
    grav_parameter=6.5138e10,
    radius=200000,
)

# Константы ракеты
VESSEL_AREA = 12.56      # Примерная площадь для тяжелой ракеты (радиус ~2м)
DRAG_COEFFICIENT = 0.25  # Коэффициент лобового сопротивления

# Данные для гравитационного поворота
GRAV_TURN_START = 1000   # Высота для начала
GRAV_TURN_CEIL = 51500  # "Потолок" поворота гравитационного манёвра

# Данные для трансфера к Муне
MUN_ORBIT_R = 12000000   # Радиус орбиты Муны (м)
KERBIN_ORBIT_R = 190000  # Высота парковочной орбиты (как в Аполлоне 11)


# https://wiki.kerbalspaceprogram.com/wiki/Kerbin, Atmosphere
wiki_atmosphere_data = [
    (0, 101325.0),
    (2500, 69015.0),
    (5000, 45625.0),
    (7500, 29126.0),
    (10000, 17934.0),
    (15000, 6726.0),
    (20000, 2549.0),
    (25000, 993.6),
    (30000, 404.1),
    (40000, 79.77),
    (50000, 15.56),
    (60000, 2.387),
]

# Ключи в словарях данных
DATA_KEYS = [
    "time", "altitude", "mass", "speed",
    "pos_x", "pos_y", "pos_z",
    "pitch", "heading", "roll",
    "vel_x", "vel_y", "vel_z",
    "horizontal_speed", "vertical_speed",
    "thrust_x", "thrust_y", "thrust_z",
    "grav_x", "grav_y", "grav_z",
    "drag_x", "drag_y", "drag_z",
    "apoapsis", "periapsis",
    "vessel_angle", "mun_angle", "phase_angle"
]


# Состояния FSM
class FlightState(Enum):
    PRELAUNCH = auto()
    LAUNCH = auto()
    GRAVITY_TURN = auto()
    COASTING_TO_SPACE = auto()
    CIRCULARIZATION_WAITING = auto()
    CIRCULARIZATION = auto()
    ORBITING = auto()



# Гомановский перелёт (Trans-Lunar Injection)

# Delta-V для TLI
def calculate_tli_delta_v(start_r: float, end_r: float) -> float:
    sm_axis = calculate_semi_major_axis(start_r, end_r)

    # Скорость на текущей круговой орбите
    start_velocity = np.sqrt(KERBIN.grav_parameter / start_r)

    # Скорость в перицентре переходного эллипса (vis-viva)
    end_velocity = np.sqrt(KERBIN.grav_parameter * (2 / start_r - 1 / sm_axis))

    return end_velocity - start_velocity


# Время полёта во время TLI
def calculate_tli_time(start_r: float, end_r: float) -> float:
    sm_axis = calculate_semi_major_axis(start_r, end_r)
    # T = 2 * pi * sqrt(a^3 / GM) / 2 = pi * sqrt(a^3 / GM)
    return np.pi * np.sqrt(sm_axis ** 3 / KERBIN.grav_parameter)


# Трансферное окно
def calculate_tli_phase_angle(start_r: float, end_r: float) -> float:
    tli_time = calculate_tli_time(start_r, end_r)

    # Угловая скорость цели (Муны) на круговой орбите Кербина (рад/с)
    angular_velocity = np.sqrt(KERBIN.grav_parameter / end_r ** 3)

    # Угол, на который переместится цель за время полета корабля (в градусах)
    angle_lead = (angular_velocity * tli_time) * (180 / np.pi)

    # Корабль должен "догнать" точку встречи, пройдя 180 градусов.
    return 180 - angle_lead


def calculate_tli_data() -> dict[str, float]:
    start_r = KERBIN_ORBIT_R + KERBIN.radius
    end_r = MUN_ORBIT_R

    delta_v = calculate_tli_delta_v(start_r, end_r)
    t_flight = calculate_tli_time(start_r, end_r)
    target_phase = calculate_tli_phase_angle(start_r, end_r)

    return {
        "delta_v": delta_v,
        "transfer_time": t_flight,
        "phase_angle": target_phase,
        "start_radius": start_r,
        "end_radius": end_r
    }


def show_tli_plots(tli_data: dict[str, float]) -> None:
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    start_r = tli_data["start_radius"]
    end_r = tli_data["end_radius"]

    # График давления
    altitudes = np.linspace(0, 75000, 500)
    pressures = calculate_pressures(KERBIN, altitudes)

    axs[0].set_title("Модель атмосферы Кербина")
    axs[0].set_xlabel("Давление (кПа)")
    axs[0].set_ylabel("Высота (км)")
    axs[0].grid(True, linestyle=":", alpha=0.6)

    axs[0].plot(pressures / 1000, altitudes / 1000, color="blue", linewidth=2)
    axs[0].axhline(y=51.5, color="orange", linestyle="--", label="Лимит поворота (51.5 км)")
    axs[0].axhline(y=70, color="red", linestyle="--", alpha=0.5, label="Граница атмосферы (70 км)")
    axs[0].fill_betweenx(altitudes / 1000, 0, pressures / 1000, color="royalblue", alpha=0.1)
    axs[0].legend()

    # --- ГРАФИК 2: ГЕОМЕТРИЯ ПЕРЕЛЕТА ---
    theta = np.linspace(0, 2 * np.pi, 200)
    phi_rad = np.radians(tli_data["phase_angle"])

    axs[1].set_aspect("equal")
    axs[1].set_title(f"Трансферное окно: {tli_data["phase_angle"]:.2f}°")
    axs[1].axis("off")

    # Масштабируем в миллионы метров (10^6) для красоты осей
    axs[1].plot((start_r / 1e6) * np.cos(theta), (start_r / 1e6) * np.sin(theta), "b--", alpha=0.5,
                label="Начальная орбита (орбита вокруг Кербина)")
    axs[1].plot((end_r / 1e6) * np.cos(theta), (end_r / 1e6) * np.sin(theta), "g--", alpha=0.5,
                label="Конечная орбита (орбита Муны)")

    # Линии векторов фазового угла
    axs[1].plot([0, start_r / 1e6], [0, 0], color="gray", linestyle="-", alpha=0.2)
    axs[1].plot([0, (end_r / 1e6) * np.cos(phi_rad)], [0, (end_r / 1e6) * np.sin(phi_rad)], color="gray", linestyle="-",
                alpha=0.2)

    # Корабль
    axs[1].plot([start_r / 1e6], [0], "bo", markersize=7.5, label="Положение корабля")

    # Муна под углом
    axs[1].plot([(end_r / 1e6) * np.cos(phi_rad)], [(end_r / 1e6) * np.sin(phi_rad)], "go", markersize=7.5,
             label="Положение Муны")

    axs[1].legend(loc="lower left")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    print(f"{" ТЕОРЕТИЧЕСКИЙ РАСЧЕТ ":#^60}")
    print("(На основе данных с https://wiki.kerbalspaceprogram.com/wiki/Kerbin)\n")

    print(f"{" КЕРБИН ":-^60}\n")

    scale_heights = [  # Среднее среди попарно выбранных замеров давления
        scale_height_points(p1[0], p2[0], p1[1], p2[1])
        for p1, p2 in zip(wiki_atmosphere_data, wiki_atmosphere_data[1:])]

    print(f"Шкала высот Кербина (на основе таблицы давлений): {sum(scale_heights) / len(scale_heights)}")
    print(f"(как константа берётся {KERBIN.scale_height})\n")

    print(f"{" ВЫВОД НА ОРБИТУ ":-^60}\n")

    print(f"Итоговая высота гравитационного разворота (0.01% давления): {calculate_grav_turn_ceil(KERBIN)}м")
    print(f"(как константа берётся {GRAV_TURN_CEIL}м)\n")

    print(f"{" TRANS-LUNAR INJECTION ":-^60}\n")

    tli_data = calculate_tli_data()

    print(f"Требуемая Delta-V: {tli_data["delta_v"]:.2f} м/с")
    print(f"Время полета: {tli_data["transfer_time"] / 3600:.2f} часов")
    print(f"Угол для старта (phase_angle): {tli_data["phase_angle"]:.2f} градусов\n")
    print(f"{"#" * 60}")

    show_tli_plots(tli_data)
