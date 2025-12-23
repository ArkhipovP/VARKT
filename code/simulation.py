from utils import normalize_data, show_orbit_plot, data_to_csv
from constants import *
import numpy as np


class Vessel:
    def __init__(self):
        # Параметры ракеты остаются прежними
        self.stages = [
            {  # S14
                "thrust": 13790000.0,
                "m0": 899000.0,  # кумулятивная масса
                "fuel": 592000.0,
                "isp": 285.0  # Среднее значение для старта
            },
            {  # S12
                "thrust": 2434900.0,
                "m0": 169000.0,
                "fuel": 53800.0,
                "isp": 295.0
            },
            {  # S10
                "thrust": 936510.0,
                "m0": 88900.0,
                "fuel": 32000.0,
                "isp": 295.0
            }
        ]
        self.current_stage_index = 0
        stage = self.stages[self.current_stage_index]
        self.mass = stage["m0"]
        self.thrust = stage["thrust"]
        self.isp = stage["isp"]
        self.fuel = stage["fuel"]

        # Начальное положение: на поверхности (по оси Y, Кербин центрирован в 0,0)
        self.pos = np.array([0.0, KERBIN_R])
        self.vel = np.array([0.0, 0.0])
        self.throttle = 0.0
        self.pitch = 90.0

    def update_physics(self, dt):
        r_mag = np.linalg.norm(self.pos)
        unit_r = self.pos / r_mag
        v_mag = np.linalg.norm(self.vel)

        # 1. Логика разделения ступеней
        if self.fuel <= 0 and self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1
            new_stage = self.stages[self.current_stage_index]
            self.mass = new_stage["m0"]
            self.thrust = new_stage["thrust"]
            self.isp = new_stage["isp"]
            self.fuel = new_stage["fuel"]

        # 2. Расчет сил
        # В 2D симуляции Pitch отсчитывается от горизонта.
        # Чтобы взлететь вверх от поверхности (unit_r), нам нужно вращать вектор тяги
        angle_to_up = np.arctan2(unit_r[1], unit_r[0])
        # Угол тяги в глобальных координатах
        thrust_angle = angle_to_up + np.radians(self.pitch - 90)
        thrust_dir = np.array([np.cos(thrust_angle), np.sin(thrust_angle)])

        current_thrust = self.thrust * self.throttle if self.fuel > 0 else 0.0
        f_thrust = thrust_dir * current_thrust
        f_grav = -unit_r * (KERBIN_GRAV * self.mass / r_mag ** 2)

        f_drag = np.array([0.0, 0.0])
        if (r_mag - KERBIN_R) < KERBIN_ATMOSPHERE_R:
            rho = KERBIN_DENSITY * np.exp(-(r_mag - KERBIN_R) / H_SCALE)
            if v_mag > 0.1:
                f_drag = -(self.vel / v_mag) * (0.5 * rho * v_mag ** 2 * C_D * S_REF)

        accel = (f_thrust + f_grav + f_drag) / self.mass
        self.vel += accel * dt
        self.pos += self.vel * dt

        # 3. Расход топлива
        if self.throttle > 0 and self.fuel > 0:
            dm = (self.throttle * self.thrust) / (self.isp * 9.81) * dt
            self.mass -= dm
            self.fuel -= dm

    def altitude(self):
        return np.linalg.norm(self.pos) - KERBIN_R

    def get_orbital_elements(self):
        r_vec = self.pos
        v_vec = self.vel
        r_mag = np.linalg.norm(r_vec)
        v_mag = np.linalg.norm(v_vec)

        energy = (v_mag ** 2 / 2) - (KERBIN_GRAV / r_mag)
        a = -KERBIN_GRAV / (2 * energy)

        # Угловой момент (в 2D это скаляр)
        h = r_vec[0] * v_vec[1] - r_vec[1] * v_vec[0]

        e_sq = 1 + (2 * energy * h ** 2) / (KERBIN_GRAV ** 2)
        e = np.sqrt(max(0, e_sq))

        ap = a * (1 + e) - KERBIN_R
        pe = a * (1 - e) - KERBIN_R
        return ap, pe, e


def run_simulation():
    vessel = Vessel()
    state = FlightState.PRELAUNCH
    t = 0.0
    dt = 0.1  # Для ускорения расчета увеличим шаг

    # Полный набор ключей как в KSP скрипте
    data = {key: [] for key in DATA_KEYS}

    while state != FlightState.ORBITING and t < 2000:
        ap, pe, ecc = vessel.get_orbital_elements()
        alt = vessel.altitude()
        v_mag = np.linalg.norm(vessel.vel)

        # Логика FSM (Линейный подход)
        if state == FlightState.PRELAUNCH:
            vessel.throttle = 1.0
            state = FlightState.LAUNCH
        elif state == FlightState.LAUNCH:
            vessel.pitch = 90
            if alt > 500: state = FlightState.GRAVITY_TURN
        elif state == FlightState.GRAVITY_TURN:
            frac = max(0, min(1, (alt - 500) / (GRAV_TURN_CEIL - 500)))
            vessel.pitch = 90 - (frac * 90)  # Линейный наклон
            if ap >= KERBIN_ORBIT_R:
                vessel.throttle = 0.0
                state = FlightState.CIRCULARIZATION_WAITING
        elif state == FlightState.CIRCULARIZATION_WAITING:
            vessel.pitch = 0
            if alt >= 70000 and (KERBIN_ORBIT_R - pe) > 100:
                # В симуляции просто включаем тягу когда высоко
                state = FlightState.CIRCULARIZATION
        elif state == FlightState.CIRCULARIZATION:
            vessel.pitch = 0
            until_pe = KERBIN_ORBIT_R - pe
            vessel.throttle = 1.0 if until_pe > 5000 else 0.1
            if pe >= ap * 0.98 or until_pe <= 0:
                vessel.throttle = 0.0
                state = FlightState.ORBITING

        vessel.update_physics(dt)
        t += dt

        # ЗАПИСЬ ТЕЛЕМЕТРИИ СОГЛАСНО DATA_KEYS
        data["time"].append(t)
        data["altitude"].append(alt)
        data["mass"].append(vessel.mass)

        # Позиция (в 2D, Z оставляем 0)
        data["position_x"].append(vessel.pos[0])
        data["position_y"].append(vessel.pos[1])
        data["position_z"].append(0.0)

        # Скорость
        data["velocity"].append(v_mag)
        data["velocity_x"].append(vessel.vel[0])
        data["velocity_y"].append(vessel.vel[1])
        data["velocity_z"].append(0.0)

        # Орбитальные данные
        data["apoapsis"].append(ap)
        data["periapsis"].append(pe)

        # Углы
        v_angle = np.degrees(np.arctan2(vessel.pos[1], vessel.pos[0]))
        data["vessel_angle"].append(v_angle)
        data["mun_angle"].append(0.0)  # Для простоты Муна в точке 0 градусов
        data["phase_angle"].append(0.0 - v_angle)

    normalize_data(data)
    return data


if __name__ == "__main__":
    print(f"{" СИМУЛЯЦИЯ ПОЛЁТА ":#^40}")
    data = run_simulation()
    print(f"{" ПОЛЁТ ЗАВЕРШЁН ":#^40}")

    data_to_csv(data, "simulation_data_orbit.csv")
    print("\nДанные симуляции о выводе на орбиту экспортированы в simulation_data_orbit.csv")
    print("Построение визуализации...")
    show_orbit_plot(data)
