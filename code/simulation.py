import pandas as pd
from utils import normalize_data, data_to_csv, get_starting_data
from physics import *
import numpy as np


class Vessel:
    def __init__(self, starting_data: dict[str, float]):
        # Относительно этого вектора измеряются pitch и heading (как в KSP мы начинаем со стартовой площадки)
        self.base_orientation = np.array([starting_data["pitch"] - 90, starting_data["heading"], starting_data["roll"]])

        # Параметры ступеней
        self.stages = [
            {   # S14, "Грохот" х10
                "number": 14,
                "thrust_vac": 1500000 * 10,  # 15 000 кН
                "thrust_sea": 1379000 * 10,  # 13 790 кН
                "m0": 899000,
                "fuel": 592000,
                "isp_vac": 310,
                "isp_sea": 285
            },
            {   # S12, "Вектор" x4
                "number": 12,
                "thrust_vac": 1000000 * 4,
                "thrust_sea": 936510 * 4,
                "m0": 169000,
                "fuel": 53800,
                "isp_vac": 315,
                "isp_sea": 295
            },
            {   # S10, "Вектор" x1
                "number": 10,
                "thrust_vac": 1000000,
                "thrust_sea": 936510,
                "m0": 88900.0,
                "fuel": 32000.0,
                "isp_vac": 315.0,
                "isp_sea": 295.0
            },
        ]
        self.current_stage_index = 0
        self.stage = self.stages[self.current_stage_index]
        self.current_stage_number = self.stage["number"]
        self.mass = self.stage["m0"]
        self.fuel = self.stage["fuel"]

        # Начальное положение: на поверхности (по оси Y, Кербин центрирован в 0,0)
        self.pos = np.array([starting_data["position_x"], starting_data["position_y"], starting_data["position_z"]])
        self.vel = np.array([0.0, 0.0, 0.0])
        self.accel = np.array([0.0, 0.0, 0.0])
        self.throttle = 0
        self.pitch = 90
        self.heading = 90

    # Загрузка данных ступени
    def _load_stage_data(self, index: int) -> None:
        stage = self.stages[index]
        self.mass = stage["m0"]
        self.thrust = stage["thrust"]
        self.fuel = stage["fuel"]

    # Активация следующей ступени
    def activate_next_stage(self) -> bool:
        if self.current_stage_index < len(self.stages) - 1:
            self.current_stage_index += 1

            self._load_stage_data(self.current_stage_index)

            # print(f"[{self.current_stage_index - 1} -> {self.current_stage_index}] Ступень отделена. "
            #       f"Новая масса: {self.mass / 1000:.1f}т, Тяга: {self.thrust / 1000:.1f}кН")
            return True
        else:
            print("Последняя ступень уже активна. Сбрасывать нечего.")
            return False

    def target_pitch_and_heading(self, pitch: float, heading: float) -> None:
        self.pitch = pitch + self.base_orientation()
        self.heading = heading

    # Получение высоты над поверхностью планеты
    def altitude(self, planet: Planet) -> float:
        return np.linalg.norm(self.pos) - planet.radius

    # Получение скорости относительно планеты
    def velocity(self, planet: Planet) -> float:
        return np.linalg.norm(self.vel)

    # Суммирование всех сил, действующих на корабль (второй закон Ньютона)
    def get_forces(self, pos: np.ndarray, vel: np.ndarray, mass: float) -> np.ndarray:
        alt = self.altitude(KERBIN)

        f_grav = calculate_gravity_force(KERBIN, mass, pos)
        f_drag = calculate_drag_force(KERBIN, alt, vel, DRAG_COEFFICIENT, VESSEL_AREA)
        f_thrust = calculate_thrust_force(KERBIN, alt, self.throttle, self.pitch, self.heading, self.stage)
        # print(f_grav, f_drag, f_thrust, f_grav + f_drag + f_thrust)
        return f_grav + f_drag + f_thrust

    # Такт обновления физики (метод Верле второго порядка)
    def update_physics(self, dt: float):
        # Ускорение в текущий момент времени t (второй закон Ньютона)
        f_curr = self.get_forces(self.pos, self.vel, self.mass)
        self.accel = f_curr / self.mass

        # Положение на основе текущих скорости и ускорения: r(t + dt) = r(t) + v(t)*dt + 0.5 * a(t) * dt^2
        new_pos = self.pos + self.vel * dt + 0.5 * self.accel * dt ** 2

        # "Полушаговая" скоорость для расчёта в новой точке (предиктор)
        v_mid = self.vel + self.accel * dt

        # Рассчитываем ускорение в новый момент времени t + dt
        # Мы учитываем изменение массы (упрощенно считаем расход линейным за dt)
        dm = calculate_fuel_consumption(self.stage, self.throttle, dt)
        new_mass = self.mass - dm

        f_next = self.get_forces(new_pos, v_mid, new_mass)
        new_accel = f_next / new_mass

        # Обновляем скорость: v(t + dt) = v(t) + 0.5 * (a(t) + a(t+dt)) * dt
        self.vel = self.vel + 0.5 * (self.accel + new_accel) * dt

        self.pos = new_pos
        self.mass = new_mass


def run_simulation(starting_data: dict[str, float], dt: float, render_time: float) -> pd.DataFrame:
    vessel = Vessel(starting_data)
    state = FlightState.PRELAUNCH
    prev_state = None
    t = 0
    last_stage_time = 0
    last_render_time = 0

    # Инициализация словаря для данных
    data = {key: [] for key in DATA_KEYS}

    try:
        # Основной цикл управления и сбора данных
        # Работаем, пока не выйдем на круговую орбиту
        while state != FlightState.ORBITING:
            pos = vessel.pos
            vel = vessel.vel

            if state != FlightState.PRELAUNCH:
                vessel.update_physics(dt)

            if t > last_render_time + render_time:
                # Телеметрия
                data["time"].append(t)
                data["altitude"].append(vessel.altitude(KERBIN))
                data["mass"].append(vessel.mass)

                data["position_x"].append(pos[0])
                data["position_y"].append(pos[1])
                data["position_z"].append(pos[2])

                data["velocity"].append(vessel.velocity(KERBIN))
                data["velocity_x"].append(vessel.vel[0])
                data["velocity_y"].append(vessel.vel[1])
                data["velocity_z"].append(vessel.vel[2])

                # Орбитальные параметры
                data["apoapsis"].append(calculate_apoapsis_altitude(KERBIN, pos, vel))
                data["periapsis"].append(calculate_periapsis_altitude(KERBIN, pos, vel))

                # Углы
                # data["vessel_angle"].append(get_angle(reference_frame, vessel))
                # data["mun_angle"].append(get_angle(reference_frame, mun))
                # data["phase_angle"].append(get_phase_angle(reference_frame, vessel, mun))
                data["vessel_angle"].append(0)
                data["mun_angle"].append(0)
                data["phase_angle"].append(0)
                last_render_time = t

            # Условие сброса
            if (vessel.fuel < 0.05 * vessel.stage["fuel"]) and (t - last_stage_time > 1.5):
                print(f"[{int(t)}с] Сброс ступени S{vessel.current_stage_number}. Активация следующей ступени...")
                vessel.activate_next_stage()
                last_stage_time = t

            if state != prev_state:
                prev_state = state
                state_str = f" [{int(t)}с] СОСТОЯНИЕ: {state.name} "
                print(f"\n{state_str:-^60}\n")

            if state == FlightState.PRELAUNCH:
                for i in range(3, 0, -1):
                    print(f"{i}...")
                    t += 1 - dt
                print("Пуск!")
                vessel.throttle = 1.0
                vessel.target_pitch_and_heading(90, 90)
                state = FlightState.LAUNCH

            elif state == FlightState.LAUNCH:
                if vessel.altitude(KERBIN) > GRAV_TURN_START:
                    state = FlightState.GRAVITY_TURN

            elif state == FlightState.GRAVITY_TURN:
                target_pitch = calculate_pitch(vessel.altitude(KERBIN), GRAV_TURN_START, GRAV_TURN_CEIL, 0.5)
                vessel.target_pitch_and_heading(target_pitch, 90)

                if calculate_apoapsis_altitude(KERBIN, pos, vel) >= KERBIN_ORBIT_R:
                    vessel.throttle = 0.0
                    print(f"Апоцентр {KERBIN_ORBIT_R}м достигнут. Инерциальный полет.")

                    vessel.target_pitch_and_heading(90, 90)
                    state = FlightState.CIRCULARIZATION_WAITING

            elif state == FlightState.CIRCULARIZATION_WAITING:
                vessel.target_pitch_and_heading(0, 90)
                if vessel.altitude(KERBIN) >= KERBIN.atmosphere_height and \
                        calculate_time_to_apoapsis(KERBIN, pos, vel) < 15:

                    print("Точка манёвра достигнута. Начало циркуляризации.")
                    state = FlightState.CIRCULARIZATION

            elif state == FlightState.CIRCULARIZATION:
                vessel.target_pitch_and_heading(0, 90)
                until_pe = KERBIN_ORBIT_R - calculate_periapsis_altitude(KERBIN, pos, vel)
                # Условие выхода на стабильную орбиту
                if until_pe <= 0 or calculate_periapsis_altitude(KERBIN, pos, vel) >= \
                        calculate_apoapsis_altitude(KERBIN, pos, vel) * 0.98:

                    vessel.throttle = 0.0
                    print("Орбита сформирована.")
                    state = FlightState.ORBITING
                else:
                    # Плавная тяга для точности
                    vessel.throttle = 0.05 if until_pe < 5000 else 1.0

            t += dt
    except KeyboardInterrupt:
        print("Симуляция остановлена преждевременно.\n")

    print("Сбор данных завершён.\n")

    normalize_data(data)
    return pd.DataFrame(data)


if __name__ == "__main__":
    print("Загрузка стартовых данных из data/ksp_data_orbit.csv...")
    starting_data = get_starting_data("ksp_data_orbit.csv")
    if starting_data is None:
        print("Загрузка не удалась, стартовые данные по умолчанию.")
        starting_data = {key: 0 for key in DATA_KEYS}
        starting_data["altitude"] = 85
        starting_data["position_z"] = 600000

    print(f"\n{" СИМУЛЯЦИЯ ПОЛЁТА ":#^60}")
    data = run_simulation(starting_data, 0.01, 0.5)
    print(f"{" ПОЛЁТ ЗАВЕРШЁН ":#^60}")

    data_to_csv(data, "sim_data_orbit.csv")
    print("\nДанные симуляции о выводе на орбиту экспортированы в sim_data_orbit.csv")
    print("Построение визуализации...")
    # show_orbit_plot(data)
