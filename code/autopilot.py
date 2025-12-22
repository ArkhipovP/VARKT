import os
import time
from enum import Enum, auto

import krpc
import pandas as pd
from math_utils import get_angle, get_phase_angle, show_orbit_plot
from constants import *


class FlightState(Enum):
    PRELAUNCH = auto()                # Ожидание старта
    LAUNCH = auto()                   # Вертикальный взлет
    GRAVITY_TURN = auto()             # Гравитационный маневр (до 45 км)
    CIRCULARIZATION_WAITING = auto()  # Инерциальный полет в ожидании точки апоцентра
    CIRCULARIZATION = auto()          # Циркуляризация: прожиг в апоцентре
    ORBITING = auto()                 # Достигнута круглая орбита


def control_vessel(vessel, conn):
    # Потоки данных
    ut = conn.add_stream(getattr, conn.space_center, "ut")
    altitude = conn.add_stream(getattr, vessel.flight(), "mean_altitude")
    apoapsis = conn.add_stream(getattr, vessel.orbit, "apoapsis_altitude")
    periapsis = conn.add_stream(getattr, vessel.orbit, "periapsis_altitude")
    velocity = conn.add_stream(getattr, vessel.flight(vessel.orbit.body.reference_frame), "speed")
    mass = conn.add_stream(getattr, vessel, "mass")
    time_to_ap = conn.add_stream(getattr, vessel.orbit, "time_to_apoapsis")

    # Словарь для данных
    data = {
        "time": [],
        "altitude": [],
        "velocity": [],
        "mass": [],
        "apoapsis": [],
        "periapsis": [],
        "vessel_angle": [],
        "mun_angle": [],
        "phase_angle": [],
    }

    prev_state = None
    state = FlightState.PRELAUNCH
    start_time = ut()
    kerbin = conn.space_center.bodies["Kerbin"]
    mun = conn.space_center.bodies["Mun"]

    # Основной цикл управления и сбора данных
    try:
        # Работаем, пока не выйдем на круговую орбиту
        while state != FlightState.ORBITING:
            curr_t = ut() - start_time

            # Сбор телеметрии (каждую итерацию)
            data["time"].append(curr_t)
            data["altitude"].append(altitude())
            data["velocity"].append(velocity())
            data["mass"].append(mass())
            data["apoapsis"].append(apoapsis())
            data["periapsis"].append(periapsis())
            data["vessel_angle"].append(get_angle(kerbin.reference_frame, vessel))
            data["mun_angle"].append(get_angle(kerbin.reference_frame, mun))
            data["phase_angle"].append(get_phase_angle(kerbin.reference_frame, vessel, mun))

            # Проверка ступеней
            if velocity() > 50 and vessel.thrust < 0.1:
                vessel.control.activate_next_stage()
                print(f"[{int(curr_t)}с] Активация следующей ступени...")

            if state != prev_state:
                prev_state = state
                state_str = f" [{int(curr_t)}с] СОСТОЯНИЕ: {state.name} "
                print(f"{state_str:-^40}")

            if state == FlightState.PRELAUNCH:
                for i in range(3, 0, -1):
                    print(f"{i}...")
                    time.sleep(1)
                print("Пуск!")
                vessel.control.throttle = 1.0
                vessel.control.activate_next_stage()
                vessel.auto_pilot.target_pitch_and_heading(90, 90)
                state = FlightState.LAUNCH

            elif state == FlightState.LAUNCH:
                if altitude() > 250:
                    state = FlightState.GRAVITY_TURN

            elif state == FlightState.GRAVITY_TURN:
                # Наклон до границы GRAV_TURN_R (45 км)
                frac = (altitude() - 250) / (GRAV_TURN_R - 250)
                vessel.auto_pilot.target_pitch_and_heading(90 - (frac * 90), 90)

                if apoapsis() >= KERBIN_ORBIT_R:
                    vessel.control.throttle = 0.0
                    print(f"Апоцентр {KERBIN_ORBIT_R}м достигнут. Инерциальный полет.")
                    state = FlightState.CIRCULARIZATION_WAITING

            elif state == FlightState.CIRCULARIZATION_WAITING:
                vessel.auto_pilot.target_pitch_and_heading(0, 90)
                if altitude() >= KERBIN_ATMOSPHERE_R and time_to_ap() < 15:
                    print("Точка манёвра достигнута. Начало циркуляризации.")
                    state = FlightState.CIRCULARIZATION

            elif state == FlightState.CIRCULARIZATION:
                vessel.auto_pilot.target_pitch_and_heading(0, 90)
                until_pe = KERBIN_ORBIT_R - periapsis()

                # Условие выхода на стабильную орбиту
                if until_pe <= 0 or periapsis() >= apoapsis() * 0.98:
                    vessel.control.throttle = 0.0
                    print("Орбита сформирована.")
                    state = FlightState.ORBITING
                # Условие выхода из эффективной зоны апоцентра
                elif 30 < time_to_ap() < (vessel.orbit.period - 30):
                    vessel.control.throttle = 0.0
                    print("Выход из эффективной зоны апоцентра. Ждем следующий виток...")
                    state = FlightState.CIRCULARIZATION_WAITING
                else:
                    # Плавная тяга для точности
                    vessel.control.throttle = 0.05 if until_pe < 5000 else 1.0

            time.sleep(0.1)  # Частота обновления данных
    except KeyboardInterrupt:
        print("Автопилот остановлен преждевременно.")

    print("Сбор данных завершён.")
    vessel.auto_pilot.disengage()

    min_length = len(min(data.values(), key=len))

    for key in data.keys():
        data[key] = data[key][:min_length]

    df_ksp = pd.DataFrame(data)

    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_folder = os.path.join(current_dir, "..", "data")
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    file_name = os.path.join(data_folder, "ksp_data_orbit.csv")

    df_ksp.to_csv(file_name, index=False)
    print("KSP данные о выводе на орбиту экспортированы в ksp_data_orbit.csv")
    print("Построение визуализации...")
    show_orbit_plot(data)


if __name__ == "__main__":
    print(f"{" АВТОПИЛОТ КОРАБЛЯ ":#^40}")
    conn = krpc.connect(name="Apollo 11 launch autopilot")
    control_vessel(conn.space_center.active_vessel, conn)
    control_vessel(conn.space_center.active_vessel, conn)
    print(f"{" ПОЛЁТ ЗАВЕРШЁН ":#^40}")
