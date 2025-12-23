import os
import time
import krpc
import pandas as pd
from utils import get_angle, get_phase_angle, show_orbit_plot, normalize_data, data_to_csv
from constants import *


def control_vessel(vessel, conn):
    # Константы
    kerbin = conn.space_center.bodies["Kerbin"]
    mun = conn.space_center.bodies["Mun"]
    reference_frame = kerbin.reference_frame

    # Потоки данных
    ut = conn.add_stream(getattr, conn.space_center, "ut")
    altitude = conn.add_stream(getattr, vessel.flight(), "mean_altitude")
    mass = conn.add_stream(getattr, vessel, "mass")
    speed = conn.add_stream(getattr, vessel.flight(reference_frame), "speed")
    apoapsis = conn.add_stream(getattr, vessel.orbit, "apoapsis_altitude")
    periapsis = conn.add_stream(getattr, vessel.orbit, "periapsis_altitude")
    time_to_ap = conn.add_stream(getattr, vessel.orbit, "time_to_apoapsis")

    # Векторные величины
    position = conn.add_stream(vessel.position, reference_frame)
    velocity = conn.add_stream(vessel.velocity, reference_frame)

    # Ресурсы ступеней
    current_stage = vessel.control.current_stage - 1
    liquid_fuel = conn.add_stream(vessel.resources_in_decouple_stage(current_stage - 1).amount, "LiquidFuel")
    oxidizer = conn.add_stream(vessel.resources_in_decouple_stage(current_stage - 1).amount, "Oxidizer")

    # Инициализация словаря для данных
    data = {key: [] for key in DATA_KEYS}

    prev_state = None
    state = FlightState.PRELAUNCH
    start_time = ut()
    last_stage_time = 0

    vessel.control.sas = False
    vessel.auto_pilot.engage()

    try:
        # Основной цикл управления и сбора данных
        # Работаем, пока не выйдем на круговую орбиту
        while state != FlightState.ORBITING:
            curr_t = ut() - start_time
            pos = position()
            vel = velocity()

            # Телеметрия
            data["time"].append(curr_t)
            data["altitude"].append(altitude())
            data["mass"].append(mass())

            data["position_x"].append(pos[0])
            data["position_y"].append(pos[1])
            data["position_z"].append(pos[2])

            data["velocity"].append(speed())
            data["velocity_x"].append(vel[0])
            data["velocity_y"].append(vel[1])
            data["velocity_z"].append(vel[2])

            # Орбитальные параметры
            data["apoapsis"].append(apoapsis())
            data["periapsis"].append(periapsis())

            # Углы (вычисляются через ваши функции из math_utils)
            # Передаем ref_frame, чтобы углы считались в той же системе координат
            data["vessel_angle"].append(get_angle(reference_frame, vessel))
            data["mun_angle"].append(get_angle(reference_frame, mun))
            data["phase_angle"].append(get_phase_angle(reference_frame, vessel, mun))

            # Условие сброса
            if (liquid_fuel() < 0.05 or oxidizer() < 0.05) and (ut() - last_stage_time > 0.6):
                vessel.control.activate_next_stage()
                print(f"[{int(curr_t)}с] Сброс ступени S{current_stage}. Активация следующей ступени...")
                current_stage = vessel.control.current_stage
                liquid_fuel = conn.add_stream(vessel.resources_in_decouple_stage(current_stage - 1).amount, "LiquidFuel")
                oxidizer = conn.add_stream(vessel.resources_in_decouple_stage(current_stage - 1).amount, "Oxidizer")
                last_stage_time = ut()

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
                if altitude() > GRAV_TURN_START:
                    state = FlightState.GRAVITY_TURN

            elif state == FlightState.GRAVITY_TURN:
                frac = (altitude() - GRAV_TURN_START) / (GRAV_TURN_CEIL - GRAV_TURN_START)
                frac = max(0, min(1, frac))

                target_pitch = 90 - (frac * 90)
                vessel.auto_pilot.target_pitch_and_heading(target_pitch, 90)

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

    normalize_data(data)
    return data


if __name__ == "__main__":
    print(f"{" АВТОПИЛОТ КОРАБЛЯ ":#^40}")
    conn = krpc.connect(name="Apollo 11 launch autopilot")
    data = control_vessel(conn.space_center.active_vessel, conn)
    print(f"{" ПОЛЁТ ЗАВЕРШЁН ":#^40}")

    data_to_csv(data, "ksp_data_orbit.csv")
    print("\nKSP данные о выводе на орбиту экспортированы в ksp_data_orbit.csv")
    print("Построение визуализации...")
    show_orbit_plot(data)
