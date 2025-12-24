import krpc
import math
import pandas as pd
import numpy as np
from krpc.services.spacecenter import ReferenceFrame, Vessel
from utils import show_orbit_plot, normalize_data, data_to_csv
from physics import *


def get_angle(reference: ReferenceFrame, target) -> float:
    t_pos = target.position(reference)
    t_angle = math.atan2(t_pos[2], t_pos[0]) * 180 / math.pi
    return t_angle


def get_phase_angle(reference: ReferenceFrame, vessel: Vessel, target) -> float:
    return (get_angle(reference, target) - get_angle(reference, vessel)) % 360


def run_autopilot(vessel: Vessel, conn, render_time: float) -> pd.DataFrame:
    # Инициализация небесных тел
    kerbin = conn.space_center.bodies["Kerbin"]
    mun = conn.space_center.bodies["Mun"]

    # Системы координат
    reference_frame = kerbin.reference_frame
    flight = vessel.flight(reference_frame)

    # Потоки данных
    ut = conn.add_stream(getattr, conn.space_center, "ut")
    altitude = conn.add_stream(getattr, vessel.flight(), "mean_altitude")
    mass = conn.add_stream(getattr, vessel, "mass")
    speed = conn.add_stream(getattr, flight, "speed")
    apoapsis = conn.add_stream(getattr, vessel.orbit, "apoapsis_altitude")
    periapsis = conn.add_stream(getattr, vessel.orbit, "periapsis_altitude")
    time_to_ap = conn.add_stream(getattr, vessel.orbit, "time_to_apoapsis")

    # Векторные потоки
    position = conn.add_stream(vessel.position, reference_frame)
    velocity = conn.add_stream(vessel.velocity, reference_frame)

    # Инициализация данных
    data = {key: [] for key in DATA_KEYS}

    prev_state = None
    state = FlightState.PRELAUNCH
    start_time = ut()
    last_stage_time = 0
    last_render_time = -render_time
    last_count_time = 0
    count = 3

    vessel.control.sas = False
    vessel.auto_pilot.engage()

    try:
        while state != FlightState.ORBITING:
            curr_ut = ut()
            curr_t = curr_ut - start_time
            pos_vector = np.array(position())
            vel_vector = np.array(velocity())

            # Сбор телеметрии
            if curr_t >= last_render_time + render_time:
                data["time"].append(curr_t)
                data["altitude"].append(altitude())
                data["mass"].append(mass())
                data["speed"].append(speed())
                data["pos_x"].append(pos_vector[0])
                data["pos_y"].append(pos_vector[1])
                data["pos_z"].append(pos_vector[2])
                data["vel_x"].append(vel_vector[0])
                data["vel_y"].append(vel_vector[1])
                data["vel_z"].append(vel_vector[2])

                # Силы
                f_grav = (-pos_vector / np.linalg.norm(pos_vector)) * flight.g_force

                # Направление тяги совпадает с направлением корабля
                direction = conn.space_center.transform_direction((0, 0, 1), vessel.reference_frame, reference_frame)
                f_thrust = np.array(direction) * vessel.thrust

                # Суммарная аэродинамическая сила (drag + lift)
                f_aero = vessel.flight(reference_frame).aerodynamic_force

                data["grav_x"].append(f_grav[0])
                data["grav_y"].append(f_grav[1])
                data["grav_z"].append(f_grav[2])

                data["thrust_x"].append(f_thrust[0])
                data["thrust_y"].append(f_thrust[1])
                data["thrust_z"].append(f_thrust[2])
                data["drag_x"].append(f_aero[0])
                data["drag_y"].append(f_aero[1])
                data["drag_z"].append(f_aero[2])

                data["pitch"].append(flight.pitch)
                data["heading"].append(flight.heading)
                data["roll"].append(flight.roll)
                data["apoapsis"].append(apoapsis())
                data["periapsis"].append(periapsis())
                data["horizontal_speed"].append(flight.horizontal_speed)
                data["vertical_speed"].append(flight.vertical_speed)

                # Углы
                data["vessel_angle"].append(get_angle(reference_frame, vessel))
                data["mun_angle"].append(get_angle(reference_frame, mun))
                data["phase_angle"].append(get_phase_angle(reference_frame, vessel, mun))

                last_render_time = curr_t

            # Условие сброса
            if vessel.thrust < 0.01 and state in [FlightState.LAUNCH, FlightState.GRAVITY_TURN,
                                                  FlightState.CIRCULARIZATION]:
                if curr_ut - last_stage_time > 1.0:
                    vessel.control.activate_next_stage()
                    print(f"[{int(curr_t)}с] Ступень пуста. Разделение.")
                    last_stage_time = curr_ut

            # Конечный автомат
            if state != prev_state:
                prev_state = state
                state_str = f" [{int(curr_t)}с] СОСТОЯНИЕ: {state.name} "
                print(f"\n{state_str:-^60}\n")

            if state == FlightState.PRELAUNCH:
                if count > 0:
                    if curr_t > last_count_time + 1:
                        print(f"{count}...")
                        count -= 1
                        last_count_time = curr_t
                else:
                    print("Пуск!")
                    vessel.control.throttle = 1.0
                    vessel.control.activate_next_stage()
                    vessel.auto_pilot.target_pitch_and_heading(90, 90)
                    state = FlightState.LAUNCH

            elif state == FlightState.LAUNCH:
                if altitude() > GRAV_TURN_START:
                    state = FlightState.GRAVITY_TURN

            elif state == FlightState.GRAVITY_TURN:
                target_p = calculate_pitch(altitude(), GRAV_TURN_START, GRAV_TURN_CEIL, 0.4)
                vessel.auto_pilot.target_pitch_and_heading(target_p, 90)

                # Ускорение времени во время гравитационного маневра
                if altitude() < 70000:
                    if conn.space_center.physics_warp_factor < 3:
                        conn.space_center.physics_warp_factor = 3
                else:
                    conn.space_center.physics_warp_factor = 0

                if apoapsis() >= KERBIN_ORBIT_R:
                    vessel.control.throttle = 0.0
                    conn.space_center.physics_warp_factor = 0  # Сброс при достижении цели
                    print(f"Целевой апоцентр достигнут.")
                    state = FlightState.CIRCULARIZATION_WAITING

            elif state == FlightState.CIRCULARIZATION_WAITING:
                vessel.auto_pilot.target_pitch_and_heading(0, 90)

                # Ускорение времени (Warp) в космосе
                if altitude() > 70000:
                    if time_to_ap() > 60:
                        conn.space_center.rails_warp_factor = 3
                    elif time_to_ap() > 20:
                        conn.space_center.rails_warp_factor = 2
                    else:
                        conn.space_center.rails_warp_factor = 0

                # Порог включения двигателей: 10 секунд до АП
                if altitude() > 70000 and time_to_ap() < 10:
                    conn.space_center.rails_warp_factor = 0
                    print("Начало финального импульса.")
                    state = FlightState.CIRCULARIZATION

            elif state == FlightState.CIRCULARIZATION:
                vessel.auto_pilot.target_pitch_and_heading(0, 90)

                # Проверка: если мы пролетели апоцентр (time_to_ap стал очень большим)
                # и орбита еще не готова, возвращаемся в ожидание следующего витка
                if time_to_ap() > 100 and periapsis() < 70000:
                    vessel.control.throttle = 0.0
                    print("Апоцентр пройден. Ожидание следующего окна...")
                    state = FlightState.CIRCULARIZATION_WAITING
                    continue

                diff = KERBIN_ORBIT_R - periapsis()

                # УСИЛЕННАЯ ЗАЩИТА: расчет динамической тяги
                if periapsis() >= KERBIN_ORBIT_R * 0.9999 or diff < 10:
                    vessel.control.throttle = 0.0
                    state = FlightState.ORBITING
                elif diff < 500:
                    # Финальное "подталкивание" на минимально возможной тяге
                    vessel.control.throttle = 0.005
                elif diff < 2000:
                    # Плавное торможение (тяга от 5% до 1%)
                    vessel.control.throttle = max(0.01, diff / 40000)
                elif diff < 10000:
                    # Снижение тяги до 20% при приближении
                    vessel.control.throttle = 0.2
                else:
                    vessel.control.throttle = 1.0

    except KeyboardInterrupt as _:
        print(f"Автопилот остановлен преждевременно.\n")
    except Exception as e:
        print(f"i don't care but {e}")
    finally:
        # Гарантируем сброс ускорения при выходе
        conn.space_center.physics_warp_factor = 0
        conn.space_center.rails_warp_factor = 0

    print("Сбор данных завершён.\n")
    try:
        vessel.auto_pilot.disengage()
    except Exception as e:
        print(f"i don't care but {e}")

    normalize_data(data)
    return pd.DataFrame(data)


if __name__ == "__main__":
    print(f"{" АВТОПИЛОТ КОРАБЛЯ ":#^60}x")
    conn = krpc.connect(name="Apollo 11 launch autopilot")
    data = run_autopilot(conn.space_center.active_vessel, conn, 0.5)
    print(f"{" ПОЛЁТ ЗАВЕРШЁН ":#^60}")

    data_to_csv(data, "ksp_data_orbit.csv")
    print("\nKSP данные о выводе на орбиту экспортированы в ksp_data_orbit.csv")
    print("Построение визуализации...")
    show_orbit_plot(data)