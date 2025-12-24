import krpc
import math
import pandas as pd
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
    # Константы
    kerbin = conn.space_center.bodies["Kerbin"]
    mun = conn.space_center.bodies["Mun"]
    reference_frame = kerbin.reference_frame
    flight = vessel.flight(reference_frame)

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

    # Инициализация словаря для данных
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
        # Основной цикл управления и сбора данных
        # Работаем, пока не выйдем на круговую орбиту
        while state != FlightState.ORBITING:
            curr_t = ut() - start_time
            pos = position()
            vel = velocity()

            # Телеметрия
            if curr_t >= last_render_time + render_time:
                data["time"].append(curr_t)
                data["altitude"].append(altitude())
                data["mass"].append(mass())
                data["speed"].append(speed())

                pos_vector = np.array(pos)
                data["pos_x"].append(pos_vector[0])
                data["pos_y"].append(pos_vector[1])
                data["pos_z"].append(pos_vector[2])
                data["pitch"].append(flight.pitch)
                data["heading"].append(flight.heading)
                data["roll"].append(flight.roll)

                vel_vector = np.array(vel)
                data["vel_x"].append(vel_vector[0])
                data["vel_y"].append(vel_vector[1])
                data["vel_z"].append(vel_vector[2])
                data["horizontal_speed"].append(flight.horizontal_speed)
                data["vertical_speed"].append(flight.vertical_speed)

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

                # Орбитальные параметры
                data["apoapsis"].append(apoapsis())
                data["periapsis"].append(periapsis())

                # Углы
                data["vessel_angle"].append(get_angle(reference_frame, vessel))
                data["mun_angle"].append(get_angle(reference_frame, mun))
                data["phase_angle"].append(get_phase_angle(reference_frame, vessel, mun))

                last_render_time = curr_t

            # Условие сброса
            if altitude() > GRAV_TURN_CEIL and vessel.control.throttle > 0.01 and ut() - last_stage_time > 1.5:
                vessel.control.activate_next_stage()
                print(f"[{int(curr_t)}с] Сброс ступени S{vessel.control.current_stage}. Активация следующей ступени...")
                last_stage_time = ut()

            if state != prev_state:
                prev_state = state
                state_str = f" [{int(curr_t)}с] СОСТОЯНИЕ: {state.name} "
                print(f"\n{state_str:-^60}\n")

            if state == FlightState.PRELAUNCH:
                if count:
                    if curr_t > last_count_time + 1:
                        last_count_time = curr_t
                        count -= 1
                        print(f"{count + 1}...")
                else:
                    print("Пуск!")
                    vessel.control.throttle = 1.0
                    vessel.control.activate_next_stage()
                    vessel.auto_pilot.target_pitch_and_heading(90, 90)
                    last_stage_time = ut()
                    state = FlightState.LAUNCH

            elif state == FlightState.LAUNCH:
                vessel.auto_pilot.target_pitch_and_heading(90, 90)
                if altitude() > GRAV_TURN_START:
                    state = FlightState.GRAVITY_TURN

            elif state == FlightState.GRAVITY_TURN:
                target_pitch = calculate_pitch(altitude(), GRAV_TURN_START, GRAV_TURN_CEIL, 0.5)
                vessel.auto_pilot.target_pitch_and_heading(target_pitch, 90)

                if apoapsis() >= KERBIN_ORBIT_R:
                    vessel.control.throttle = 0.0
                    print(f"Апоцентр {KERBIN_ORBIT_R}м достигнут. Инерциальный полет.")
                    if vessel.control.current_stage == 14:  # Сброс ступени с ненужным топливом
                        vessel.control.activate_next_stage()
                        print(f"[{int(curr_t)}с] Сброс ступени S{vessel.control.current_stage}. "
                              f"Активация следующей ступени...")
                        last_stage_time = ut()
                    #
                    vessel.auto_pilot.target_pitch_and_heading(90, 90)
                    state = FlightState.CIRCULARIZATION_WAITING

            elif state == FlightState.CIRCULARIZATION_WAITING:
                vessel.auto_pilot.target_pitch_and_heading(0, 90)
                if altitude() >= KERBIN.atmosphere_height and time_to_ap() < 15:
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
                else:
                    # Плавная тяга для точности
                    vessel.control.throttle = 0.05 if until_pe < 5000 else 1.0
    except KeyboardInterrupt:
        print("Автопилот остановлен преждевременно.\n")

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
