import pandas as pd
import matplotlib.pyplot as plt
import os


def create_plots():
    # Определение путей
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    plots_dir = os.path.join(base_dir, "..", "plots")

    os.makedirs(plots_dir, exist_ok=True)

    # Пути к файлам
    ksp_path = os.path.join(data_dir, "ksp_data_orbit.csv")
    sim_path = os.path.join(data_dir, "sim_data_orbit.csv")

    # Загрузка данных
    try:
        ksp = pd.read_csv(ksp_path)
        sim = pd.read_csv(sim_path)
        print("Данные успешно загружены.")
    except FileNotFoundError as e:
        print(f"Ошибка: Файлы данных не найдены по пути {data_dir}!")
        return

    fig_size = (12, 7)

    # Высота
    plt.figure(figsize=fig_size)
    plt.plot(ksp["time"], ksp["altitude"] / 1000, "b-", label="KSP Высота", linewidth=2)
    plt.plot(sim["time"], sim["altitude"] / 1000, "r--", label="SIM Высота")
    plt.axhline(y=70, color="gray", linestyle=":", alpha=0.5, label="Граница атмосферы (70км)")
    plt.axhline(y=190, color="green", linestyle="-", alpha=0.5, label="Парковочная орбита (190км)")
    plt.title("Высота")
    plt.xlabel("t (с)")
    plt.ylabel("h (км)")
    plt.ylim(0, 200)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "altitude.png"))

    # Проекции скорости
    plt.figure(figsize=fig_size)
    plt.plot(ksp["time"], ksp["vel_x"], "r-", label="KSP Vx")
    plt.plot(ksp["time"], ksp["vel_z"], "b-", label="KSP Vz")
    plt.plot(sim["time"], sim["vel_x"], "r--", alpha=0.5, label="SIM Vx")
    plt.plot(sim["time"], sim["vel_z"], "b--", alpha=0.5, label="SIM Vz")
    plt.title("Скорость")
    plt.xlabel("t (с)")
    plt.ylabel("v (м/с)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "velocity.png"))

    # Масса
    plt.figure(figsize=fig_size)
    plt.plot(ksp["time"], ksp["mass"], "k-", label="KSP Масса", linewidth=2)
    plt.plot(sim["time"], sim["mass"], "r--", label="SIM Масса")
    plt.title("Масса")
    plt.xlabel("t (с)")
    plt.ylabel("m (кг)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "mass.png"))

    # Тракетория
    plt.figure(figsize=(10, 10))
    planet = plt.Circle((0, 0), 600000, color="#1f77b4", alpha=0.3, label="Кербин (R=600км)")
    plt.gca().add_artist(planet)

    plt.plot(ksp["pos_x"], ksp["pos_z"], "r-", label="KSP Траектория")
    plt.plot(sim["pos_x"], sim["pos_z"], "g--", label="SIM Траектория")

    limit = 900000
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.gca().set_aspect("equal")
    plt.title("2D Траектория вывода на орбиту")
    plt.xlabel("X (м)")
    plt.ylabel("Z (м)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "trajectory_2d.png"))

    # fig, axs = plt.subplots(3, 1, figsize=fig_size, sharex=True)
    # fig.suptitle("Анализ действующих сил (SIM)", fontsize=16)
    #
    # # Проверка наличия колонок с силами в sim_data
    # # Если их нет в CSV, их нужно рассчитать или убедиться, что симулятор их пишет
    # time_sim = sim["time"]
    #
    # # Проекция на X (Горизонтальная)
    # if "thrust_x" in sim.columns:
    #     axs[0].plot(time_sim, sim["force_thrust_x"], "r-", label="Тяга (X)")
    #     axs[0].plot(time_sim, sim["force_drag_x"], "g--", label="Сопротивление (X)")
    #     axs[0].set_ylabel("Сила X (Н)")
    #     axs[0].legend(loc="upper right")
    #     axs[0].grid(True, alpha=0.3)
    #
    # # Проекция на Y (Вертикальная / Высота)
    # # Используем v_axis из sim_data
    # v_axis_force = "force_thrust_y" if "force_thrust_y" in sim.columns else "force_thrust_z"
    # g_axis_force = "force_grav_y" if "force_grav_y" in sim.columns else "force_grav_z"
    # d_axis_force = "force_drag_y" if "force_drag_y" in sim.columns else "force_drag_z"
    #
    # if v_axis_force in sim.columns:
    #     axs[1].plot(time_sim, sim[v_axis_force], "r-", label="Тяга (Vert)")
    #     axs[1].plot(time_sim, sim[g_axis_force], "b-", label="Гравитация")
    #     axs[1].plot(time_sim, sim[d_axis_force], "g--", label="Сопротивление (Vert)")
    #     axs[1].set_ylabel("Сила Vert (Н)")
    #     axs[1].legend(loc="upper right")
    #     axs[1].grid(True, alpha=0.3)
    #
    # # Результирующее ускорение (F_total / m)
    # if "accel_total" in sim.columns:
    #     axs[2].plot(time_sim, sim["accel_total"], "k-", label="Суммарное ускорение")
    #     axs[2].set_ylabel("Ускорение (м/с²)")
    #     axs[2].set_xlabel("Время (с)")
    #     axs[2].legend(loc="upper right")
    #     axs[2].grid(True, alpha=0.3)
    #
    # plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    # plt.savefig(os.path.join(plots_dir, "forces.png"))

    plt.close("all")
    print(f"Готово! Все графики сохранены в: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    create_plots()
