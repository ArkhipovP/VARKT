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
        print("Данные успешно загружены для построения графиков.")
    except FileNotFoundError as e:
        print(f"Ошибка: Файлы данных не найдены в {data_dir}!")
        return

    # --- Создание комбинированного графика (Altitude, Velocity, Mass) ---
    fig, axs = plt.subplots(3, 1, figsize=(12, 15), sharex=True)
    plt.subplots_adjust(hspace=0.2)

    # 1. Высота
    axs[0].plot(ksp["time"], ksp["altitude"] / 1000, "b-", label="KSP", linewidth=2)
    axs[0].plot(sim["time"], sim["altitude"] / 1000, "r--", label="SIM", alpha=0.8)
    axs[0].axhline(y=70, color="gray", linestyle=":", alpha=0.5, label="Атмосфера (70км)")
    axs[0].set_ylabel("Высота (км)")
    axs[0].set_title("Сравнение телеметрии: KSP vs SIM")
    axs[0].legend(loc="upper left")
    axs[0].grid(True, alpha=0.3)

    # 2. Скорость (Общая)
    axs[1].plot(ksp["time"], ksp["speed"], "b-", label="KSP V", linewidth=2)
    axs[1].plot(sim["time"], sim["speed"], "r--", label="SIM V", alpha=0.8)
    axs[1].set_ylabel("Скорость (м/с)")
    axs[1].legend(loc="upper left")
    axs[1].grid(True, alpha=0.3)

    # 3. Масса
    axs[2].plot(ksp["time"], ksp["mass"], "b-", label="KSP Масса", linewidth=2)
    axs[2].plot(sim["time"], sim["mass"], "r--", label="SIM Масса", alpha=0.8)
    axs[2].set_ylabel("Масса (кг)")
    axs[2].set_xlabel("Время (с)")
    axs[2].legend(loc="upper left")
    axs[2].grid(True, alpha=0.3)

    # Сохранение комбинированного графика
    combined_path = os.path.join(plots_dir, "telemetry_combined.png")
    plt.savefig(combined_path, bbox_inches='tight')
    print(f"Комбинированный график сохранен: {combined_path}")

    # --- Траектория 2D (отдельный файл) ---
    plt.figure(figsize=(10, 10))
    planet = plt.Circle((0, 0), 600000, color="#1f77b4", alpha=0.2, label="Кербин (R=600км)")
    plt.gca().add_artist(planet)
    plt.plot(ksp["pos_x"], ksp["pos_z"], "r-", label="KSP", linewidth=1.5)
    plt.plot(sim["pos_x"], sim["pos_z"], "g--", label="SIM", alpha=0.7)

    limit = 900000
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.gca().set_aspect("equal")
    plt.title("Траектория (X-Z проекция)")
    plt.xlabel("X (м)")
    plt.ylabel("Z (м)")
    plt.legend()
    plt.grid(True, alpha=0.2)

    plt.savefig(os.path.join(plots_dir, "trajectory_2d.png"))

    plt.close("all")
    print(f"Все операции завершены. Папка: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    create_plots()