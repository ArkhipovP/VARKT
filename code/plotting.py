import pandas as pd
import matplotlib.pyplot as plt
import os


def create_plots():
    # 1. Определяем пути: на один уровень выше текущего скрипта
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "..", "data")
    plots_dir = os.path.join(base_dir, "..", "plots")

    # Создаем папку plots, если её нет
    os.makedirs(plots_dir, exist_ok=True)

    # Пути к файлам
    ksp_path = os.path.join(data_dir, "ksp_data_orbit.csv")
    sim_path = os.path.join(data_dir, "simulation_data_orbit.csv")

    # 2. Загрузка данных
    try:
        ksp = pd.read_csv(ksp_path)
        sim = pd.read_csv(sim_path)
        print("Данные успешно загружены.")
    except FileNotFoundError as e:
        print(f"Ошибка: Файлы данных не найдены по пути {data_dir}!")
        return

    # Настройка стиля (используем стандартный, если seaborn не установлен)
    plt.style.use('ggplot')
    fig_size = (12, 7)

    # --- ГРАФИК 1: Высота и Апоцентр ---
    plt.figure(figsize=fig_size)
    plt.plot(ksp['time'], ksp['altitude'] / 1000, 'b-', label='KSP Высота', linewidth=2)
    plt.plot(sim['time'], sim['altitude'] / 1000, 'b--', alpha=0.6, label='SIM Высота')
    plt.plot(ksp['time'], ksp['apoapsis'] / 1000, 'r-', label='KSP Апоцентр')
    plt.axhline(y=70, color='gray', linestyle=':', label='Граница атмосферы (70км)')
    plt.title("Сравнение профиля высоты")
    plt.xlabel("Время (с)")
    plt.ylabel("Высота (км)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "altitude_comparison.png"))

    # --- ГРАФИК 2: Проекции скорости ---
    plt.figure(figsize=fig_size)
    plt.plot(ksp['time'], ksp['velocity_x'], 'r-', label='KSP Vx (Горизонтальная)')
    plt.plot(ksp['time'], ksp['velocity_y'], 'b-', label='KSP Vy (Вертикальная)')
    plt.plot(sim['time'], sim['velocity_x'], 'r--', alpha=0.5, label='SIM Vx')
    plt.plot(sim['time'], sim['velocity_y'], 'b--', alpha=0.5, label='SIM Vy')
    plt.title("Анализ векторов скорости")
    plt.xlabel("Время (с)")
    plt.ylabel("Скорость (м/с)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "velocity_projections.png"))

    # --- ГРАФИК 3: Масса (Ступени) ---
    plt.figure(figsize=fig_size)
    plt.plot(ksp['time'], ksp['mass'], 'k-', label='KSP Масса', linewidth=2)
    plt.plot(sim['time'], sim['mass'], 'r--', label='SIM Масса')
    plt.title("Динамика изменения массы (Сброс ступеней)")
    plt.xlabel("Время (с)")
    plt.ylabel("Масса (кг)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "mass_dynamics.png"))

    # --- ГРАФИК 4: Траектория (Орбитальный вид) ---
    plt.figure(figsize=(10, 10))
    # Рисуем планету Кербин
    planet = plt.Circle((0, 0), 600000, color='#1f77b4', alpha=0.3, label='Кербин (R=600км)')
    plt.gca().add_artist(planet)

    plt.plot(ksp['position_x'], ksp['position_y'], 'r-', label='KSP Траектория')
    plt.plot(sim['position_x'], sim['position_y'], 'g--', label='SIM Траектория')

    limit = 900000  # Ограничение осей для наглядности
    plt.xlim(-limit, limit)
    plt.ylim(-limit, limit)
    plt.gca().set_aspect('equal')
    plt.title("2D Траектория вывода на орбиту")
    plt.xlabel("X координата (м)")
    plt.ylabel("Y координата (м)")
    plt.legend()
    plt.savefig(os.path.join(plots_dir, "trajectory_2d.png"))

    plt.close('all')
    print(f"Готово! Все графики сохранены в: {os.path.abspath(plots_dir)}")


if __name__ == "__main__":
    create_plots()
