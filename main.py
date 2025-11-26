import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
import sys

# ==================================================================================
# КЛАСС 1: МАТЕМАТИЧЕСКАЯ МОДЕЛЬ (SugarBeetDSS)
# Отвечает за генерацию данных и логику стратегий
# ==================================================================================

class SugarBeetDSS:
    def __init__(self):
        # Параметры по умолчанию
        self.n = 15          # Количество партий/дней
        self.nu = 7          # День переключения стратегии (для TG/GT)
        
        # Флаги модели
        self.use_ripening = False     
        self.use_chemistry = True     
        self.distribution_type = 'uniform' 
        
        # Хранение данных текущей сессии
        self.matrix_s = None        # Матрица выхода сахара (S)
        self.matrix_c = None        # Матрица содержания сахара (C)
        self.matrix_beta_avg = None # Средний коэф. деградации партии (для CTG)
        
        # === ДИАПАЗОНЫ (из методички, п. 2, 4, 6) ===
        self.range_a = (12.0, 22.0)           # Начальная сахаристость (%)
        self.range_beta_wither = (0.85, 0.99) # Увядание (<1)
        self.range_beta_ripen = (1.00, 1.05)  # Дозаривание (>1)
        
        # Химия
        self.range_K = (4.8, 7.05)
        self.range_Na = (0.21, 0.82)
        self.range_N = (1.58, 2.80)
        self.range_I0 = (0.62, 0.64) 

    def configure_manually(self):
        """Ручная настройка параметров (для одиночного режима)"""
        print("\n--- НАСТРОЙКА ПАРАМЕТРОВ ---")
        try:
            val_n = input(f"Количество этапов (n) [15]: ")
            self.n = int(val_n) if val_n else 15
            
            rip = input("Учитывать дозаривание? (y/n) [n]: ").lower()
            self.use_ripening = (rip == 'y')
            
            # Если есть дозаривание, nu - это конец дозаривания. Иначе - просто середина.
            def_nu = int(self.n // 2.1) if self.use_ripening else int(self.n / 2)
            val_nu = input(f"День переключения/дозаривания (nu) [{def_nu}]: ")
            self.nu = int(val_nu) if val_nu else def_nu
                
            chem = input("Учитывать потери от химии (K, Na, N)? (y/n) [y]: ").lower()
            self.use_chemistry = (chem != 'n')
            
            dist = input("Распределение деградации: (1) Равномерное, (2) Концентрированное [1]: ")
            self.distribution_type = 'concentrated' if dist == '2' else 'uniform'
            
        except ValueError:
            print("Ошибка ввода! Использованы значения по умолчанию.")

    def set_experiment_defaults(self):
        """Жесткие настройки для массового эксперимента (согласно методичке)"""
        self.n = 15 
        self.nu = 7
        self.use_ripening = True        # В эксперименте обычно учитывают всё
        self.use_chemistry = True
        self.distribution_type = 'concentrated' # Наиболее интересный случай

    def _get_beta(self, stage_idx, row_idx, row_centers):
        """Генерация коэффициента b_ij согласно п. 8.2"""
        limit = self.nu if self.use_ripening else 0
        
        if stage_idx < limit:
            bounds = self.range_beta_ripen
        else:
            bounds = self.range_beta_wither
        
        low, high = bounds

        if self.distribution_type == 'uniform':
            return np.random.uniform(low, high)
        
        elif self.distribution_type == 'concentrated':
            center = row_centers[row_idx]
            # Если центр деградации выпадает из границ (смена режима), берем новый случайный
            if not (low <= center <= high):
                return np.random.uniform(low, high)
            
            # Формула из текста: delta <= |beta1 - beta2| / 4
            delta = abs(high - low) / 4.0 
            return np.random.uniform(max(low, center - delta), min(high, center + delta))
        
        return 1.0

    def generate_matrix(self):
        """Полный цикл генерации данных (C, L, S)"""
        # Инициализация
        C = np.zeros((self.n, self.n))
        S = np.zeros((self.n, self.n))
        
        # 1. Генерация свойств партий (п. 5)
        a = np.random.uniform(*self.range_a, self.n)
        K = np.random.uniform(*self.range_K, self.n)
        Na = np.random.uniform(*self.range_Na, self.n)
        N = np.random.uniform(*self.range_N, self.n)
        I0 = np.random.uniform(*self.range_I0, self.n)
        
        # Центры для концентрированного распределения
        row_centers = np.random.uniform(self.range_beta_wither[0], self.range_beta_wither[1], self.n)
        self.matrix_beta_avg = row_centers 

        # 2. Поэтапный расчет
        for j in range(self.n): # Этапы
            for i in range(self.n): # Партии
                # Сахаристость C
                if j == 0:
                    C[i, j] = a[i]
                else:
                    beta = self._get_beta(j, i, row_centers)
                    C[i, j] = C[i, j-1] * beta
                
                # Потери L (п. 9)
                loss_val = 0
                if self.use_chemistry:
                    I_curr = I0[i] * (1.029 ** j)
                    M_Cx = 0.1541*(K[i]+Na[i]) + 0.2159*N[i] + 0.9989*I_curr + 0.1967
                    loss_val = M_Cx + 1.1 # +1.1% заводские потери
                
                # Итоговый выход S
                S[i, j] = max(0, C[i, j] - loss_val)

        self.matrix_s = S
        self.matrix_c = C

    def solve_hungarian(self):
        """Находит Абсолютный максимум (S*) через Венгерский алгоритм"""
        if self.matrix_s is None: return 0, []
        # scipy ищет минимум, поэтому берем отрицательную матрицу
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        
        # Восстанавливаем расписание: (день, партия)
        schedule = sorted(zip(col_ind, row_ind), key=lambda x: x[0])
        daily_yields = [self.matrix_s[batch, day] for day, batch in schedule]
        return total, daily_yields

    # --- ЛОГИКА СТРАТЕГИЙ ---

    def _run_strategy_logic(self, strategy_func):
        """Универсальный 'движок' для запуска любой стратегии"""
        available = set(range(self.n))
        daily_yields = []
        total_s = 0
        
        for day in range(self.n):
            # Спрашиваем у стратегии: какую партию взять?
            batch = strategy_func(day, available)
            
            val = self.matrix_s[batch, day]
            total_s += val
            daily_yields.append(val)
            available.remove(batch)
            
        return total_s, daily_yields

    # 1. Жадная (Greedy)
    def logic_greedy(self, day, available):
        # Максимум сахара СЕЙЧАС
        return max(available, key=lambda i: self.matrix_s[i, day])

    # 2. Бережливая (Thrifty)
    def logic_thrifty(self, day, available):
        # Минимум сахара СЕЙЧАС
        return min(available, key=lambda i: self.matrix_s[i, day])

    # 3. Бережливая -> Жадная (TG)
    def logic_tg(self, day, available):
        # До этапа nu (индекс nu-1) бережем, потом жадничаем
        if day < (self.nu - 1):
            return self.logic_thrifty(day, available)
        return self.logic_greedy(day, available)

    # 4. Жадная -> Бережливая (GT)
    def logic_gt(self, day, available):
        if day < (self.nu - 1):
            return self.logic_greedy(day, available)
        return self.logic_thrifty(day, available)

    # 5. CTG (Сортировка по деградации)
    def logic_ctg(self, day, available):
        # Выбираем ту партию, которая портится быстрее всего (min beta)
        return min(available, key=lambda i: self.matrix_beta_avg[i])
        
    # 6. АВТОРСКАЯ СТРАТЕГИЯ: "SaveLoss" (Спасение от потерь)
    def logic_saveloss(self, day, available):
        """
        Выбираем партию, которая к завтрашнему дню потеряет больше всего сахара.
        Loss = S_current - S_next
        Максимизируем Loss.
        """
        best_batch = -1
        max_predicted_loss = -99999.0
        
        for i in available:
            curr_s = self.matrix_s[i, day]
            beta = self.matrix_beta_avg[i] # Средний темп этой партии
            
            # Прогноз на завтра
            next_s = curr_s * beta
            loss = curr_s - next_s
            
            if loss > max_predicted_loss:
                max_predicted_loss = loss
                best_batch = i
        
        return best_batch


# ==================================================================================
# КЛАСС 2: УПРАВЛЕНИЕ ЭКСПЕРИМЕНТОМ (ExperimentManager)
# Отвечает за циклы, сбор статистики и графики
# ==================================================================================

class ExperimentManager:
    def __init__(self):
        self.dss = SugarBeetDSS()
        self.num_runs = 50  # По условию задачи (п. 1)
        
        # Хранилище статистики
        self.stats = {} 
        
        # Список стратегий для проверки
        self.strategies = {
            'Greedy': self.dss.logic_greedy,
            'Thrifty': self.dss.logic_thrifty,
            'Thrifty->Greedy': self.dss.logic_tg,
            'Greedy->Thrifty': self.dss.logic_gt,
            'CTG (BetaSort)': self.dss.logic_ctg,
            'SaveLoss (Custom)': self.dss.logic_saveloss
        }

    # --- РЕЖИМ 1: ОДИНОЧНЫЙ ЗАПУСК ---
    def run_single_detailed(self):
        print("\n=== ОДИНОЧНЫЙ АНАЛИЗ (Детально) ===")
        self.dss.generate_matrix()
        
        # 1. Идеал
        ideal_total, _ = self.dss.solve_hungarian()
        
        print(f"\nСгенерирована матрица {self.dss.n}x{self.dss.n}.")
        print(f"{'СТРАТЕГИЯ':<25} | {'ИТОГ (S)':<15} | {'ПОТЕРИ (%)':<10}")
        print("-" * 55)
        
        print(f"{'Ideal (S*)':<25} | {ideal_total:<15.2f} | 0.00%")
        
        # 2. Прогон всех стратегий на этой матрице
        results = []
        for name, func in self.strategies.items():
            total, _ = self.dss._run_strategy_logic(func)
            loss_pct = (1 - total / ideal_total) * 100 if ideal_total > 0 else 0
            results.append((name, total, loss_pct))
            
        # Сортировка по потерям
        results.sort(key=lambda x: x[2])
        
        for name, total, loss in results:
            print(f"{name:<25} | {total:<15.2f} | {loss:<10.2f}%")
            
        print("-" * 55)
        print("Совет: Используйте 'Массовый эксперимент' для надежных выводов.")

    # --- РЕЖИМ 2: МАССОВЫЙ ЭКСПЕРИМЕНТ ---
    def run_full_experiment(self):
        print(f"\n=== ЗАПУСК МАССОВОГО ЭКСПЕРИМЕНТА ({self.num_runs} прогонов) ===")
        print(f"Параметры: n={self.dss.n}, nu={self.dss.nu}, Химия={self.dss.use_chemistry}, Распр={self.dss.distribution_type}")
        
        # Очистка статистики
        self.stats = {}
        for name in self.strategies:
            self.stats[name] = {'totals': [], 'dynamics_sum': np.zeros(self.dss.n)}
        self.stats['Ideal'] = {'totals': [], 'dynamics_sum': np.zeros(self.dss.n)}

        # Главный цикл
        for r in range(self.num_runs):
            if (r+1) % 10 == 0: print(f"-> Выполняется прогон {r+1} из {self.num_runs}...")
            
            # А. Генерация новой уникальной ситуации
            self.dss.generate_matrix()
            
            # Б. Расчет Идеала
            ideal_sum, ideal_dyn = self.dss.solve_hungarian()
            self.stats['Ideal']['totals'].append(ideal_sum)
            self.stats['Ideal']['dynamics_sum'] += np.array(ideal_dyn)
            
            # В. Расчет всех стратегий
            for name, func in self.strategies.items():
                total, dyn = self.dss._run_strategy_logic(func)
                self.stats[name]['totals'].append(total)
                self.stats[name]['dynamics_sum'] += np.array(dyn)

        print("\nЭксперимент завершен. Анализ данных...\n")
        self.print_analysis_table()
        self.plot_results()
        self.give_recommendation()

    def print_analysis_table(self):
        """Вывод усредненных результатов (п. 11)"""
        print(f"{'СТРАТЕГИЯ':<25} | {'СР. ВЫХОД':<10} | {'ПОТЕРИ (%)':<10}")
        print("-" * 51)
        
        # Считаем среднее для идеала
        avg_ideal = np.mean(self.stats['Ideal']['totals'])
        print(f"{'Ideal (S*)':<25} | {avg_ideal:<10.2f} | 0.00%")
        print("-" * 51)
        
        # Считаем среднее для остальных
        results = []
        for name in self.strategies:
            avg_val = np.mean(self.stats[name]['totals'])
            loss_pct = (1 - avg_val / avg_ideal) * 100
            results.append((name, avg_val, loss_pct))
        
        # Сортируем (лучшие сверху)
        results.sort(key=lambda x: x[2])
        self.sorted_results = results # Сохраняем для рекомендаций
        
        for name, val, loss in results:
            print(f"{name:<25} | {val:<10.2f} | {loss:<10.2f}%")

    def plot_results(self):
        """Построение графика динамики (п. 12)"""
        try:
            plt.figure(figsize=(10, 6))
            days = range(1, self.dss.n + 1)
            
            # 1. График идеала
            # (Делим сумму векторов на кол-во прогонов, чтобы получить средний вектор)
            avg_dyn_ideal = self.stats['Ideal']['dynamics_sum'] / self.num_runs
            cum_ideal = np.cumsum(avg_dyn_ideal)
            plt.plot(days, cum_ideal, 'k--', linewidth=2, label='Ideal (S*)')
            
            # 2. Графики стратегий
            for name in self.strategies:
                avg_dyn = self.stats[name]['dynamics_sum'] / self.num_runs
                cum_val = np.cumsum(avg_dyn)
                plt.plot(days, cum_val, label=name)
                
            plt.title(f"Накопительный выход сахара (Усреднено по {self.num_runs} опытам)")
            plt.xlabel("День переработки")
            plt.ylabel("Суммарный выход сахара (%)")
            plt.legend()
            plt.grid(True)
            print("График построен. (Окно графика может открыться в фоне).")
            plt.show()
        except Exception as e:
            print(f"Не удалось построить график: {e}")

    def give_recommendation(self):
        """Рекомендация по итогам (п. 13)"""
        best_strat = self.sorted_results[0]
        name, val, loss = best_strat
        
        print("\n" + "="*50)
        print("РЕКОМЕНДАЦИЯ СППР")
        print("="*50)
        print(f"Лучшая стратегия: {name.upper()}")
        print(f"Ожидаемые потери: {loss:.2f}% (относительно S*)")
        print("="*50 + "\n")


# ==================================================================================
# ГЛАВНОЕ МЕНЮ
# ==================================================================================

def main():
    manager = ExperimentManager()
    
    while True:
        print("\n=== СППР: ОПТИМИЗАЦИЯ ПЕРЕРАБОТКИ СВЕКЛЫ ===")
        print("1. Одиночный прогон (Проверка одной ситуации)")
        print("2. МАССОВЫЙ ЭКСПЕРИМЕНТ (50 прогонов + График)")
        print("0. Выход")
        
        choice = input("Ваш выбор: ")
        
        if choice == '1':
            manager.dss.configure_manually()
            manager.run_single_detailed()
            
        elif choice == '2':
            manager.dss.set_experiment_defaults()
            manager.run_full_experiment()
            
        elif choice == '0':
            print("Выход.")
            sys.exit()
        else:
            print("Неверный ввод.")

if __name__ == "__main__":
    main()