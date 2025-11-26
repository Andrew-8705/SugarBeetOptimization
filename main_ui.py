import customtkinter as ctk
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# ЧАСТЬ 1: ЛОГИКА (MODEL)
# Полностью скопирована и адаптирована из нашего предыдущего кода
# =============================================================================

class SugarBeetModel:
    def __init__(self):
        self.n = 15
        self.nu = 7
        self.use_ripening = True
        self.use_chemistry = True
        self.distribution_type = 'concentrated'
        
        # Данные
        self.matrix_s = None
        self.matrix_beta_avg = None
        
        # Диапазоны
        self.range_a = (12.0, 22.0)
        self.range_beta_wither = (0.85, 0.99)
        self.range_beta_ripen = (1.00, 1.05)
        self.range_K = (4.8, 7.05)
        self.range_Na = (0.21, 0.82)
        self.range_N = (1.58, 2.80)
        self.range_I0 = (0.62, 0.64)

    def _get_beta(self, stage_idx, row_idx, row_centers):
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
            if not (low <= center <= high):
                return np.random.uniform(low, high)
            delta = abs(high - low) / 4.0
            return np.random.uniform(max(low, center - delta), min(high, center + delta))
        return 1.0

    def generate_matrix(self):
        C = np.zeros((self.n, self.n))
        S = np.zeros((self.n, self.n))
        
        a = np.random.uniform(*self.range_a, self.n)
        K = np.random.uniform(*self.range_K, self.n)
        Na = np.random.uniform(*self.range_Na, self.n)
        N = np.random.uniform(*self.range_N, self.n)
        I0 = np.random.uniform(*self.range_I0, self.n)
        
        row_centers = np.random.uniform(self.range_beta_wither[0], self.range_beta_wither[1], self.n)
        self.matrix_beta_avg = row_centers

        for j in range(self.n):
            for i in range(self.n):
                if j == 0:
                    C[i, j] = a[i]
                else:
                    beta = self._get_beta(j, i, row_centers)
                    C[i, j] = C[i, j-1] * beta
                
                loss_val = 0
                if self.use_chemistry:
                    I_curr = I0[i] * (1.029 ** j)
                    M_Cx = 0.1541*(K[i]+Na[i]) + 0.2159*N[i] + 0.9989*I_curr + 0.1967
                    loss_val = M_Cx + 1.1
                
                S[i, j] = max(0, C[i, j] - loss_val)
        self.matrix_s = S

    def solve_hungarian(self):
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        schedule = sorted(zip(col_ind, row_ind), key=lambda x: x[0])
        daily_yields = [self.matrix_s[batch, day] for day, batch in schedule]
        return total, daily_yields

    # Стратегии
    def logic_greedy(self, day, available):
        return max(available, key=lambda i: self.matrix_s[i, day])

    def logic_thrifty(self, day, available):
        return min(available, key=lambda i: self.matrix_s[i, day])

    def logic_tg(self, day, available):
        if day < (self.nu - 1): return self.logic_thrifty(day, available)
        return self.logic_greedy(day, available)

    def logic_gt(self, day, available):
        if day < (self.nu - 1): return self.logic_greedy(day, available)
        return self.logic_thrifty(day, available)

    def logic_ctg(self, day, available):
        return min(available, key=lambda i: self.matrix_beta_avg[i])
        
    def logic_saveloss(self, day, available):
        best_batch = -1
        max_loss = -99999.0
        for i in available:
            curr_s = self.matrix_s[i, day]
            beta = self.matrix_beta_avg[i]
            loss = curr_s - (curr_s * beta)
            if loss > max_loss:
                max_loss = loss
                best_batch = i
        return best_batch

    def run_simulation(self, runs=50):
        strategies = {
            'Greedy': self.logic_greedy,
            'Thrifty': self.logic_thrifty,
            'Thrifty->Greedy': self.logic_tg,
            'Greedy->Thrifty': self.logic_gt,
            'CTG (BetaSort)': self.logic_ctg,
            'SaveLoss (Custom)': self.logic_saveloss
        }
        
        stats = {k: {'totals': [], 'dynamics_sum': np.zeros(self.n)} for k in strategies}
        stats['Ideal'] = {'totals': [], 'dynamics_sum': np.zeros(self.n)}
        
        for r in range(runs):
            self.generate_matrix()
            
            # Ideal
            id_sum, id_dyn = self.solve_hungarian()
            stats['Ideal']['totals'].append(id_sum)
            stats['Ideal']['dynamics_sum'] += np.array(id_dyn)
            
            # Others
            for name, func in strategies.items():
                available = set(range(self.n))
                daily = []
                tot = 0
                for day in range(self.n):
                    b = func(day, available)
                    val = self.matrix_s[b, day]
                    tot += val
                    daily.append(val)
                    available.remove(b)
                
                stats[name]['totals'].append(tot)
                stats[name]['dynamics_sum'] += np.array(daily)
                
        return stats

# =============================================================================
# ЧАСТЬ 2: ИНТЕРФЕЙС (VIEW / GUI)
# =============================================================================

ctk.set_appearance_mode("Dark")  # Modes: "System" (standard), "Dark", "Light"
ctk.set_default_color_theme("blue")  # Themes: "blue" (standard), "green", "dark-blue"

class App(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- Настройки окна ---
        self.title("СППР: Переработка Сахарной Свеклы")
        self.geometry("1100x650")
        
        self.model = SugarBeetModel()

        # --- Сетка ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # --- ЛЕВАЯ ПАНЕЛЬ (НАСТРОЙКИ) ---
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(7, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="Настройки модели", font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        # Ввод N
        self.label_n = ctk.CTkLabel(self.sidebar_frame, text="Кол-во этапов (n):")
        self.label_n.grid(row=1, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_n = ctk.CTkEntry(self.sidebar_frame)
        self.entry_n.grid(row=2, column=0, padx=20, pady=(0, 10))
        self.entry_n.insert(0, "15")

        # Ввод Nu
        self.label_nu = ctk.CTkLabel(self.sidebar_frame, text="Переключение (nu):")
        self.label_nu.grid(row=3, column=0, padx=20, pady=(10, 0), sticky="w")
        self.entry_nu = ctk.CTkEntry(self.sidebar_frame)
        self.entry_nu.grid(row=4, column=0, padx=20, pady=(0, 10))
        self.entry_nu.insert(0, "7")

        # Свитч Дозаривание
        self.switch_ripen = ctk.CTkSwitch(self.sidebar_frame, text="Дозаривание")
        self.switch_ripen.grid(row=5, column=0, padx=20, pady=10, sticky="w")
        self.switch_ripen.select()

        # Свитч Химия
        self.switch_chem = ctk.CTkSwitch(self.sidebar_frame, text="Учет химии")
        self.switch_chem.grid(row=6, column=0, padx=20, pady=10, sticky="w")
        self.switch_chem.select()
        
        # Кнопка Запуска
        self.btn_run = ctk.CTkButton(self.sidebar_frame, text="ЗАПУСТИТЬ АНАЛИЗ", command=self.run_experiment, fg_color="green", hover_color="darkgreen")
        self.btn_run.grid(row=8, column=0, padx=20, pady=20)

        # --- ПРАВАЯ ПАНЕЛЬ (ВКЛАДКИ) ---
        self.tabview = ctk.CTkTabview(self, width=400)
        self.tabview.grid(row=0, column=1, padx=20, pady=10, sticky="nsew")
        self.tabview.add("Отчет")
        self.tabview.add("График")
        
        # Вкладка ОТЧЕТ
        self.textbox = ctk.CTkTextbox(self.tabview.tab("Отчет"), font=("Consolas", 14))
        self.textbox.pack(expand=True, fill="both", padx=10, pady=10)
        
        # Вкладка ГРАФИК
        self.plot_frame = ctk.CTkFrame(self.tabview.tab("График"))
        self.plot_frame.pack(expand=True, fill="both", padx=10, pady=10)
        self.canvas = None

    def run_experiment(self):
        # 1. Считываем настройки
        try:
            self.model.n = int(self.entry_n.get())
            self.model.nu = int(self.entry_nu.get())
            self.model.use_ripening = bool(self.switch_ripen.get())
            self.model.use_chemistry = bool(self.switch_chem.get())
            self.model.distribution_type = 'concentrated' # По умолчанию
        except ValueError:
            self.textbox.delete("0.0", "end")
            self.textbox.insert("0.0", "Ошибка: введите корректные числа для n и nu.")
            return

        self.btn_run.configure(state="disabled", text="Вычисляю...")
        self.update() # Обновить GUI

        # 2. Запускаем модель (50 прогонов)
        stats = self.model.run_simulation(runs=50)

        # 3. Формируем текст отчета
        self.textbox.delete("0.0", "end")
        report_text = f"=== РЕЗУЛЬТАТЫ ЭКСПЕРИМЕНТА ===\n"
        report_text += f"Параметры: n={self.model.n}, nu={self.model.nu}, Runs=50\n\n"
        report_text += f"{'СТРАТЕГИЯ':<25} | {'СР. ВЫХОД':<10} | {'ПОТЕРИ (%)':<10}\n"
        report_text += "-"*55 + "\n"

        avg_ideal = np.mean(stats['Ideal']['totals'])
        report_text += f"{'Ideal (S*)':<25} | {avg_ideal:<10.2f} | 0.00%\n"
        report_text += "-"*55 + "\n"

        # Сортировка
        results = []
        for name in stats:
            if name == 'Ideal': continue
            avg_val = np.mean(stats[name]['totals'])
            loss = (1 - avg_val/avg_ideal) * 100
            results.append((name, avg_val, loss))
        
        results.sort(key=lambda x: x[2])

        for name, val, loss in results:
            report_text += f"{name:<25} | {val:<10.2f} | {loss:<10.2f}%\n"
        
        # Рекомендация
        best_name = results[0][0]
        report_text += "\n" + "="*55 + "\n"
        report_text += f"РЕКОМЕНДАЦИЯ: Лучшая стратегия -> {best_name}\n"
        report_text += "="*55 + "\n"

        self.textbox.insert("0.0", report_text)

        # 4. Строим график
        self.draw_chart(stats, runs=50)

        self.btn_run.configure(state="normal", text="ЗАПУСТИТЬ АНАЛИЗ")

    def draw_chart(self, stats, runs):
        # Очистка старого графика
        if self.canvas:
            self.canvas.get_tk_widget().pack_forget()
        
        # Создание фигуры Matplotlib
        fig = Figure(figsize=(5, 4), dpi=100)
        ax = fig.add_subplot(111)
        
        days = range(1, self.model.n + 1)
        
        # Идеал
        avg_dyn_ideal = stats['Ideal']['dynamics_sum'] / runs
        cum_ideal = np.cumsum(avg_dyn_ideal)
        ax.plot(days, cum_ideal, 'k--', linewidth=2, label='Ideal', alpha=0.7)
        
        # Остальные
        for name in stats:
            if name == 'Ideal': continue
            avg_dyn = stats[name]['dynamics_sum'] / runs
            cum_val = np.cumsum(avg_dyn)
            ax.plot(days, cum_val, label=name)
        
        ax.set_title("Накопительный выход сахара (среднее)")
        ax.set_xlabel("Дни")
        ax.set_ylabel("Сумма S")
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.legend(fontsize='small')
        
        # Встраивание в Tkinter
        self.canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(expand=True, fill="both")

if __name__ == "__main__":
    app = App()
    app.mainloop()