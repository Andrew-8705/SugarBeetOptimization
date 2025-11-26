import customtkinter as ctk
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

# =============================================================================
# 1. МОДЕЛЬ (LOGIC)
# =============================================================================

class SugarBeetModel:
    def __init__(self):
        self.n = 15
        self.nu = 7
        self.use_ripening = True
        self.use_chemistry = True
        self.distribution_type = 'concentrated'
        
        self.matrix_s = None
        self.matrix_beta_avg = None
        
        self.ranges = {
            'a': (12.0, 22.0),
            'beta_wither': (0.85, 0.99),
            'beta_ripen': (1.00, 1.05),
            'K': (4.8, 7.05),
            'Na': (0.21, 0.82),
            'N': (1.58, 2.80),
            'I0': (0.62, 0.64)
        }

    def _get_beta(self, stage_idx, row_idx, row_centers):
        limit = self.nu if self.use_ripening else 0
        if stage_idx < limit:
            bounds = self.ranges['beta_ripen']
        else:
            bounds = self.ranges['beta_wither']
        
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
        
        r = self.ranges
        a = np.random.uniform(*r['a'], self.n)
        K = np.random.uniform(*r['K'], self.n)
        Na = np.random.uniform(*r['Na'], self.n)
        N = np.random.uniform(*r['N'], self.n)
        I0 = np.random.uniform(*r['I0'], self.n)
        
        row_centers = np.random.uniform(r['beta_wither'][0], r['beta_wither'][1], self.n)
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
        
    def run_simulation(self, runs=50):
        strategies = {
            'Greedy': self.logic_greedy,
            'Thrifty': self.logic_thrifty,
            'Thrifty->Greedy': self.logic_tg,
            'Greedy->Thrifty': self.logic_gt,
            'CTG (Beta)': self.logic_ctg
        }
        
        stats = {k: {'totals': [], 'dynamics_sum': np.zeros(self.n)} for k in strategies}
        stats['Ideal'] = {'totals': [], 'dynamics_sum': np.zeros(self.n)}
        
        for r in range(runs):
            self.generate_matrix()
            # Ideal
            id_sum, id_dyn = self.solve_hungarian()
            stats['Ideal']['totals'].append(id_sum)
            stats['Ideal']['dynamics_sum'] += np.array(id_dyn)
            # Strategies
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
# 2. UI КОМПОНЕНТЫ
# =============================================================================

class InfoCard(ctk.CTkFrame):
    """Красивая плашка с информацией (KPI)"""
    def __init__(self, master, title, value, color="#3a7ebf"):
        super().__init__(master, fg_color="#2b2b2b", corner_radius=10)
        self.grid_columnconfigure(1, weight=1)
        
        # Цветная полоска
        self.bar = ctk.CTkFrame(self, width=6, fg_color=color, corner_radius=6)
        self.bar.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(5, 10), pady=5)
        
        # Заголовок
        self.lbl_title = ctk.CTkLabel(self, text=title, font=("Arial", 12), text_color="#a0a0a0", anchor="w")
        self.lbl_title.grid(row=0, column=1, sticky="w", pady=(8, 0))
        
        # Значение
        self.lbl_val = ctk.CTkLabel(self, text=value, font=("Arial", 20, "bold"), text_color="white", anchor="w")
        self.lbl_val.grid(row=1, column=1, sticky="w", pady=(0, 8))

    def update_value(self, new_value):
        self.lbl_val.configure(text=new_value)

class ScrollableSettingsFrame(ctk.CTkScrollableFrame):
    """Боковая панель (без изменений)"""
    def __init__(self, master, model, **kwargs):
        super().__init__(master, **kwargs)
        self.entries = {}
        
        self.add_section("Общие параметры")
        self.add_input("N (Кол-во партий)", "n", str(model.n))
        self.add_input("Nu (День перекл.)", "nu", str(model.nu))
        self.add_input("Число экспериментов", "runs", "50")
        
        self.add_section("Флаги модели")
        self.sw_rip = ctk.CTkSwitch(self, text="Дозаривание")
        if model.use_ripening: self.sw_rip.select()
        self.sw_rip.pack(anchor="w", padx=10, pady=5)
        
        self.sw_chem = ctk.CTkSwitch(self, text="Учет химии")
        if model.use_chemistry: self.sw_chem.select()
        self.sw_chem.pack(anchor="w", padx=10, pady=5)
        
        self.sw_dist = ctk.CTkSwitch(self, text="Конц. распределение")
        if model.distribution_type == 'concentrated': self.sw_dist.select()
        self.sw_dist.pack(anchor="w", padx=10, pady=5)

        self.add_section("Диапазоны: Сахар и Деградация")
        self.add_range_input("Сахар (a)", "a", model.ranges['a'])
        self.add_range_input("Увядание (beta < 1)", "beta_wither", model.ranges['beta_wither'])
        self.add_range_input("Дозаривание (beta > 1)", "beta_ripen", model.ranges['beta_ripen'])

        self.add_section("Диапазоны: Химия")
        self.add_range_input("Калий (K)", "K", model.ranges['K'])
        self.add_range_input("Натрий (Na)", "Na", model.ranges['Na'])
        self.add_range_input("Азот (N)", "N", model.ranges['N'])
        
    def add_section(self, text):
        lbl = ctk.CTkLabel(self, text=text, font=("Arial", 14, "bold"), text_color="#3a7ebf")
        lbl.pack(anchor="w", padx=5, pady=(15, 5))
    
    def add_input(self, label, key, default):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f, text=label).pack(side="left")
        e = ctk.CTkEntry(f, width=80)
        e.insert(0, default)
        e.pack(side="right")
        self.entries[key] = e
        
    def add_range_input(self, label, key, default_tuple):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f, text=label).pack(side="left", anchor="n")
        e1 = ctk.CTkEntry(f, width=60)
        e1.insert(0, str(default_tuple[0]))
        e1.pack(side="right", padx=2)
        e2 = ctk.CTkEntry(f, width=60)
        e2.insert(0, str(default_tuple[1]))
        e2.pack(side="right", padx=2)
        self.entries[key] = (e1, e2)

    def get_values(self):
        vals = {'n': int(self.entries['n'].get()), 'nu': int(self.entries['nu'].get()), 'runs': int(self.entries['runs'].get())}
        vals['use_ripening'] = bool(self.sw_rip.get())
        vals['use_chemistry'] = bool(self.sw_chem.get())
        vals['distribution'] = 'concentrated' if self.sw_dist.get() else 'uniform'
        ranges = {}
        for k in ['a', 'beta_wither', 'beta_ripen', 'K', 'Na', 'N']:
            e1, e2 = self.entries[k]
            ranges[k] = (float(e1.get()), float(e2.get()))
        ranges['I0'] = (0.62, 0.64)
        vals['ranges'] = ranges
        return vals

# =============================================================================
# 3. ГЛАВНОЕ ОКНО
# =============================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class FinalApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sugar Beet Optimization DSS")
        self.geometry("1300x850")
        
        self.model = SugarBeetModel()
        
        # Сетка
        self.grid_columnconfigure(0, weight=0, minsize=320) # Настройки
        self.grid_columnconfigure(1, weight=1) # Дашборд
        self.grid_rowconfigure(0, weight=1)
        
        # --- ЛЕВАЯ ПАНЕЛЬ ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.left_frame, text="КОНФИГУРАЦИЯ", font=("Arial", 20, "bold")).pack(pady=20)
        self.settings_frame = ScrollableSettingsFrame(self.left_frame, self.model, width=280)
        self.settings_frame.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.btn_run = ctk.CTkButton(self.left_frame, text="ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ", 
                                     height=50, fg_color="green", font=("Arial", 14, "bold"),
                                     command=self.run_process)
        self.btn_run.pack(padx=20, pady=20, fill="x")

        # --- ПРАВАЯ ПАНЕЛЬ (ДАШБОРД) ---
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        
        # Сетка дашборда
        self.right_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.right_frame.grid_rowconfigure(0, weight=0) # KPI
        self.right_frame.grid_rowconfigure(1, weight=1) # Графики
        self.right_frame.grid_rowconfigure(2, weight=0) # Рекомендация
        
        # 1. Верхние карточки (KPI)
        self.card_ideal = InfoCard(self.right_frame, "Max Possible Yield (Ideal)", "---", color="#2ec4b6")
        self.card_ideal.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 10))
        
        self.card_best = InfoCard(self.right_frame, "Best Strategy", "---", color="#e76f51")
        self.card_best.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 10))
        
        self.card_loss = InfoCard(self.right_frame, "Min Loss", "--- %", color="#e9c46a")
        self.card_loss.grid(row=0, column=2, sticky="ew", padx=5, pady=(0, 10))

        # 2. Вкладки с графиками
        self.tabs = ctk.CTkTabview(self.right_frame)
        self.tabs.grid(row=1, column=0, columnspan=3, sticky="nsew", padx=5, pady=5)
        self.tabs.add("Динамика накопления")
        self.tabs.add("Итоговое сравнение")
        
        # Фреймы внутри вкладок
        self.frame_line_chart = ctk.CTkFrame(self.tabs.tab("Динамика накопления"), fg_color="transparent")
        self.frame_line_chart.pack(expand=True, fill="both")
        
        self.frame_bar_chart = ctk.CTkFrame(self.tabs.tab("Итоговое сравнение"), fg_color="transparent")
        self.frame_bar_chart.pack(expand=True, fill="both")
        
        self.canvas_line = None
        self.canvas_bar = None

        # 3. Блок Рекомендации
        self.rec_frame = ctk.CTkFrame(self.right_frame, fg_color="#2b2b2b", corner_radius=10, border_width=1, border_color="#555")
        self.rec_frame.grid(row=2, column=0, columnspan=3, sticky="ew", padx=5, pady=(15, 0))
        
        ctk.CTkLabel(self.rec_frame, text="РЕКОМЕНДАЦИЯ СППР", font=("Arial", 14, "bold"), text_color="#3a7ebf").pack(anchor="w", padx=20, pady=(10, 0))
        self.lbl_rec_text = ctk.CTkLabel(self.rec_frame, text="Запустите моделирование, чтобы получить совет...", font=("Consolas", 14), justify="left", wraplength=800)
        self.lbl_rec_text.pack(anchor="w", padx=20, pady=(5, 15))

    def run_process(self):
        try:
            p = self.settings_frame.get_values()
            self.model.n = p['n']
            self.model.nu = p['nu']
            self.model.use_ripening = p['use_ripening']
            self.model.use_chemistry = p['use_chemistry']
            self.model.distribution_type = p['distribution']
            self.model.ranges = p['ranges']
            
            self.btn_run.configure(text="Вычисление...", state="disabled")
            self.update()
            
            stats = self.model.run_simulation(runs=p['runs'])
            
            # Обработка результатов
            avg_ideal = np.mean(stats['Ideal']['totals'])
            
            results = []
            for name in stats:
                if name == 'Ideal': continue
                val = np.mean(stats[name]['totals'])
                loss = (1 - val/avg_ideal) * 100
                results.append((name, val, loss))
            results.sort(key=lambda x: x[2]) # Сортировка по минимальным потерям
            
            best_strat = results[0] # (name, val, loss)

            # Обновление UI
            self.card_ideal.update_value(f"{avg_ideal:.2f}")
            self.card_best.update_value(best_strat[0])
            self.card_loss.update_value(f"{best_strat[2]:.2f}%")
            
            self.update_recommendation(best_strat[0], best_strat[2])
            self.draw_line_chart(stats, p['runs'])
            self.draw_bar_chart(results, avg_ideal)

        except Exception as e:
            print(e)
        finally:
            self.btn_run.configure(text="ЗАПУСТИТЬ МОДЕЛИРОВАНИЕ", state="normal")

    def update_recommendation(self, best_name, loss):
        text = f"Лучшая стратегия показала потери всего {loss:.2f}% относительно идеала.\n"
        if "Thrifty->Greedy" in best_name:
            text += f"СОВЕТ: В начале сезона (первые {self.model.nu} дней) используйте 'Бережливую' тактику (перерабатывайте худшее сырье).\n"
            text += "Это позволит качественной свекле дозреть. Затем резко переходите на 'Жадную' тактику."
        elif "Greedy" in best_name:
            text += "СОВЕТ: Не ждите! Перерабатывайте партии с самым высоким содержанием сахара сразу.\n"
            text += "В текущих условиях (высокая деградация или отсутствие дозаривания) ожидание приведет к убыткам."
        elif "CTG" in best_name:
            text += "СОВЕТ: В первую очередь отправляйте на завод партии с низкой лежкостью (быстро гниющие).\n"
            text += "Стабильные партии оставьте на конец сезона."
        else:
            text += f"СОВЕТ: Следуйте стратегии {best_name}."
            
        self.lbl_rec_text.configure(text=text)

    def draw_line_chart(self, stats, runs):
        if self.canvas_line: self.canvas_line.get_tk_widget().pack_forget()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        days = range(1, self.model.n + 1)
        
        # Ideal
        cum_ideal = np.cumsum(stats['Ideal']['dynamics_sum'] / runs)
        ax.plot(days, cum_ideal, 'w--', linewidth=2, label='Ideal', alpha=0.6)
        
        colors = ['#e76f51', '#2a9d8f', '#e9c46a', '#f4a261', '#81b29a']
        idx = 0
        for name in stats:
            if name == 'Ideal': continue
            cum_val = np.cumsum(stats[name]['dynamics_sum'] / runs)
            ax.plot(days, cum_val, label=name, color=colors[idx % len(colors)], linewidth=2)
            idx += 1
            
        ax.set_title("Кумулятивная динамика выхода", color='white', pad=10)
        ax.grid(True, linestyle='--', alpha=0.3)
        ax.legend(facecolor='#2b2b2b', labelcolor='white', loc='upper left')
        ax.tick_params(colors='white')
        for s in ax.spines.values(): s.set_color('white')
        
        self.canvas_line = FigureCanvasTkAgg(fig, master=self.frame_line_chart)
        self.canvas_line.draw()
        self.canvas_line.get_tk_widget().pack(expand=True, fill="both")

    def draw_bar_chart(self, results, max_val):
        if self.canvas_bar: self.canvas_bar.get_tk_widget().pack_forget()
        
        fig = Figure(figsize=(6, 4), dpi=100)
        fig.patch.set_facecolor('#2b2b2b')
        ax = fig.add_subplot(111)
        ax.set_facecolor('#2b2b2b')
        
        names = ["Max"] + [r[0] for r in results]
        vals = [max_val] + [r[1] for r in results]
        
        colors = ['#2ec4b6'] + ['#e76f51' if i==0 else '#264653' for i in range(len(results))]
        
        bars = ax.bar(names, vals, color=colors, alpha=0.9)
        
        ax.set_title("Сравнение итогового выхода", color='white', pad=10)
        ax.tick_params(colors='white', axis='x', labelsize=9)
        ax.tick_params(colors='white', axis='y')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        for s in ax.spines.values(): s.set_color('white')
        
        # Подписи значений
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', color='white', fontsize=9)
        
        self.canvas_bar = FigureCanvasTkAgg(fig, master=self.frame_bar_chart)
        self.canvas_bar.draw()
        self.canvas_bar.get_tk_widget().pack(expand=True, fill="both")

if __name__ == "__main__":
    app = FinalApp()
    app.mainloop()