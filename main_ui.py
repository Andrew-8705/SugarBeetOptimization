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
        # Дефолтные параметры
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
            'K': (4.8, 7.05), 'Na': (0.21, 0.82), 'N': (1.58, 2.80), 'I0': (0.62, 0.64)
        }

    # --- Генерация ---
    def _get_beta(self, stage_idx, row_idx, row_centers):
        limit = self.nu if self.use_ripening else 0
        bounds = self.ranges['beta_ripen'] if stage_idx < limit else self.ranges['beta_wither']
        low, high = bounds
        
        if self.distribution_type == 'uniform':
            return np.random.uniform(low, high)
        elif self.distribution_type == 'concentrated':
            center = row_centers[row_idx]
            if not (low <= center <= high): return np.random.uniform(low, high)
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
                if j == 0: C[i, j] = a[i]
                else: C[i, j] = C[i, j-1] * self._get_beta(j, i, row_centers)
                
                loss_val = 0
                if self.use_chemistry:
                    I_curr = I0[i] * (1.029 ** j)
                    M_Cx = 0.1541*(K[i]+Na[i]) + 0.2159*N[i] + 0.9989*I_curr + 0.1967
                    loss_val = M_Cx + 1.1
                S[i, j] = max(0, C[i, j] - loss_val)
        self.matrix_s = S

    # --- Ручной ввод ---
    def set_manual_matrix(self, matrix, manual_nu):
        self.matrix_s = np.array(matrix)
        self.n = self.matrix_s.shape[0]
        self.nu = manual_nu
        # Для ручного режима CTG (BetaSort) не имеет данных о деградации,
        # поэтому заполняем заглушкой, чтобы стратегия работала (как рандом)
        self.matrix_beta_avg = np.random.uniform(0.9, 0.95, self.n)

    # --- Стратегии ---
    def solve_hungarian(self):
        # Венгерский алгоритм (Идеал)
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        schedule = sorted(zip(col_ind, row_ind), key=lambda x: x[0])
        daily_yields = [self.matrix_s[batch, day] for day, batch in schedule]
        return total, daily_yields

    def logic_greedy(self, day, available): return max(available, key=lambda i: self.matrix_s[i, day])
    def logic_thrifty(self, day, available): return min(available, key=lambda i: self.matrix_s[i, day])
    
    def logic_tg(self, day, available):
        # Используем self.nu для переключения
        return self.logic_thrifty(day, available) if day < (self.nu - 1) else self.logic_greedy(day, available)
    
    def logic_gt(self, day, available):
        return self.logic_greedy(day, available) if day < (self.nu - 1) else self.logic_thrifty(day, available)
    
    def logic_ctg(self, day, available):
        return min(available, key=lambda i: self.matrix_beta_avg[i])
        
    def run_simulation(self, runs=50, manual_mode=False):
        strategies = {
            'Greedy': self.logic_greedy,
            'Thrifty': self.logic_thrifty,
            'Thrifty->Greedy': self.logic_tg,
            'Greedy->Thrifty': self.logic_gt,
            'CTG (Beta)': self.logic_ctg
        }
        
        stats = {k: {'totals': [], 'dynamics_sum': np.zeros(self.n)} for k in strategies}
        stats['Ideal'] = {'totals': [], 'dynamics_sum': np.zeros(self.n)}
        
        effective_runs = 1 if manual_mode else runs
        
        for r in range(effective_runs):
            if not manual_mode:
                self.generate_matrix()
            
            # Расчет идеала
            id_sum, id_dyn = self.solve_hungarian()
            stats['Ideal']['totals'].append(id_sum)
            stats['Ideal']['dynamics_sum'] += np.array(id_dyn)
            
            # Расчет стратегий
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
                
        return stats, effective_runs

# =============================================================================
# 2. UI: ПАНЕЛИ НАСТРОЕК
# =============================================================================

class AutoSettingsFrame(ctk.CTkScrollableFrame):
    """Настройки для автоматической генерации"""
    def __init__(self, master, model, **kwargs):
        super().__init__(master, **kwargs)
        self.entries = {}
        
        self.add_section("1. Размерность")
        self.add_input("N (Кол-во партий)", "n", str(model.n))
        self.add_input("Число прогонов", "runs", "50")
        
        self.add_section("2. Логика модели")
        self.add_input("Nu (День перекл.)", "nu", str(model.nu))
        
        self.sw_rip = ctk.CTkSwitch(self, text="Дозаривание")
        if model.use_ripening: self.sw_rip.select()
        self.sw_rip.pack(anchor="w", padx=10, pady=5)
        
        self.sw_chem = ctk.CTkSwitch(self, text="Учет химии")
        if model.use_chemistry: self.sw_chem.select()
        self.sw_chem.pack(anchor="w", padx=10, pady=5)
        
        self.sw_dist = ctk.CTkSwitch(self, text="Конц. распределение")
        if model.distribution_type == 'concentrated': self.sw_dist.select()
        self.sw_dist.pack(anchor="w", padx=10, pady=5)

        self.add_section("3. Диапазоны параметров")
        self.add_range_input("Сахар (нач.)", "a", model.ranges['a'])
        self.add_range_input("Увядание (beta<1)", "beta_wither", model.ranges['beta_wither'])
        self.add_range_input("Дозаривание (beta>1)", "beta_ripen", model.ranges['beta_ripen'])
        self.add_range_input("Калий (K)", "K", model.ranges['K'])
        self.add_range_input("Натрий (Na)", "Na", model.ranges['Na'])
        self.add_range_input("Азот (N)", "N", model.ranges['N'])

    def add_section(self, text):
        ctk.CTkLabel(self, text=text, font=("Arial", 13, "bold"), text_color="#3a7ebf").pack(anchor="w", padx=5, pady=(15, 2))
    def add_input(self, label, key, default):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f, text=label).pack(side="left")
        e = ctk.CTkEntry(f, width=60); e.insert(0, default); e.pack(side="right")
        self.entries[key] = e
    def add_range_input(self, label, key, default_tuple):
        f = ctk.CTkFrame(self, fg_color="transparent")
        f.pack(fill="x", padx=5, pady=2)
        ctk.CTkLabel(f, text=label).pack(side="left")
        e1 = ctk.CTkEntry(f, width=45); e1.insert(0, str(default_tuple[0])); e1.pack(side="right", padx=2)
        e2 = ctk.CTkEntry(f, width=45); e2.insert(0, str(default_tuple[1])); e2.pack(side="right", padx=2)
        self.entries[key] = (e1, e2)
        
    def get_params(self):
        vals = {
            'n': int(self.entries['n'].get()),
            'nu': int(self.entries['nu'].get()),
            'runs': int(self.entries['runs'].get()),
            'use_ripening': bool(self.sw_rip.get()),
            'use_chemistry': bool(self.sw_chem.get()),
            'distribution': 'concentrated' if self.sw_dist.get() else 'uniform',
            'ranges': {}
        }
        for k in ['a', 'beta_wither', 'beta_ripen', 'K', 'Na', 'N']:
            vals['ranges'][k] = (float(self.entries[k][0].get()), float(self.entries[k][1].get()))
        vals['ranges']['I0'] = (0.62, 0.64)
        return vals

class ManualSettingsFrame(ctk.CTkFrame):
    """Настройки для ручного ввода"""
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        # 1. Поле для матрицы
        ctk.CTkLabel(self, text="Матрица выхода S (строки через Enter):", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 5))
        ctk.CTkLabel(self, text="Пример: 15.5 14.0\n        14.0 13.0", font=("Consolas", 10), text_color="gray").pack(anchor="w")
        
        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 12), height=200)
        self.textbox.pack(fill="x", pady=5)
        self.textbox.insert("0.0", "15.5 14.8 13.0\n14.0 13.5 12.0\n16.2 15.5 14.0")

        # 2. Настройка Nu (обязательна для логики стратегий)
        sep = ctk.CTkFrame(self, height=2, fg_color="gray")
        sep.pack(fill="x", pady=15)
        
        ctk.CTkLabel(self, text="Параметры обработки:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        f_nu = ctk.CTkFrame(self, fg_color="transparent")
        f_nu.pack(fill="x", pady=5)
        ctk.CTkLabel(f_nu, text="Nu (День смены стратегии):").pack(side="left")
        self.entry_nu = ctk.CTkEntry(f_nu, width=60)
        self.entry_nu.insert(0, "2")
        self.entry_nu.pack(side="right")
        
        ctk.CTkLabel(self, text="* Остальные параметры (химия, диапазоны)\nпри ручном вводе не применяются,\nт.к. вы вводите уже финальный выход.", 
                     font=("Arial", 11), text_color="#e07a5f").pack(pady=10)

    def get_data(self):
        text = self.textbox.get("0.0", "end").strip()
        if not text: return None, None
        try:
            # Парсинг матрицы
            rows = text.split('\n')
            matrix = []
            for r in rows:
                if r.strip():
                    matrix.append([float(x) for x in r.replace(',', '.').split()])
            nu = int(self.entry_nu.get())
            return matrix, nu
        except ValueError:
            return None, None

class InfoCard(ctk.CTkFrame):
    def __init__(self, master, title, value, color="#3a7ebf"):
        super().__init__(master, fg_color="#2b2b2b", corner_radius=10)
        self.grid_columnconfigure(1, weight=1)
        self.bar = ctk.CTkFrame(self, width=6, fg_color=color, corner_radius=6)
        self.bar.grid(row=0, column=0, rowspan=2, sticky="ns", padx=(5, 10), pady=5)
        self.lbl_title = ctk.CTkLabel(self, text=title, font=("Arial", 12), text_color="#a0a0a0", anchor="w")
        self.lbl_title.grid(row=0, column=1, sticky="w", pady=(8, 0))
        self.lbl_val = ctk.CTkLabel(self, text=value, font=("Arial", 20, "bold"), text_color="white", anchor="w")
        self.lbl_val.grid(row=1, column=1, sticky="w", pady=(0, 8))
    def update_value(self, new_value): self.lbl_val.configure(text=new_value)

# =============================================================================
# 3. ГЛАВНОЕ ОКНО
# =============================================================================

ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("dark-blue")

class FinalApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Sugar Beet Optimization DSS")
        self.geometry("1400x850")
        
        self.model = SugarBeetModel()
        
        self.grid_columnconfigure(0, weight=0, minsize=350)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- ЛЕВАЯ ПАНЕЛЬ ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        ctk.CTkLabel(self.left_frame, text="ИСТОЧНИК ДАННЫХ", font=("Arial", 20, "bold")).pack(pady=(20, 10))
        
        # Вкладки (Tabs) для выбора режима
        self.tab_selector = ctk.CTkTabview(self.left_frame)
        self.tab_selector.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        
        self.tab_auto = self.tab_selector.add("Авто-Генерация")
        self.tab_manual = self.tab_selector.add("Ручной Ввод")
        
        # 1. Содержимое вкладки Авто
        self.auto_config = AutoSettingsFrame(self.tab_auto, self.model)
        self.auto_config.pack(expand=True, fill="both")
        
        # 2. Содержимое вкладки Ручной
        self.manual_config = ManualSettingsFrame(self.tab_manual)
        self.manual_config.pack(expand=True, fill="both")
        
        # Кнопка Запуска
        self.btn_run = ctk.CTkButton(self.left_frame, text="ЗАПУСТИТЬ РАСЧЕТ", 
                                     height=50, fg_color="green", font=("Arial", 14, "bold"),
                                     command=self.run_process)
        self.btn_run.pack(padx=20, pady=20, fill="x")

        # --- ПРАВАЯ ПАНЕЛЬ ---
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_frame.grid_columnconfigure((0, 1, 2), weight=1)
        self.right_frame.grid_rowconfigure(1, weight=1)

        # KPI
        self.card_ideal = InfoCard(self.right_frame, "Max Possible Yield", "---", color="#2ec4b6")
        self.card_ideal.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 10))
        self.card_best = InfoCard(self.right_frame, "Best Strategy", "---", color="#e76f51")
        self.card_best.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 10))
        self.card_loss = InfoCard(self.right_frame, "Min Loss", "--- %", color="#e9c46a")
        self.card_loss.grid(row=0, column=2, sticky="ew", padx=5, pady=(0, 10))

        # Графики
        self.tabs_graph = ctk.CTkTabview(self.right_frame)
        self.tabs_graph.grid(row=1, column=0, columnspan=3, sticky="nsew")
        self.tabs_graph.add("Динамика")
        self.tabs_graph.add("Итоги")
        
        self.frame_line = ctk.CTkFrame(self.tabs_graph.tab("Динамика"), fg_color="transparent")
        self.frame_line.pack(fill="both", expand=True)
        self.frame_bar = ctk.CTkFrame(self.tabs_graph.tab("Итоги"), fg_color="transparent")
        self.frame_bar.pack(fill="both", expand=True)
        self.canvas_line = None; self.canvas_bar = None

        # Рекомендация
        self.rec_frame = ctk.CTkFrame(self.right_frame, fg_color="#2b2b2b")
        self.rec_frame.grid(row=2, column=0, columnspan=3, sticky="ew", pady=(15, 0))
        self.lbl_rec = ctk.CTkLabel(self.rec_frame, text="Выберите режим и запустите расчет...", font=("Consolas", 13), justify="left", wraplength=900)
        self.lbl_rec.pack(anchor="w", padx=20, pady=15)

    def run_process(self):
        try:
            self.btn_run.configure(text="Вычисление...", state="disabled")
            self.update()
            
            # Проверяем, какая вкладка активна
            active_tab = self.tab_selector.get()
            manual_mode = (active_tab == "Ручной Ввод")
            
            runs = 50
            
            if manual_mode:
                # Получаем данные из вкладки Manual
                matrix, manual_nu = self.manual_config.get_data()
                if matrix is None: raise ValueError("Ошибка данных матрицы")
                
                self.model.set_manual_matrix(matrix, manual_nu)
            else:
                # Получаем данные из вкладки Auto
                p = self.auto_config.get_params()
                self.model.n = p['n']
                self.model.nu = p['nu']
                self.model.use_ripening = p['use_ripening']
                self.model.use_chemistry = p['use_chemistry']
                self.model.distribution_type = p['distribution']
                self.model.ranges = p['ranges']
                runs = p['runs']

            # Запуск модели
            stats, effective_runs = self.model.run_simulation(runs=runs, manual_mode=manual_mode)
            
            # Анализ результатов
            avg_ideal = np.mean(stats['Ideal']['totals'])
            results = []
            for name in stats:
                if name == 'Ideal': continue
                val = np.mean(stats[name]['totals'])
                loss = (1 - val/avg_ideal) * 100 if avg_ideal != 0 else 0
                results.append((name, val, loss))
            results.sort(key=lambda x: x[2])
            
            best = results[0]
            self.card_ideal.update_value(f"{avg_ideal:.2f}")
            self.card_best.update_value(best[0])
            self.card_loss.update_value(f"{best[2]:.2f}%")
            
            rec_text = f"РЕЖИМ: {active_tab.upper()}\n"
            rec_text += f"Лучшая стратегия: {best[0]} (Выход {best[1]:.2f}, Потери {best[2]:.2f}%).\n"
            if manual_mode: rec_text += "В ручном режиме диапазоны сахара и химии игнорируются, используется введенная матрица."
            
            self.lbl_rec.configure(text=rec_text)
            self.draw_graphs(stats, effective_runs)
            
        except Exception as e:
            self.lbl_rec.configure(text=f"ОШИБКА: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.btn_run.configure(text="ЗАПУСТИТЬ РАСЧЕТ", state="normal")

    def draw_graphs(self, stats, runs):
        # 1. Линейный график
        if self.canvas_line: self.canvas_line.get_tk_widget().destroy()
        fig1 = Figure(figsize=(6, 4), dpi=100)
        fig1.patch.set_facecolor('#2b2b2b'); ax1 = fig1.add_subplot(111); ax1.set_facecolor('#2b2b2b')
        days = range(1, self.model.n + 1)
        
        ax1.plot(days, np.cumsum(stats['Ideal']['dynamics_sum']/runs), 'w--', label='Ideal')
        colors = ['#e76f51', '#2a9d8f', '#e9c46a', '#f4a261']
        for i, name in enumerate([k for k in stats if k!='Ideal']):
            ax1.plot(days, np.cumsum(stats[name]['dynamics_sum']/runs), color=colors[i%4], label=name)
        
        ax1.grid(True, linestyle='--', alpha=0.3); ax1.legend(facecolor='#2b2b2b', labelcolor='white')
        ax1.tick_params(colors='white'); [s.set_color('white') for s in ax1.spines.values()]
        
        self.canvas_line = FigureCanvasTkAgg(fig1, master=self.frame_line)
        self.canvas_line.draw(); self.canvas_line.get_tk_widget().pack(fill="both", expand=True)
        
        # 2. Бар график
        if self.canvas_bar: self.canvas_bar.get_tk_widget().destroy()
        fig2 = Figure(figsize=(6, 4), dpi=100)
        fig2.patch.set_facecolor('#2b2b2b'); ax2 = fig2.add_subplot(111); ax2.set_facecolor('#2b2b2b')
        
        names = ['Max'] + [k for k in stats if k!='Ideal']
        vals = [np.mean(stats['Ideal']['totals'])] + [np.mean(stats[k]['totals']) for k in stats if k!='Ideal']
        
        ax2.bar(names, vals, color=['#2ec4b6'] + ['#457b9d']*len(names), alpha=0.9)
        ax2.tick_params(colors='white'); [s.set_color('white') for s in ax2.spines.values()]
        
        self.canvas_bar = FigureCanvasTkAgg(fig2, master=self.frame_bar)
        self.canvas_bar.draw(); self.canvas_bar.get_tk_widget().pack(fill="both", expand=True)


if __name__ == "__main__":
    app = FinalApp()
    app.mainloop()