import customtkinter as ctk
import numpy as np
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.figure import Figure
import tkinter as tk
from tkinter import ttk, messagebox

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
        
        # Параметры масштабирования (ТЗ)
        self.daily_mass = 3000.0  # Тонн в сутки
        self.days_per_stage = 7.0  # Дней в одном этапе (неделя)
        
        self.matrix_s = None
        self.matrix_beta_avg = None 
        
        # Диапазоны (ТЗ)
        # Сахар задается в долях (0.12 - 0.22)
        self.ranges = {
            'a': (0.12, 0.22), 
            'beta_wither': (0.85, 1.00),
            'beta_ripen': (1.00, 1.15),
            'K': (4.8, 7.05), 
            'Na': (0.21, 0.82), 
            'N': (1.58, 2.80), 
            'I0': (0.62, 0.64)
        }

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
        # Внутренние расчеты потерь ведутся в процентах (%), 
        # но в матрицу S мы записываем долю (0.0 - 1.0)
        
        C_fraction = np.zeros((self.n, self.n)) # Матрица сахара в долях без потерь
        S_fraction = np.zeros((self.n, self.n)) # Итоговая матрица выхода в долях
        r = self.ranges
        
        a = np.random.uniform(*r['a'], self.n) # Начальный сахар (доли, напр 0.16)
        K = np.random.uniform(*r['K'], self.n)
        Na = np.random.uniform(*r['Na'], self.n)
        N = np.random.uniform(*r['N'], self.n)
        I0 = np.random.uniform(*r['I0'], self.n)
        
        row_centers = np.random.uniform(r['beta_wither'][0], r['beta_wither'][1], self.n)
        self.matrix_beta_avg = row_centers

        for j in range(self.n): # Столбцы (этапы)
            # Реальное количество прошедших дней
            days_passed = j * self.days_per_stage
            
            for i in range(self.n): # Строки (партии)
                # 1. Расчет коэффициента лежкости (Beta)
                beta = 1.0
                if j > 0:
                    beta = self._get_beta(j, i, row_centers)
                
                # 2. Расчет сахаристости с учетом увядания/дозаривания (без химии пока)
                if j == 0:
                    C_fraction[i, j] = a[i]
                else:
                    C_fraction[i, j] = C_fraction[i, j-1] * beta
                
                # Переводим текущую долю сахара в проценты для формул потерь
                Cx_percent = C_fraction[i, j] * 100.0
                
                # 3. Расчет химических потерь (ТЗ)
                loss_percent = 0
                if self.use_chemistry:
                    # I растет от времени: I = I0 * (1.029)^days
                    # Формула ТЗ: I(x) = I0 * (1.029)^x
                    I_curr = I0[i] * (1.029 ** days_passed)
                    
                    # Формула (51): M_Cx - потери в мелассе
                    M_Cx = 0.1541*(K[i] + Na[i]) + 0.2159*N[i] + 0.9989*I_curr + 0.1967
                    
                    # Формула (50): L - общие потери (M_Cx + 1.1% заводские)
                    loss_percent = M_Cx + 1.1
                
                # Итоговый выход сахара в процентах
                S_percent = Cx_percent - loss_percent
                
                # Переводим обратно в доли и записываем в матрицу. Не может быть меньше 0.
                S_fraction[i, j] = max(0.0, S_percent / 100.0)
                
        self.matrix_s = S_fraction

    def set_manual_matrix(self, matrix, manual_nu):
        self.matrix_s = np.array(matrix)
        self.n = self.matrix_s.shape[0]
        self.nu = manual_nu
        rng = np.random.RandomState(42)
        self.matrix_beta_avg = rng.uniform(0.9, 0.98, self.n)

    def solve_hungarian_max(self):
        """Решает задачу максимизации (Max Possible Yield)"""
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        return total

    def solve_hungarian_min(self):
        """Решает задачу минимизации (Min Possible Yield)"""
        row_ind, col_ind = linear_sum_assignment(self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        return total
    
    def solve_hungarian_dynamics(self):
        """Возвращает динамику для идеального случая (Max)"""
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        schedule = sorted(zip(col_ind, row_ind), key=lambda x: x[0])
        daily_yields = [self.matrix_s[batch, day] for day, batch in schedule]
        return total, daily_yields

    # --- Стратегии ---
    def logic_greedy(self, day, available): return max(available, key=lambda i: self.matrix_s[i, day])
    def logic_thrifty(self, day, available): return min(available, key=lambda i: self.matrix_s[i, day])
    def logic_tg(self, day, available):
        return self.logic_thrifty(day, available) if day < (self.nu - 1) else self.logic_greedy(day, available)
    def logic_gt(self, day, available):
        return self.logic_greedy(day, available) if day < (self.nu - 1) else self.logic_thrifty(day, available)
    def logic_ctg(self, day, available):
        return min(available, key=lambda i: self.matrix_beta_avg[i])
    def logic_critical(self, day, available):
        return max(available, key=lambda i: self.matrix_s[i, day] / self.matrix_beta_avg[i])
    def logic_mean_std(self, day, available):
        vals = [self.matrix_s[i, day] for i in available]
        mu = np.mean(vals)
        sigma = np.std(vals)
        threshold = mu + 0.5 * sigma
        candidates = [i for i in available if self.matrix_s[i, day] >= threshold]
        if candidates: return max(candidates, key=lambda i: self.matrix_s[i, day])
        else: return self.logic_greedy(day, available)
    def logic_classification(self, day, available):
        progress = day / self.n
        if progress < 0.3: return self.logic_thrifty(day, available)
        elif progress < 0.7: return self.logic_ctg(day, available)
        else: return self.logic_greedy(day, available)
        
    def run_simulation(self, runs=50, manual_mode=False):
        strategies = {
            'Greedy': self.logic_greedy,
            'Thrifty': self.logic_thrifty,
            'Thrifty->Greedy': self.logic_tg,
            'Greedy->Thrifty': self.logic_gt,
            'CTG (BetaSort)': self.logic_ctg,
            'Critical Ratio': self.logic_critical,
            'Mean+StdDev': self.logic_mean_std,
            'Classification': self.logic_classification
        }
        
        stats = {k: {'totals': [], 'dynamics_sum': np.zeros(self.n)} for k in strategies}
        stats['Ideal'] = {'totals': [], 'dynamics_sum': np.zeros(self.n)}
        
        # Добавляем хранилище для Min yield
        min_yields = []
        
        effective_runs = 1 if manual_mode else runs
        
        for r in range(effective_runs):
            if not manual_mode:
                self.generate_matrix()
            
            # Max Ideal
            id_sum, id_dyn = self.solve_hungarian_dynamics()
            stats['Ideal']['totals'].append(id_sum)
            stats['Ideal']['dynamics_sum'] += np.array(id_dyn)
            
            # Min Ideal (только сумма)
            min_val = self.solve_hungarian_min()
            min_yields.append(min_val)
            
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
                
        return stats, min_yields, effective_runs

# =============================================================================
# 2. UI: ОКНА И ПАНЕЛИ
# =============================================================================

class StrategyHelpWindow(ctk.CTkToplevel):
    def __init__(self, master):
        super().__init__(master)
        self.title("Справочник стратегий")
        self.geometry("600x500")
        self.resizable(False, False)
        ctk.CTkLabel(self, text="Описание алгоритмов", font=("Arial", 20, "bold")).pack(pady=10)
        textbox = ctk.CTkTextbox(self, width=550, height=400, font=("Arial", 14), wrap="word")
        textbox.pack(padx=20, pady=10)
        info_text = (
            "1. Greedy (Жадная)\nНа каждом шаге выбирает партию с максимальным текущим содержанием сахара.\n\n"
            "2. Thrifty (Бережливая)\nВыбирает партию с минимальным содержанием сахара, оставляя лучшие 'на потом'.\n\n"
            "3. Thrifty -> Greedy\nДо дня N (nu) работает как Бережливая, затем переключается на Жадную.\n\n"
            "4. CTG (BetaSort)\nПриоритет отдается партиям с худшим коэффициентом лежкости.\n\n"
            "5. Critical Ratio (Критическая деградация)\nВыбирает партию с максимальным отношением Сахар / Коэф.Деградации.\n\n"
            "6. Mean + StdDev (Выбор лучших)\nРассматривает только те партии, сахар в которых выше 'Среднего + 0.5 Std'.\n\n"
            "7. Classification (Группировка)\nГибрид: первые 30% — Бережливая, середина — CTG, концовка — Жадная."
        )
        textbox.insert("0.0", info_text)
        textbox.configure(state="disabled")

class AutoSettingsFrame(ctk.CTkScrollableFrame):
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
        # Порядок: (min, max)
        self.add_range_input("Сахар (доли)", "a", model.ranges['a'])
        self.add_range_input("Увядание (beta<1)", "beta_wither", model.ranges['beta_wither'])
        self.add_range_input("Дозаривание (beta>1)", "beta_ripen", model.ranges['beta_ripen'])
        self.add_range_input("Калий (K)", "K", model.ranges['K'])
        self.add_range_input("Натрий (Na)", "Na", model.ranges['Na'])
        self.add_range_input("Азот (N)", "N", model.ranges['N'])
        
        self.add_section("4. Масштабирование")
        self.add_input("Тонн в сутки", "daily_mass", str(model.daily_mass))
        self.add_input("Дней в этапе", "days_per_stage", str(model.days_per_stage))

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
        # e1 - левое поле (min), e2 - правое поле (max)
        e2 = ctk.CTkEntry(f, width=45); e2.insert(0, str(default_tuple[1])); e2.pack(side="right", padx=2)
        e1 = ctk.CTkEntry(f, width=45); e1.insert(0, str(default_tuple[0])); e1.pack(side="right", padx=2)
        self.entries[key] = (e1, e2)
        
    def get_params(self):
        try:
            # Сбор простых значений
            vals = {
                'n': int(self.entries['n'].get()),
                'nu': int(self.entries['nu'].get()),
                'runs': int(self.entries['runs'].get()),
                'use_ripening': bool(self.sw_rip.get()),
                'use_chemistry': bool(self.sw_chem.get()),
                'distribution': 'concentrated' if self.sw_dist.get() else 'uniform',
                'daily_mass': float(self.entries['daily_mass'].get()),
                'days_per_stage': float(self.entries['days_per_stage'].get()),
                'ranges': {}
            }
            
            # Сбор и валидация диапазонов
            for k in ['a', 'beta_wither', 'beta_ripen', 'K', 'Na', 'N']:
                v1 = float(self.entries[k][0].get()) # левое
                v2 = float(self.entries[k][1].get()) # правое
                
                # Валидация: если перепутали местами
                if v1 > v2: v1, v2 = v2, v1
                
                # Валидация: защита от отрицательных чисел (где это нелогично)
                if v1 < 0: v1 = 0
                if v2 < 0: v2 = 0
                
                vals['ranges'][k] = (v1, v2)
                
            vals['ranges']['I0'] = (0.62, 0.64)
            return vals
        except ValueError:
            return None # Ошибка парсинга

class ManualSettingsFrame(ctk.CTkFrame):
     def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        ctk.CTkLabel(self, text="Матрица выхода S (строки через Enter):", font=("Arial", 12, "bold")).pack(anchor="w", pady=(10, 5))
        self.textbox = ctk.CTkTextbox(self, font=("Consolas", 12), height=200)
        self.textbox.pack(fill="x", pady=5)
        self.textbox.insert("0.0", "0.15 0.14 0.13\n0.14 0.13 0.12\n0.16 0.15 0.14")
        sep = ctk.CTkFrame(self, height=2, fg_color="gray")
        sep.pack(fill="x", pady=15)
        ctk.CTkLabel(self, text="Параметры обработки:", font=("Arial", 12, "bold")).pack(anchor="w")
        f_nu = ctk.CTkFrame(self, fg_color="transparent")
        f_nu.pack(fill="x", pady=5)
        ctk.CTkLabel(f_nu, text="Nu (День смены стратегии):").pack(side="left")
        self.entry_nu = ctk.CTkEntry(f_nu, width=60)
        self.entry_nu.insert(0, "2")
        self.entry_nu.pack(side="right")
        ctk.CTkLabel(self, text="* Остальные параметры (химия, диапазоны)\nпри ручном вводе не применяются,\nт.к. вы вводите уже финальный выход.", font=("Arial", 11), text_color="#e07a5f").pack(pady=10)
     def get_data(self):
        text = self.textbox.get("0.0", "end").strip()
        if not text: return None, None
        try:
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
        self.title("Sugar Beet Optimization DSS v2.0")
        self.geometry("1400x900") 
        
        self.model = SugarBeetModel()
        
        self.last_stats = None
        self.last_min_yields = None # Для Min Yield
        self.last_runs = 0

        self.grid_columnconfigure(0, weight=0, minsize=350)
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)
        
        # --- ЛЕВАЯ ПАНЕЛЬ ---
        self.left_frame = ctk.CTkFrame(self, corner_radius=0)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        
        self.header_frame = ctk.CTkFrame(self.left_frame, fg_color="transparent")
        self.header_frame.pack(fill="x", padx=10, pady=(20, 10))
        ctk.CTkLabel(self.header_frame, text="ДАННЫЕ", font=("Arial", 20, "bold")).pack(side="left", padx=10)
        self.btn_help = ctk.CTkButton(self.header_frame, text="?", width=30, height=30, 
                                      fg_color="#3a7ebf", font=("Arial", 14, "bold"),
                                      command=self.open_help)
        self.btn_help.pack(side="right", padx=10)
        
        self.tab_selector = ctk.CTkTabview(self.left_frame)
        self.tab_selector.pack(expand=True, fill="both", padx=10, pady=(0, 10))
        self.tab_auto = self.tab_selector.add("Авто-Генерация")
        self.tab_manual = self.tab_selector.add("Ручной Ввод")
        self.auto_config = AutoSettingsFrame(self.tab_auto, self.model)
        self.auto_config.pack(expand=True, fill="both")
        self.manual_config = ManualSettingsFrame(self.tab_manual)
        self.manual_config.pack(expand=True, fill="both", padx=5, pady=5)
        
        self.btn_run = ctk.CTkButton(self.left_frame, text="ЗАПУСТИТЬ РАСЧЕТ", 
                                     height=50, fg_color="green", font=("Arial", 14, "bold"),
                                     command=self.run_process)
        self.btn_run.pack(padx=20, pady=(20, 10), fill="x")

        self.btn_view_matrix = ctk.CTkButton(self.left_frame, text="ПОКАЗАТЬ МАТРИЦУ (LAST)", 
                                             height=40, fg_color="#555", state="disabled", font=("Arial", 12, "bold"),
                                             command=self.open_matrix_window)
        self.btn_view_matrix.pack(padx=20, pady=(0, 20), fill="x")

        # --- ПРАВАЯ ПАНЕЛЬ ---
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_frame.grid_columnconfigure((0, 1, 2, 3), weight=1) # Теперь 4 колонки
        self.right_frame.grid_rowconfigure(2, weight=1)

        # KPI
        self.card_ideal = InfoCard(self.right_frame, "Max Possible Yield", "---", color="#2ec4b6")
        self.card_ideal.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 10))
        
        # НОВАЯ КАРТОЧКА MIN
        self.card_min = InfoCard(self.right_frame, "Min Possible Yield", "---", color="#e63946")
        self.card_min.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 10))
        
        self.card_best = InfoCard(self.right_frame, "Best Strategy", "---", color="#e76f51")
        self.card_best.grid(row=0, column=2, sticky="ew", padx=5, pady=(0, 10))
        self.card_loss = InfoCard(self.right_frame, "Min Loss", "--- %", color="#e9c46a")
        self.card_loss.grid(row=0, column=3, sticky="ew", padx=5, pady=(0, 10))

        # --- Панель управления графиками ---
        self.ctrl_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.ctrl_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        
        self.lbl_slider = ctk.CTkLabel(self.ctrl_frame, text="Топ стратегий: 5", font=("Arial", 12))
        self.lbl_slider.pack(side="left", padx=(10, 10))
        
        self.slider_strat = ctk.CTkSlider(self.ctrl_frame, from_=2, to=8, number_of_steps=6, width=200, command=self.update_graph_view)
        self.slider_strat.set(5)
        self.slider_strat.pack(side="left", padx=10)
        
        self.sw_real_view = ctk.CTkSwitch(self.ctrl_frame, text="Включить реальные единицы (Тонны/Дни)", 
                                          command=self.update_graph_view)
        self.sw_real_view.pack(side="right", padx=20)

        # Графики
        self.tabs_graph = ctk.CTkTabview(self.right_frame)
        self.tabs_graph.grid(row=2, column=0, columnspan=4, sticky="nsew")
        self.tabs_graph.add("Динамика")
        self.tabs_graph.add("Итоги")
        
        self.frame_line = ctk.CTkFrame(self.tabs_graph.tab("Динамика"), fg_color="transparent")
        self.frame_line.pack(fill="both", expand=True)
        self.frame_bar = ctk.CTkFrame(self.tabs_graph.tab("Итоги"), fg_color="transparent")
        self.frame_bar.pack(fill="both", expand=True)
        
        self.canvas_line = None; self.canvas_bar = None
        self.toolbar_line = None; self.toolbar_bar = None

        # Рекомендация
        self.rec_frame = ctk.CTkFrame(self.right_frame, fg_color="#2b2b2b", border_width=1, border_color="#555")
        self.rec_frame.grid(row=3, column=0, columnspan=4, sticky="ew", pady=(15, 0))
        
        ctk.CTkLabel(self.rec_frame, text="РЕКОМЕНДАЦИЯ СППР", font=("Arial", 14, "bold"), text_color="#3a7ebf").pack(anchor="w", padx=20, pady=(10, 0))
        self.lbl_rec = ctk.CTkLabel(self.rec_frame, text="Задайте параметры и запустите расчет...", font=("Consolas", 13), justify="left", wraplength=900)
        self.lbl_rec.pack(anchor="w", padx=20, pady=(5, 15))

    def open_help(self):
        StrategyHelpWindow(self)

    def open_matrix_window(self):
        if self.model.matrix_s is None: return
        
        top = ctk.CTkToplevel(self)
        top.title(f"Последняя сгенерированная матрица ({self.model.n}x{self.model.n})")
        top.geometry("900x600")
        
        style = ttk.Style()
        style.theme_use("default")
        style.configure("Treeview", 
                        background="#2b2b2b", 
                        foreground="white", 
                        fieldbackground="#2b2b2b",
                        font=("Arial", 11),
                        rowheight=25)
        style.configure("Treeview.Heading", 
                        background="#3a3a3a", 
                        foreground="white",
                        font=("Arial", 11, "bold"))
        style.map("Treeview", background=[('selected', '#3a7ebf')])

        frame = ctk.CTkFrame(top, fg_color="transparent")
        frame.pack(fill="both", expand=True, padx=10, pady=10)

        cols = ["Batch"] + [f"Stage {j+1}" for j in range(self.model.n)]
        
        tree = ttk.Treeview(frame, columns=cols, show="headings", style="Treeview")
        
        tree.heading("Batch", text="Партия")
        tree.column("Batch", width=80, anchor="center")
        
        for c in cols[1:]:
            tree.heading(c, text=c)
            tree.column(c, width=70, anchor="center")
            
        for i in range(self.model.n):
            row_vals = [f"Batch {i+1}"] + [f"{val:.2f}" for val in self.model.matrix_s[i]]
            tree.insert("", "end", values=row_vals)

        vsb = ttk.Scrollbar(frame, orient="vertical", command=tree.yview)
        hsb = ttk.Scrollbar(frame, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        
        frame.grid_rowconfigure(0, weight=1)
        frame.grid_columnconfigure(0, weight=1)

    def update_graph_view(self, value=None):
        val = int(self.slider_strat.get())
        self.lbl_slider.configure(text=f"Топ стратегий: {val}")
        
        if self.last_stats is not None:
            self.draw_graphs(self.last_stats, self.last_runs)

    def update_kpi_cards_display(self, best_name, best_loss, ideal_val, min_val):
        """Обновляет карточки с учетом режима отображения"""
        use_real = bool(self.sw_real_view.get())
        
        factor = 1.0
        unit = ""
        if use_real:
            # Масштабируем: Yield (unit) * Mass * Days
            factor = self.model.daily_mass * self.model.days_per_stage
            unit = " T"
        
        scaled_ideal = ideal_val * factor
        scaled_min = min_val * factor
        
        self.card_ideal.update_value(f"{scaled_ideal:,.2f}{unit}")
        self.card_min.update_value(f"{scaled_min:,.2f}{unit}")
        self.card_best.update_value(best_name)
        self.card_loss.update_value(f"{best_loss:.2f}%")

    def run_process(self):
        try:
            self.btn_run.configure(text="Вычисление...", state="disabled")
            self.btn_view_matrix.configure(state="disabled", fg_color="#555")
            self.update()
            
            active_tab = self.tab_selector.get()
            manual_mode = (active_tab == "Ручной Ввод")
            runs = 50
            
            if manual_mode:
                matrix, manual_nu = self.manual_config.get_data()
                if matrix is None: raise ValueError("Матрица пуста!")
                self.model.set_manual_matrix(matrix, manual_nu)
            else:
                p = self.auto_config.get_params()
                if p is None: raise ValueError("Проверьте введенные числа.")
                
                self.model.n = p['n']; self.model.nu = p['nu']
                self.model.use_ripening = p['use_ripening']; self.model.use_chemistry = p['use_chemistry']
                self.model.distribution_type = p['distribution']; self.model.ranges = p['ranges']
                self.model.daily_mass = p['daily_mass']
                self.model.days_per_stage = p['days_per_stage']
                runs = p['runs']

            stats, min_yields, effective_runs = self.model.run_simulation(runs=runs, manual_mode=manual_mode)
            
            self.last_stats = stats
            self.last_min_yields = min_yields
            self.last_runs = effective_runs

            # Анализ
            avg_ideal = np.mean(stats['Ideal']['totals'])
            avg_min = np.mean(min_yields)
            
            results = []
            for name in stats:
                if name == 'Ideal': continue
                val = np.mean(stats[name]['totals'])
                loss = (1 - val/avg_ideal) * 100 if avg_ideal != 0 else 0
                results.append((name, val, loss))
            results.sort(key=lambda x: x[2])
            best = results[0]
            
            self.update_kpi_cards_display(best[0], best[2], avg_ideal, avg_min)
            self.update_recommendation(best[0], best[2], manual_mode)
            self.draw_graphs(stats, effective_runs)
            
            self.btn_view_matrix.configure(state="normal", fg_color="#3a7ebf")
            
        except Exception as e:
            self.lbl_rec.configure(text=f"ОШИБКА: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.btn_run.configure(text="ЗАПУСТИТЬ РАСЧЕТ", state="normal")

    def update_recommendation(self, name, loss, manual_mode):
        text = f"Победитель: {name} (Потери {loss:.2f}%).\n\n"
        advice = ""
        if "Critical" in name:
            advice = "СОВЕТ: В текущих условиях некоторые партии с высоким сахаром портятся слишком быстро. Игнорируйте общую очередь и спасайте их в первую очередь."
        elif "Mean+StdDev" in name:
            advice = "СОВЕТ: Высокая вариативность качества сырья. Откажитесь от переработки 'середнячков', фокусируйтесь только на партиях, значительно превышающих средний уровень."
        elif "Classification" in name:
            advice = "СОВЕТ: Используйте комбинированный подход. Начало сезона - чистка склада (Бережливая), середина - сортировка по лежкости (CTG), конец - жадный сбор."
        elif "Thrifty->Greedy" in name:
            advice = f"СОВЕТ: Выраженный эффект дозаривания. Первые {self.model.nu} дней используйте 'Бережливую' тактику, затем резко переходите на 'Жадную'."
        elif "Greedy" in name:
            advice = "СОВЕТ: Сильное увядание или отсутствие дозаривания. Не ждите - перерабатывайте самое сладкое сырье немедленно."
        elif "CTG" in name:
            advice = "СОВЕТ: Ключевой фактор - лежкость. В первую очередь перерабатывайте партии, которые гниют быстрее всего."
        else:
            advice = f"СОВЕТ: Следуйте стратегии {name}."

        if manual_mode:
            advice += "\n(Примечание: Анализ выполнен для единственной введенной матрицы)."
        self.lbl_rec.configure(text=text + advice)

    def draw_graphs(self, stats, runs):
        top_n = int(self.slider_strat.get())
        use_real = bool(self.sw_real_view.get())

        scale_y = (self.model.daily_mass * self.model.days_per_stage) if use_real else 1.0
        scale_x = self.model.days_per_stage if use_real else 1.0
        
        # Обновляем KPI при смене тумблера
        if self.last_stats and self.last_min_yields:
             avg_ideal = np.mean(stats['Ideal']['totals'])
             avg_min = np.mean(self.last_min_yields)
             results = []
             for name in stats:
                if name == 'Ideal': continue
                val = np.mean(stats[name]['totals'])
                loss = (1 - val/avg_ideal) * 100 if avg_ideal != 0 else 0
                results.append((name, val, loss))
             results.sort(key=lambda x: x[2])
             self.update_kpi_cards_display(results[0][0], results[0][2], avg_ideal, avg_min)

        if self.canvas_line: 
            self.canvas_line.get_tk_widget().destroy()
        if self.toolbar_line:
            self.toolbar_line.destroy()
            
        if self.canvas_bar: 
            self.canvas_bar.get_tk_widget().destroy()
        if self.toolbar_bar:
            self.toolbar_bar.destroy()

        # 1. Line Chart
        fig1 = Figure(figsize=(6, 4), dpi=100)
        fig1.patch.set_facecolor('#2b2b2b'); ax1 = fig1.add_subplot(111); ax1.set_facecolor('#2b2b2b')
        
        steps = range(1, self.model.n + 1)
        x_vals = [s * scale_x for s in steps]
        
        y_ideal = np.cumsum(stats['Ideal']['dynamics_sum']/runs) * scale_y
        ax1.plot(x_vals, y_ideal, 'w--', label='Ideal', alpha=0.5)
        
        sorted_keys = sorted([k for k in stats if k!='Ideal'], key=lambda k: np.mean(stats[k]['totals']), reverse=True)
        top_keys = sorted_keys[:top_n]
        
        colors = ['#e76f51', '#2a9d8f', '#e9c46a', '#f4a261', '#81b29a', '#f1faee', '#a8dadc', '#457b9d']
        for i, name in enumerate(top_keys):
            col = colors[i % len(colors)]
            y_vals = np.cumsum(stats[name]['dynamics_sum']/runs) * scale_y
            ax1.plot(x_vals, y_vals, color=col, label=name, linewidth=2)
            
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(facecolor='#2b2b2b', labelcolor='white')
        ax1.tick_params(colors='white'); [s.set_color('white') for s in ax1.spines.values()]
        
        x_label = "Days (Real Time)" if use_real else "Steps (Matrix Col)"
        y_label = "Cumulative Yield (Tons)" if use_real else "Cumulative Yield (Fraction Units)"
        ax1.set_xlabel(x_label, color='white', fontsize=9)
        ax1.set_ylabel(y_label, color='white', fontsize=9)
        
        self.canvas_line = FigureCanvasTkAgg(fig1, master=self.frame_line)
        self.canvas_line.draw()
        
        self.toolbar_line = NavigationToolbar2Tk(self.canvas_line, self.frame_line)
        self.toolbar_line.update()
        self.canvas_line.get_tk_widget().pack(fill="both", expand=True)
        
        # 2. Bar Chart
        fig2 = Figure(figsize=(6, 4), dpi=100)
        fig2.patch.set_facecolor('#2b2b2b'); ax2 = fig2.add_subplot(111); ax2.set_facecolor('#2b2b2b')
        
        names = ['Max'] + top_keys
        vals = [np.mean(stats['Ideal']['totals'])] + [np.mean(stats[k]['totals']) for k in top_keys]
        vals = [v * scale_y for v in vals]
        
        bars = ax2.bar(names, vals, color=['#2ec4b6'] + ['#457b9d']*len(names), alpha=0.9)
        ax2.tick_params(colors='white', axis='x', labelsize=8); [s.set_color('white') for s in ax2.spines.values()]
        
        fmt_str = '%.0f' if use_real else '%.2f'
        ax2.bar_label(bars, fmt=fmt_str, color='white', padding=3)
        ax2.set_ylabel(y_label, color='white', fontsize=9)
        
        self.canvas_bar = FigureCanvasTkAgg(fig2, master=self.frame_bar)
        self.canvas_bar.draw()

        self.toolbar_bar = NavigationToolbar2Tk(self.canvas_bar, self.frame_bar)
        self.toolbar_bar.update()
        self.canvas_bar.get_tk_widget().pack(fill="both", expand=True)

if __name__ == "__main__":
    app = FinalApp()
    app.mainloop()