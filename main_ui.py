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
        self.nu = 10
        self.use_ripening = True
        self.use_chemistry = True
        self.distribution_type = 'concentrated'
        
        # Параметры масштабирования (ТЗ)
        self.daily_mass = 3000.0  # Тонн в сутки
        self.days_per_stage = 7.0  # Дней в одном этапе (неделя)
        
        self.matrix_s = None
        self.matrix_beta_avg = None 
        
        # Диапазоны (ТЗ)
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
        C_fraction = np.zeros((self.n, self.n)) 
        S_fraction = np.zeros((self.n, self.n)) 
        r = self.ranges
        
        a = np.random.uniform(*r['a'], self.n) 
        K = np.random.uniform(*r['K'], self.n)
        Na = np.random.uniform(*r['Na'], self.n)
        N = np.random.uniform(*r['N'], self.n)
        I0 = np.random.uniform(*r['I0'], self.n)
        
        row_centers = np.random.uniform(r['beta_wither'][0], r['beta_wither'][1], self.n)
        self.matrix_beta_avg = row_centers

        for j in range(self.n): 
            days_passed = j * self.days_per_stage
            
            for i in range(self.n): 
                beta = 1.0
                if j > 0:
                    beta = self._get_beta(j, i, row_centers)
                
                if j == 0:
                    C_fraction[i, j] = a[i]
                else:
                    C_fraction[i, j] = C_fraction[i, j-1] * beta
                
                Cx_percent = C_fraction[i, j] * 100.0
                
                loss_percent = 0
                if self.use_chemistry:
                    I_curr = I0[i] * (1.029 ** days_passed)
                    M_Cx = 0.1541*(K[i] + Na[i]) + 0.2159*N[i] + 0.9989*I_curr + 0.1967
                    loss_percent = M_Cx + 1.1
                
                S_percent = Cx_percent - loss_percent
                S_fraction[i, j] = max(0.0, S_percent / 100.0)
                
        self.matrix_s = S_fraction

    def set_manual_matrix(self, matrix, manual_nu):
        self.matrix_s = np.array(matrix)
        self.n = self.matrix_s.shape[0]
        self.nu = manual_nu
        rng = np.random.RandomState(42)
        self.matrix_beta_avg = rng.uniform(0.9, 0.98, self.n)

    def solve_hungarian_max(self):
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        return total

    def solve_hungarian_min(self):
        row_ind, col_ind = linear_sum_assignment(self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        return total
    
    def solve_hungarian_dynamics(self):
        row_ind, col_ind = linear_sum_assignment(-self.matrix_s)
        total = self.matrix_s[row_ind, col_ind].sum()
        schedule = sorted(zip(col_ind, row_ind), key=lambda x: x[0])
        daily_yields = [self.matrix_s[batch, day] for day, batch in schedule]
        return total, daily_yields
    
    def solve_hungarian_min_dynamics(self):
        row_ind, col_ind = linear_sum_assignment(self.matrix_s)
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
    def logic_tkg(self, day, available, k=1):
        if day < self.nu - 1: 
            pairs = [(i, self.matrix_s[i, day]) for i in available]
            sorted_pairs = sorted(pairs, key=lambda x: x[1])
            if k <= len(sorted_pairs):
                return sorted_pairs[k-1][0]
            else:
                return sorted_pairs[-1][0]
        else: 
            return self.logic_greedy(day, available)
        
    def run_simulation(self, runs=50, manual_mode=False, k_param=1):
        strategies = {
            'Жадная': self.logic_greedy,
            'Бережливая': self.logic_thrifty,
            'Бережливая/Жадная': self.logic_tg,
            'Жадная/Бережливая': self.logic_gt,
            'БkЖ (T(k)G)': lambda d, a: self.logic_tkg(d, a, k=k_param),
            'CTG': self.logic_ctg,
            'Критической\n деградации': self.logic_critical,
            'Среднее+Отклонение': self.logic_mean_std,
            'Фазовая группировка': self.logic_classification
        }
        
        stats = {k: {'totals': [], 'dynamics_sum': np.zeros(self.n)} for k in strategies}
        stats['Ideal'] = {'totals': [], 'dynamics_sum': np.zeros(self.n)}
        stats['Min'] = {'totals': [], 'dynamics_sum': np.zeros(self.n)}
        
        effective_runs = runs
        
        for r in range(effective_runs):
            if not manual_mode:
                self.generate_matrix()
            
            id_sum, id_dyn = self.solve_hungarian_dynamics()
            stats['Ideal']['totals'].append(id_sum)
            stats['Ideal']['dynamics_sum'] += np.array(id_dyn)
            
            # Худшая стратегия (Венгерский минимум)
            min_sum, min_dyn = self.solve_hungarian_min_dynamics() 
            stats['Min']['totals'].append(min_sum)
            stats['Min']['dynamics_sum'] += np.array(min_dyn)
            
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
            "ЭВРИСТИЧЕСКИЕ СТРАТЕГИИ ПЛАНИРОВАНИЯ ПЕРЕРАБОТКИ САХАРНОЙ СВЕКЛЫ\n\n"
            
            "1. ЖАДНАЯ СТРАТЕГИЯ (Greedy)\n"
            "На каждом этапе перерабатывается партия с максимальной текущей сахаристостью.\n"
            "Эффективна при быстрой деградации сырья, когда ожидание приводит к потерям.\n\n"
            
            "2. БЕРЕЖЛИВАЯ СТРАТЕГИЯ (Thrifty)\n"
            "На каждом этапе перерабатывается партия с минимальной сахаристостью.\n"
            "Оптимальна при дозаривании, когда качество сырья со временем улучшается.\n\n"
            
            "3. БЕРЕЖЛИВАЯ/ЖАДНАЯ СТРАТЕГИЯ\n"
            "Первые (ν-1) этапов: бережливый алгоритм, затем - жадный.\n"
            "Позволяет сначала накопить потенциал за счет дозаривания, затем собрать максимум.\n\n"
            
            "4. ЖАДНАЯ/БЕРЕЖЛИВАЯ СТРАТЕГИЯ\n"
            "Первые (ν-1) этапов: жадный алгоритм, затем - бережливый.\n"
            "Применяется, когда первоначальная переработка лучшего сырья экономически выгодна.\n\n"
            
            "5. СТРАТЕГИЯ БkЖ (T(k)G)\n"
            "На первых (ν-1) этапах перерабатывается k-я партия от наихудшей по сахаристости.\n"
            "Балансирует между сохранением лучшего сырья и использованием средних партий.\n"
            "Параметр k регулирует агрессивность: от консервативной (k=1) к более активной.\n\n"
            
            "6. СТРАТЕГИЯ CTG (Сортировка по лежкости)\n"
            "Партии упорядочиваются по коэффициентам деградации, сначала перерабатываются\n"
            "партии с наихудшей лежкостью, независимо от текущей сахаристости.\n"
            "Эффективна при сильной вариабельности сохранности партий.\n\n"
            
            "7. СТРАТЕГИЯ КРИТИЧЕСКОЙ ДЕГРАДАЦИИ (Critical Ratio)\n"
            "Максимизирует отношение сахаристость/коэффициент деградации.\n"
            "Приоритетно обрабатывает партии с высокой сахаристостью, но низкой лежкостью.\n\n"
            
            "8. СРЕДНЕЕ+ОТКЛОНЕНИЕ (Mean+StdDev)\n"
            "Рассматривает только партии с сахаристостью выше среднего + 0.5 стандартных отклонений.\n"
            "Концентрируется на лучшем сырье, игнорируя средние и низкокачественные партии.\n\n"
            
            "9. ФАЗОВАЯ ГРУППИРОВКА (Classification)\n"
            "Разделяет сезон на три фазы с разными алгоритмами:\n"
            "- Начало (30%): бережливая\n"
            "- Середина (40%): CTG\n"
            "- Завершение (30%): жадная\n"
            "Адаптируется к изменяющейся динамике процесса.\n\n"
            
            "ВЫБОР СТРАТЕГИИ ЗАВИСИТ ОТ:\n"
            "• Характера изменения сахаристости (дозаривание/увядание)\n"
            "• Вариабельности коэффициентов лежкости между партиями\n"
            "• Распределения качества сырья (равномерное/концентрированное)\n"
            "• Фазности процесса в течение сезона"
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
        self.add_input("k (для БkЖ стратегии)", "k_param", "5")
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
                'k_param': int(self.entries['k_param'].get()),
                'use_ripening': bool(self.sw_rip.get()),
                'use_chemistry': bool(self.sw_chem.get()),
                'distribution': 'concentrated' if self.sw_dist.get() else 'uniform',
                'daily_mass': float(self.entries['daily_mass'].get()),
                'days_per_stage': float(self.entries['days_per_stage'].get()),
                'ranges': {}
            }
            
            errors = []
            
            # --- ШАГ 1: Проверка n ---
            if vals['n'] <= 0:
                errors.append(f"N (кол-во партий): должно быть > 0")
            
            # --- ШАГ 2: Проверка Nu (только если n корректно) ---
            max_nu = vals['n']  # максимальное значение для Nu
            if vals['nu'] <= 0:
                errors.append(f"Nu (день переключения): должно быть > 0. Максимум: {max_nu}")
            elif vals['nu'] > vals['n']:
                errors.append(f"Nu (день переключения): не может быть больше N={vals['n']}. Максимум: {max_nu}")
            
            # --- ШАГ 3: Проверка k (только если n и nu корректны) ---
            # k может быть от 1 до (n - nu + 1), но проверяем только если n и nu валидны
            if vals['n'] > 0 and 0 < vals['nu'] <= vals['n']:
                max_k = max(1, vals['n'] - vals['nu'] + 1) if vals['nu'] <= vals['n'] else vals['n']
                if vals['k_param'] < 1:
                    errors.append(f"k (для стратегии БkЖ): должно быть ≥ 1. Максимум: {max_k}")
                elif vals['k_param'] > max_k:
                    errors.append(f"k (для стратегии БkЖ): должно быть ≤ {max_k} (n - nu + 1 = {vals['n']} - {vals['nu']} + 1)")
            
            # --- Валидация диапазонов параметров ---
            LIMITS = {
                'a': (0.12, 0.22, "Сахар (доли)"),
                'beta_wither': (0.85, 1.00, "Увядание"),
                'beta_ripen': (1.00, 1.15, "Дозаривание"),
                'K': (4.8, 7.05, "Калий"),
                'Na': (0.21, 0.82, "Натрий"),
                'N': (1.58, 2.80, "Азот")
            }
            
            # Сбор, валидация и проверка диапазонов
            for k in ['a', 'beta_wither', 'beta_ripen', 'K', 'Na', 'N']:
                v1 = float(self.entries[k][0].get()) # левое
                v2 = float(self.entries[k][1].get()) # правое
                
                # Валидация: если перепутали местами
                if v1 > v2: v1, v2 = v2, v1
                
                # Защита от отрицательных чисел
                if v1 < 0: v1 = 0
                if v2 < 0: v2 = 0
                
                # Проверка на соответствие глобальным границам ТЗ
                min_allowed, max_allowed, name = LIMITS[k]
                
                # Если введенный пользователем диапазон выходит за рамки ТЗ
                if v1 < min_allowed or v2 > max_allowed:
                    errors.append(f"{name}: Допустимо от {min_allowed} до {max_allowed}")
                
                vals['ranges'][k] = (v1, v2)

            # Проверка скалярных величин
            if vals['daily_mass'] <= 0:
                errors.append("Тонн в сутки: должно быть число > 0")
            if vals['days_per_stage'] <= 0:
                errors.append("Дней в этапе: должно быть число > 0")
            if vals['runs'] <= 0:
                errors.append("Число прогонов: должно быть > 0")
            
            # Если есть ошибки, выводим предупреждение и не возвращаем параметры
            if errors:
                error_msg = "Обнаружены некорректные данные:\n\n" + "\n".join(errors)
                messagebox.showwarning("Ошибка входных данных", error_msg)
                return None
                
            vals['ranges']['I0'] = (0.62, 0.64)
            return vals
        except ValueError:
            messagebox.showwarning("Ошибка", "Пожалуйста, убедитесь, что все поля заполнены числами.")
            return None

class ManualSettingsFrame(ctk.CTkFrame):
    def __init__(self, master, **kwargs):
        super().__init__(master, fg_color="transparent", **kwargs)
        
        # Заголовок
        ctk.CTkLabel(self, text="Ручной режим", font=("Arial", 14, "bold")).pack(anchor="w", pady=(0, 10))
        
        # Кнопка для ввода матрицы
        self.btn_open_matrix = ctk.CTkButton(self, text="📋 Ввести матрицу", 
                                           height=40, font=("Arial", 13, "bold"),
                                           command=self.open_matrix_editor)
        self.btn_open_matrix.pack(fill="x", pady=(0, 20))
        
        ctk.CTkLabel(self, text="Текущий размер: 15×15", font=("Arial", 11), 
                    text_color="#a0a0a0").pack(anchor="w", pady=(0, 10))
        
        # Разделитель
        sep = ctk.CTkFrame(self, height=2, fg_color="gray")
        sep.pack(fill="x", pady=10)
        
        # Параметры обработки
        ctk.CTkLabel(self, text="Параметры обработки:", font=("Arial", 12, "bold")).pack(anchor="w")
        
        # Nu
        f_nu = ctk.CTkFrame(self, fg_color="transparent")
        f_nu.pack(fill="x", pady=5)
        ctk.CTkLabel(f_nu, text="Nu (День смены стратегии):").pack(side="left")
        self.entry_nu = ctk.CTkEntry(f_nu, width=80)
        self.entry_nu.insert(0, "10")
        self.entry_nu.pack(side="right")
        
        # k
        f_k = ctk.CTkFrame(self, fg_color="transparent")
        f_k.pack(fill="x", pady=5)
        ctk.CTkLabel(f_k, text="k (для стратегии БkЖ):").pack(side="left")
        self.entry_k = ctk.CTkEntry(f_k, width=80)
        self.entry_k.insert(0, "5")
        self.entry_k.pack(side="right")
        
        # Информация
        ctk.CTkLabel(self, 
                    text="* Нажмите 'Ввести матрицу' для открытия редактора\n* Матрица по умолчанию: 15×15\n* Nu должен быть ≤ размеру матрицы",
                    font=("Arial", 11), text_color="#e07a5f", justify="left").pack(anchor="w", pady=15)
        
        # Хранилище данных
        self.matrix_data = None
        self.current_size = 15
    
    def open_matrix_editor(self):
        """Открывает окно редактора матрицы"""
        editor = MatrixEditorWindow(self, self.current_size, self.matrix_data)
        editor.grab_set()  # Модальное окно
        self.wait_window(editor)
        
        # Получаем данные после закрытия окна
        if editor.result_data:
            self.matrix_data = editor.result_data
            self.current_size = editor.result_size
            # Обновляем подпись
            for widget in self.winfo_children():
                if isinstance(widget, ctk.CTkLabel) and "Текущий размер:" in widget.cget("text"):
                    widget.configure(text=f"Текущий размер: {self.current_size}×{self.current_size}")
                    break
    
    def get_data(self):
        """Получает данные матрицы и параметры"""
        # Проверка наличия матрицы
        if self.matrix_data is None:
            messagebox.showwarning("Нет данных", "Сначала введите матрицу, нажав 'Ввести матрицу'")
            return None, None, None, None
        
        try:
            current_size = self.current_size
            errors = []
            
            # --- ШАГ 1: Проверка Nu ---
            try:
                nu = int(self.entry_nu.get())
                max_nu = current_size
                
                if nu <= 0:
                    errors.append(f"Nu: должно быть > 0. Максимум: {max_nu}")
                elif nu > current_size:
                    errors.append(f"Nu: не может быть больше N={current_size}.")
                else:
                    # Если Nu корректно, сохраняем его
                    pass
            except ValueError:
                errors.append("Nu: должно быть целым числом")
                nu = None
            
            # --- ШАГ 2: Проверка k (только если Nu корректно) ---
            k = None
            if nu and 0 < nu <= current_size:
                try:
                    k = int(self.entry_k.get())
                    max_k = max(1, current_size - nu + 1) if nu <= current_size else current_size
                    
                    if k < 1:
                        errors.append(f"k: должно быть ≥ 1. Максимум: {max_k}")
                    elif k > max_k:
                        errors.append(f"k: должно быть ≤ {max_k} (n - nu + 1 = {current_size} - {nu} + 1)")
                except ValueError:
                    errors.append("k: должно быть целым числом")
                    k = None
            
            # Если есть ошибки, показываем их и не возвращаем данные
            if errors:
                error_msg = "Обнаружены некорректные данные:\n\n" + "\n".join(errors)
                messagebox.showwarning("Ошибка параметров", error_msg)
                return None, None, None, None
            
            # Если все ок, устанавливаем значения по умолчанию для некорректных
            if nu is None:
                nu = min(2, current_size)  # безопасное значение
                messagebox.showwarning("Коррекция Nu", f"Nu установлен в {nu} (значение по умолчанию)")
            
            if k is None:
                max_k_safe = max(1, current_size - nu + 1) if nu <= current_size else current_size
                k = min(1, max_k_safe)  # безопасное значение
                messagebox.showwarning("Коррекция k", f"k установлен в {k} (минимальное значение)")
            
            runs = 1  # Фиксировано для ручного режима
            
            return self.matrix_data, nu, k, runs
            
        except Exception as e:
            messagebox.showerror("Ошибка", f"Ошибка обработки параметров:\n{e}")
            return None, None, None, None


class MatrixEditorWindow(ctk.CTkToplevel):
    """Окно редактора матрицы"""
    def __init__(self, parent, current_size=15, existing_data=None):
        super().__init__(parent)
        self.parent = parent
        
        self.title("Редактор матрицы")
        self.geometry("900x600")
        self.resizable(True, True)
        
        # Данные
        self.size = current_size
        self.result_data = existing_data
        self.result_size = current_size
        
        # Создаем интерфейс
        self.create_widgets()
        
        # Если есть данные - заполняем, иначе дефолт для 15×15
        if self.result_data is None and self.size == 15:
            self.fill_default_15x15()
    
    def create_widgets(self):
        """Создает интерфейс редактора"""
        # Основной контейнер
        main_container = ctk.CTkFrame(self)
        main_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Верхняя панель управления
        top_panel = ctk.CTkFrame(main_container, fg_color="transparent")
        top_panel.pack(fill="x", pady=(0, 10))
        
        ctk.CTkLabel(top_panel, text="Редактор матрицы", 
                    font=("Arial", 14, "bold")).pack(side="left")
        
        # Управление размером справа
        size_frame = ctk.CTkFrame(top_panel, fg_color="transparent")
        size_frame.pack(side="right")
        
        ctk.CTkLabel(size_frame, text="Размер N×N:").pack(side="left", padx=(0, 5))
        self.size_var = ctk.StringVar(value=str(self.size))
        self.size_combo = ctk.CTkComboBox(size_frame, 
                                         values=["5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "20", "25"],
                                         variable=self.size_var,
                                         width=70)
        self.size_combo.pack(side="left", padx=(0, 10))
        
        ctk.CTkButton(size_frame, text="Изменить", width=80,
                     command=self.change_size).pack(side="left")
        
        # Область с матрицей
        matrix_frame = ctk.CTkFrame(main_container)
        matrix_frame.pack(fill="both", expand=True, pady=(0, 10))
        
        # Создаем таблицу
        self.create_matrix_table(matrix_frame)
        
        # Нижняя панель с кнопками
        bottom_panel = ctk.CTkFrame(main_container, fg_color="transparent")
        bottom_panel.pack(fill="x")
        
        # Левая часть - кнопка очистки
        left_buttons = ctk.CTkFrame(bottom_panel, fg_color="transparent")
        left_buttons.pack(side="left")
        
        ctk.CTkButton(left_buttons, text="Очистить", 
                     width=80, command=self.clear_matrix).pack(side="left", padx=5)
        
        # Правая часть - кнопки сохранения
        right_buttons = ctk.CTkFrame(bottom_panel, fg_color="transparent")
        right_buttons.pack(side="right")
        
        ctk.CTkButton(right_buttons, text="Отмена", 
                     width=80, command=self.cancel, fg_color="#555").pack(side="left", padx=5)
        
        ctk.CTkButton(right_buttons, text="Сохранить", 
                     width=80, command=self.save, fg_color="green").pack(side="left", padx=5)
    
    def create_matrix_table(self, parent):
        """Создает таблицу для ввода матрицы"""
        # Создаем фрейм с прокруткой
        container = ctk.CTkFrame(parent)
        container.pack(fill="both", expand=True)
        
        # Канвас для прокрутки
        canvas = tk.Canvas(container, bg="#2b2b2b", highlightthickness=0)
        vsb = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        hsb = ttk.Scrollbar(container, orient="horizontal", command=canvas.xview)
        
        canvas.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
        
        # Внутренний фрейм для таблицы
        self.table_frame = ctk.CTkFrame(canvas, fg_color="#2b2b2b")
        canvas.create_window((0, 0), window=self.table_frame, anchor="nw")
        
        # Размещаем элементы
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")
        canvas.pack(side="left", fill="both", expand=True)
        
        # Создаем ячейки
        self.cells = []
        self.create_cells()
        
        # Настройка прокрутки
        self.table_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
    
    def create_cells(self):
        """Создает ячейки матрицы"""
        # Очищаем старые ячейки
        for widget in self.table_frame.winfo_children():
            widget.destroy()
        self.cells = []
        
        # Создаем заголовки столбцов
        for col in range(self.size + 1):
            for row in range(self.size + 1):
                if col == 0 and row == 0:
                    # Левый верхний угол
                    lbl = ctk.CTkLabel(self.table_frame, text="Партия/Этап", 
                                      width=100, height=30,
                                      font=("Arial", 10),
                                      fg_color="#3a3a3a", corner_radius=0)
                    lbl.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
                elif col == 0:
                    # Номера строк
                    lbl = ctk.CTkLabel(self.table_frame, text=f"#{row}", 
                                      width=50, height=30,
                                      font=("Arial", 10),
                                      fg_color="#3a3a3a", corner_radius=0)
                    lbl.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
                elif row == 0:
                    # Номера столбцов
                    lbl = ctk.CTkLabel(self.table_frame, text=f"{col}", 
                                      width=70, height=30,
                                      font=("Arial", 10),
                                      fg_color="#3a3a3a", corner_radius=0)
                    lbl.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
                else:
                    # Ячейки для ввода
                    entry = ctk.CTkEntry(self.table_frame, width=70, height=30,
                                        font=("Arial", 10), justify="center",
                                        placeholder_text="0.00")
                    entry.grid(row=row, column=col, padx=1, pady=1, sticky="nsew")
                    
                    # Сохраняем ссылку
                    if len(self.cells) <= row-1:
                        self.cells.append([])
                    self.cells[row-1].append(entry)
        
        # Если есть данные - заполняем
        if self.result_data:
            self.fill_from_data()
    
    def fill_from_data(self):
        """Заполняет таблицу из сохраненных данных"""
        if self.result_data and len(self.result_data) == self.size:
            for i in range(self.size):
                for j in range(self.size):
                    if i < len(self.cells) and j < len(self.cells[i]):
                        self.cells[i][j].delete(0, "end")
                        self.cells[i][j].insert(0, f"{self.result_data[i][j]:.3f}")
    
    def fill_default_15x15(self):
        """Заполняет таблицу 15×15 дефолтными значениями"""
        if self.size != 15:
            return  # Просто не заполняем для других размеров
        
        default_data = [
            [0.16, 0.20, 0.22, 0.25, 0.26, 0.27, 0.28, 0.25, 0.21, 0.19, 0.15, 0.10, 0.07, 0.04, 0.01],
            [0.18, 0.18, 0.21, 0.23, 0.26, 0.30, 0.33, 0.29, 0.24, 0.20, 0.18, 0.14, 0.11, 0.08, 0.04],
            [0.17, 0.18, 0.18, 0.18, 0.18, 0.19, 0.22, 0.18, 0.14, 0.10, 0.07, 0.05, 0.02, 0.00, 0.00],
            [0.15, 0.17, 0.19, 0.22, 0.25, 0.25, 0.23, 0.19, 0.17, 0.13, 0.10, 0.07, 0.04, 0.01, 0.00],
            [0.11, 0.11, 0.13, 0.14, 0.15, 0.16, 0.18, 0.13, 0.11, 0.09, 0.07, 0.04, 0.02, 0.00, 0.00],
            [0.16, 0.18, 0.20, 0.20, 0.23, 0.26, 0.26, 0.23, 0.19, 0.15, 0.11, 0.08, 0.06, 0.03, 0.00],
            [0.16, 0.16, 0.16, 0.17, 0.18, 0.18, 0.19, 0.17, 0.18, 0.14, 0.11, 0.09, 0.06, 0.03, 0.00],
            [0.10, 0.10, 0.12, 0.12, 0.13, 0.15, 0.15, 0.14, 0.13, 0.11, 0.10, 0.09, 0.07, 0.05, 0.00],
            [0.18, 0.18, 0.21, 0.23, 0.26, 0.26, 0.26, 0.22, 0.20, 0.18, 0.15, 0.13, 0.10, 0.07, 0.04],
            [0.16, 0.17, 0.18, 0.21, 0.23, 0.24, 0.26, 0.22, 0.20, 0.18, 0.13, 0.10, 0.07, 0.04, 0.01],
            [0.11, 0.13, 0.15, 0.15, 0.16, 0.17, 0.18, 0.16, 0.14, 0.12, 0.09, 0.06, 0.04, 0.01, 0.00],
            [0.13, 0.13, 0.13, 0.14, 0.15, 0.16, 0.19, 0.15, 0.13, 0.10, 0.07, 0.05, 0.02, 0.00, 0.00],
            [0.11, 0.13, 0.14, 0.14, 0.16, 0.17, 0.17, 0.15, 0.12, 0.09, 0.06, 0.03, 0.01, 0.00, 0.00],
            [0.15, 0.13, 0.13, 0.20, 0.21, 0.24, 0.26, 0.27, 0.26, 0.24, 0.22, 0.21, 0.19, 0.17, 0.14],
            [0.10, 0.11, 0.12, 0.13, 0.13, 0.14, 0.14, 0.12, 0.10, 0.09, 0.07, 0.05, 0.04, 0.01, 0.00]
        ]
        
        for i in range(self.size):
            for j in range(self.size):
                if i < len(self.cells) and j < len(self.cells[i]):
                    self.cells[i][j].delete(0, "end")
                    self.cells[i][j].insert(0, f"{default_data[i][j]:.3f}")
    
    def clear_matrix(self):
        """Очищает все ячейки матрицы"""
        for row in self.cells:
            for cell in row:
                cell.delete(0, "end")
    
    def change_size(self):
        """Изменяет размер матрицы"""
        try:
            new_size = int(self.size_var.get())
            if 1 <= new_size <= 30:
                self.size = new_size
                self.result_data = None  # Сбрасываем данные при изменении размера
                self.create_cells()
                # Автоматически заполняем дефолтом если 15×15
                if self.size == 15 and self.result_data is None:
                    self.fill_default_15x15()
            else:
                messagebox.showwarning("Ошибка", "Размер должен быть от 1 до 30")
        except ValueError:
            messagebox.showwarning("Ошибка", "Введите целое число")
    
    def save(self):
        """Сохраняет матрицу"""
        try:
            matrix = []
            for i in range(self.size):
                row = []
                for j in range(self.size):
                    value = self.cells[i][j].get().strip()
                    if value == "":
                        row.append(0.0)
                    else:
                        row.append(float(value.replace(',', '.')))
                matrix.append(row)
            
            self.result_data = matrix
            self.result_size = self.size
            self.destroy()
            
        except ValueError as e:
            messagebox.showerror("Ошибка ввода", f"Некорректные данные в матрице:\n{e}")
    
    def cancel(self):
        """Отменяет редактирование"""
        self.result_data = None
        self.destroy()

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

        self.btn_view_matrix = ctk.CTkButton(self.left_frame, text="ПОКАЗАТЬ МАТРИЦУ (последнюю)", 
                                             height=40, fg_color="#555", state="disabled", font=("Arial", 12, "bold"),
                                             command=self.open_matrix_window)
        self.btn_view_matrix.pack(padx=20, pady=(0, 20), fill="x")

        # --- ПРАВАЯ ПАНЕЛЬ ---
        self.right_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.right_frame.grid(row=0, column=1, sticky="nsew", padx=20, pady=20)
        self.right_frame.grid_columnconfigure((0, 1, 2, 3), weight=1) # Теперь 4 колонки
        self.right_frame.grid_rowconfigure(2, weight=1)

        # KPI
        self.card_ideal = InfoCard(self.right_frame, "Максимально возможный урожай", "---", color="#2ec4b6")
        self.card_ideal.grid(row=0, column=0, sticky="ew", padx=5, pady=(0, 10))
        
        # НОВАЯ КАРТОЧКА MIN
        self.card_min = InfoCard(self.right_frame, "Минимально возможный урожай", "---", color="#e63946")
        self.card_min.grid(row=0, column=1, sticky="ew", padx=5, pady=(0, 10))
        
        self.card_best = InfoCard(self.right_frame, "Лучшая стратегия", "---", color="#e76f51")
        self.card_best.grid(row=0, column=2, sticky="ew", padx=5, pady=(0, 10))
        self.card_loss = InfoCard(self.right_frame, "Минимальное отклонение", "--- %", color="#e9c46a")
        self.card_loss.grid(row=0, column=3, sticky="ew", padx=5, pady=(0, 10))

        # --- Панель управления графиками ---
        self.ctrl_frame = ctk.CTkFrame(self.right_frame, fg_color="transparent")
        self.ctrl_frame.grid(row=1, column=0, columnspan=4, sticky="ew", pady=(0, 10))
        
        self.lbl_slider = ctk.CTkLabel(self.ctrl_frame, text="Топ стратегий: 5", font=("Arial", 12))
        self.lbl_slider.pack(side="left", padx=(10, 10))
        
        self.slider_strat = ctk.CTkSlider(self.ctrl_frame, from_=1, to=10, number_of_steps=9, width=250, command=self.update_graph_view)
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
            unit = " тонн"
        
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
            
            if manual_mode:
                matrix, manual_nu, k_param, runs = self.manual_config.get_data()
                if matrix is None: raise ValueError("Матрица пуста!")
                if k_param is None:
                    k_param = 1
                n_rows = len(matrix)
                max_k = max(1, n_rows - manual_nu + 1) if manual_nu <= n_rows else 1
                if k_param > max_k:
                    k_param = max_k
                self.model.set_manual_matrix(matrix, manual_nu)
            else:
                p = self.auto_config.get_params()
                if p is None: 
                    # Ошибка уже показана в messagebox внутри get_params
                    # Сбрасываем UI и выходим
                    self.lbl_rec.configure(text="Исправьте параметры и запустите расчет снова.")
                    return
                
                self.model.n = p['n']; self.model.nu = p['nu']
                self.model.use_ripening = p['use_ripening']; self.model.use_chemistry = p['use_chemistry']
                self.model.distribution_type = p['distribution']; self.model.ranges = p['ranges']
                self.model.daily_mass = p['daily_mass']
                self.model.days_per_stage = p['days_per_stage']
                runs = p['runs']
                k_param = p.get('k_param', 1)

            stats, effective_runs = self.model.run_simulation(runs=runs, manual_mode=manual_mode, k_param = k_param)
            
            self.last_stats = stats
            self.last_runs = effective_runs

            # Анализ
            avg_ideal = np.mean(stats['Ideal']['totals'])
            avg_min = np.mean(stats['Min']['totals'])
            
            results = []
            for name in stats:
                if name in ['Ideal', 'Min']: continue
                val = np.mean(stats[name]['totals'])
                loss = (1 - val/avg_ideal) * 100 if avg_ideal != 0 else 0
                results.append((name, val, loss))
            results.sort(key=lambda x: x[2])
            best = results[0]
            
            self.update_kpi_cards_display(best[0], best[2], avg_ideal, avg_min)
            self.update_recommendation(best[0], best[2], manual_mode, k_param)
            self.draw_graphs(stats, effective_runs)
            
            self.btn_view_matrix.configure(state="normal", fg_color="#3a7ebf")
            
        except Exception as e:
            self.lbl_rec.configure(text=f"ОШИБКА: {e}")
            import traceback; traceback.print_exc()
        finally:
            self.btn_run.configure(text="ЗАПУСТИТЬ РАСЧЕТ", state="normal")

    def update_recommendation(self, name, loss, manual_mode, k_param=1):
        recommendation_text = f"РЕКОМЕНДУЕМАЯ СТРАТЕГИЯ: {name}\n"
        recommendation_text += f"Отклонение от теоретического максимума: {loss:.2f}%\n\n"
        
        # Лаконичные, но содержательные выводы
        analysis_dict = {
            "Жадная": (
                "Данные показывают преобладание процессов увядания.\n"
                "Оптимальна стратегия немедленной переработки сырья с максимальной сахаристостью."
            ),
            
            "Бережливая": (
                "Наблюдается выраженный эффект дозаривания.\n"
                "Рекомендуется отложенная переработка для накопления потенциала."
            ),
            
            "Бережливая/Жадная": (
                "Процесс демонстрирует двухфазную динамику.\n"
                "Эффективна стратегия накопления с последующим активным сбором."
            ),
            
            "Жадная/Бережливая": (
                "Начальное качество имеет критическое значение.\n"
                "Первоочередная переработка лучшего сырья экономически оправдана."
            ),
            
            "БkЖ (T(k)G)": (
                "Требуется балансировка между различными подходами.\n"
                "Стратегия с промежуточным выбором обеспечивает оптимальный компромисс."
            ),
            
            "CTG (Сортировка по лежкости)": (
                "Ключевой фактор - вариабельность сохранности партий.\n"
                "Приоритет должен отдаваться партиям с наихудшей лежкостью."
            ),
            
            "Критической деградации": (
                "Выявлены партии с особыми характеристиками сохранности.\n"
                "Эффективен подход, учитывающий как текущее качество, так и скорость деградации."
            ),
            
            "Среднее+Отклонение": (
                "Качество сырья имеет выраженную неоднородность.\n"
                "Концентрация на лучшей части партий максимизирует выход продукции."
            ),
            
            "Фазовая группировка": (
                "Динамика процесса изменяется в течение сезона.\n"
                "Адаптивная стратегия с разными подходами на разных этапах оптимальна."
            )
        }
        
        # Практические указания
        if name in analysis_dict:
            recommendation_text += "АНАЛИЗ:\n" + analysis_dict[name] + "\n\n"
             
        self.lbl_rec.configure(text=recommendation_text)

    def draw_graphs(self, stats, runs):
        top_n = int(self.slider_strat.get())
        use_real = bool(self.sw_real_view.get())

        scale_y = (self.model.daily_mass * self.model.days_per_stage) if use_real else 1.0
        scale_x = self.model.days_per_stage if use_real else 1.0
        
        # Обновляем KPI при смене тумблера
        if self.last_stats:
             avg_ideal = np.mean(stats['Ideal']['totals'])
             avg_min = np.mean(stats['Min']['totals'])
             results = []
             for name in stats:
                if name in ['Ideal', 'Min']: continue
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
        ax1.plot(x_vals, y_ideal, 'w--', label='Максимум', alpha=0.5)

        y_min = np.cumsum(stats['Min']['dynamics_sum']/runs) * scale_y
        ax1.plot(x_vals, y_min, 'r--', label='Минимум', alpha=0.5, linewidth=2)
        
        # Сортируем стратегии (исключая Ideal и Min)
        strategy_names = [k for k in stats if k not in ['Ideal', 'Min']]
        sorted_keys = sorted(strategy_names, key=lambda k: np.mean(stats[k]['totals']), reverse=True)
    
        # Показываем топ стратегий
        if top_n >= len(sorted_keys):
            top_keys = sorted_keys  # Все стратегии
        else:
            top_keys = sorted_keys[:top_n]
        
        colors = ['#e76f51', '#2a9d8f', '#e9c46a', '#f4a261', '#81b29a', '#f1faee', '#a8dadc', '#457b9d']
        for i, name in enumerate(top_keys):
            col = colors[i % len(colors)]
            y_vals = np.cumsum(stats[name]['dynamics_sum']/runs) * scale_y
            ax1.plot(x_vals, y_vals, color=col, label=name, linewidth=2)
            
        ax1.grid(True, linestyle='--', alpha=0.3)
        ax1.legend(facecolor='#2b2b2b', labelcolor='white')
        ax1.tick_params(colors='white'); [s.set_color('white') for s in ax1.spines.values()]
        
        x_label = "Дни" if use_real else "Этапы (Столбцы Матрицы)"
        y_label = "Совокупный урожай (тонны)" if use_real else "Совокупный урожай (дробные единицы)"
        ax1.set_xlabel(x_label, color='white', fontsize=9)
        ax1.set_ylabel(y_label, color='white', fontsize=9)
        
        self.canvas_line = FigureCanvasTkAgg(fig1, master=self.frame_line)
        self.canvas_line.draw()
        
        self.toolbar_line = NavigationToolbar2Tk(self.canvas_line, self.frame_line)
        self.toolbar_line.update()
        self.canvas_line.get_tk_widget().pack(fill="both", expand=True)
        
        # 2. Bar Chart
        fig2 = Figure(figsize=(6, 4), dpi=100)
        fig2.patch.set_facecolor('#2b2b2b')
        ax2 = fig2.add_subplot(111)
        ax2.set_facecolor('#2b2b2b')

        # Создаем правильные списки names и vals (ДОБАВЬТЕ ЭТИ СТРОЧКИ)
        names = ['Максимум', 'Минимум'] + top_keys
        vals = [np.mean(stats['Ideal']['totals']), np.mean(stats['Min']['totals'])] + [np.mean(stats[k]['totals']) for k in top_keys]
        vals = [v * scale_y for v in vals]

        # Создаем правильный список цветов
        colors_bar = ['#2ec4b6', '#e63946'] + ['#457b9d']*len(top_keys)  # Зеленый, красный, синие
        bars = ax2.bar(names, vals, color=colors_bar, alpha=0.9)
        ax2.tick_params(colors='white', axis='x', labelsize=8)
        ax2.tick_params(colors='white', axis='y', labelsize=8)
        # УВЕЛИЧИВАЕМ ВЫСОТУ РАМКИ - добавьте эти строки:
        current_ymax = ax2.get_ylim()[1]  # Текущая максимальная высота графика
        ax2.set_ylim(top=current_ymax * 1.2)  # Увеличить верхнюю границу на 20%

        [s.set_color('white') for s in ax2.spines.values()]

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