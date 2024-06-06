import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.integrate import solve_ivp
from instruction import instrucrion_text

class MainApplication(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Визуализатор фазовых портретов")
        self.geometry("1200x900")

        self.create_widgets()
        self.figures = []

    def create_widgets(self):
        self.create_menu()
        self.create_equation_label()
        self.create_input_fields()
        self.create_buttons()
        self.create_plot_settings()
        
        self.figure, self.ax = plt.subplots()
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        self.figure_canvas.get_tk_widget().grid(row=1, column=3, rowspan=10, padx=10, pady=10)

        self.figure_canvas.mpl_connect("button_press_event", self.on_click)

    def create_menu(self):
        menu_bar = tk.Menu(self)
        file_menu = tk.Menu(menu_bar, tearoff=0)
        file_menu.add_command(label="Сохранить как EPS", command=self.save_as_eps)
        file_menu.add_separator()
        file_menu.add_command(label="Выход", command=self.quit)
        menu_bar.add_cascade(label="Файл", menu=file_menu)
        
        help_menu = tk.Menu(menu_bar, tearoff=0)
        help_menu.add_command(label="Руководство пользователя", command=self.show_help)
        menu_bar.add_cascade(label="Помощь", menu=help_menu)
        
        self.config(menu=menu_bar)

    def create_input_fields(self):
        tk.Label(self, text="Параметры уравнения").grid(row=1, column=0, columnspan=2) 
        tk.Label(self, text="alpha (α)").grid(row=2, column=0)
        tk.Label(self, text="beta (β)").grid(row=3, column=0)
        tk.Label(self, text="delta (δ)").grid(row=4, column=0)
        tk.Label(self, text="gamma (γ)").grid(row=5, column=0)

        self.alpha_entry = tk.Entry(self)
        self.alpha_entry.grid(row=2, column=1)
        self.alpha_entry.insert(0, "0.1")
        self.beta_entry = tk.Entry(self)
        self.beta_entry.grid(row=3, column=1)
        self.beta_entry.insert(0, "0.5")
        self.delta_entry = tk.Entry(self)
        self.delta_entry.grid(row=4, column=1)
        self.delta_entry.insert(0, "0.02")
        self.gamma_entry = tk.Entry(self)
        self.gamma_entry.grid(row=5, column=1)
        self.gamma_entry.insert(0, "0.9")

        tk.Label(self, text="Начальные условия").grid(row=6, column=0, columnspan=2)
        tk.Label(self, text="x0 (нач. значение x)").grid(row=7, column=0)
        tk.Label(self, text="y0 (нач. значение y)").grid(row=8, column=0)

        self.x0_entry = tk.Entry(self)
        self.x0_entry.grid(row=7, column=1)
        self.x0_entry.insert(0, "1.0")
        self.y0_entry = tk.Entry(self)
        self.y0_entry.grid(row=8, column=1)
        self.y0_entry.insert(0, "0.0")

        tk.Label(self, text="Настройки счета").grid(row=9, column=0, columnspan=2)
        tk.Label(self, text="t_max (макс. время)").grid(row=10, column=0)
        tk.Label(self, text="atol (абс. погрешность)").grid(row=11, column=0)
        tk.Label(self, text="rtol (отн. погрешность)").grid(row=12, column=0)
        tk.Label(self, text="h (шаг)").grid(row=13, column=0)

        self.t_max_entry = tk.Entry(self)
        self.t_max_entry.grid(row=10, column=1)
        self.t_max_entry.insert(0, "50")
        self.atol_entry = tk.Entry(self)
        self.atol_entry.grid(row=11, column=1)
        self.atol_entry.insert(0, "1e-6")
        self.rtol_entry = tk.Entry(self)
        self.rtol_entry.grid(row=12, column=1)
        self.rtol_entry.insert(0, "1e-3")
        self.h_entry = tk.Entry(self)
        self.h_entry.grid(row=13, column=1)
        self.h_entry.insert(0, "0.01")

        self.method_var = tk.StringVar(self)
        self.method_var.set("RK45")
        tk.Label(self, text="Метод").grid(row=14, column=0)
        self.method_menu = tk.OptionMenu(self, self.method_var, "RK45", "Euler", "Adams")
        self.method_menu.grid(row=14, column=1)
            
        self.backward_var = tk.BooleanVar()
        tk.Checkbutton(self, text="Обратное время", variable=self.backward_var).grid(row=15, column=0, columnspan=2)

    def create_buttons(self):
        self.solve_button = tk.Button(self, text="Построить траекторию", command=self.solve_and_plot)
        self.solve_button.grid(row=16, column=0, columnspan=2)
        self.clear_button = tk.Button(self, text="Очистить экран", command=self.clear_screen)
        self.clear_button.grid(row=17, column=0)
        self.undo_button = tk.Button(self, text="Отменить последнее построение", command=self.undo_last_plot)
        self.undo_button.grid(row=19, column=0, columnspan=2)

    def create_plot_settings(self):
        tk.Label(self, text="Настройки построения").grid(row=20, column=0, columnspan=2)
        tk.Label(self, text="Цвет траектории").grid(row=21, column=0)
        tk.Label(self, text="Толщина линии").grid(row=22, column=0)
        tk.Label(self, text="Ширина графика").grid(row=23, column=0)
        tk.Label(self, text="Высота графика").grid(row=24, column=0)

        self.color_var = tk.StringVar(self)
        self.color_var.set("red")
        self.line_width_var = tk.DoubleVar(self)
        self.line_width_var.set(1.5)
        self.width_var = tk.DoubleVar(self)
        self.width_var.set(5)
        self.height_var = tk.DoubleVar(self)
        self.height_var.set(4)
        
        self.color_entry = tk.Entry(self, textvariable=self.color_var)
        self.color_entry.grid(row=21, column=1)
        self.line_width_entry = tk.Entry(self, textvariable=self.line_width_var)
        self.line_width_entry.grid(row=22, column=1)
        self.width_entry = tk.Entry(self, textvariable=self.width_var)
        self.width_entry.grid(row=23, column=1)
        self.height_entry = tk.Entry(self, textvariable=self.height_var)
        self.height_entry.grid(row=24, column=1)

    def create_equation_label(self):
        equation_text = "Решаемое уравнение: ẍ = γx - βx³ - δẋ - αx²ẋ"
        self.equation_label = tk.Label(self, text=equation_text, wraplength=400)
        self.equation_label.grid(row=0, column=0, columnspan=4)

    def show_help(self):
        messagebox.showinfo("Руководство пользователя", instrucrion_text)

    def solve_and_plot(self):
        try:
            alpha = float(self.alpha_entry.get())
            beta = float(self.beta_entry.get())
            delta = float(self.delta_entry.get())
            gamma = float(self.gamma_entry.get())
            x0 = float(self.x0_entry.get())
            y0 = float(self.y0_entry.get())
            t_max = float(self.t_max_entry.get())
            atol = float(self.atol_entry.get())
            rtol = float(self.rtol_entry.get())
            h = float(self.h_entry.get())
            method = self.method_var.get()
            backward = self.backward_var.get()
            
            params = [alpha, beta, delta, gamma, x0, y0, t_max, atol, rtol, h]
            sol = self.solve_system(method, params, backward)

            fig, ax = plt.subplots(figsize=(self.width_var.get(), self.height_var.get()))
            ax.plot(sol.y[0], sol.y[1], color=self.color_var.get(), linewidth=self.line_width_var.get())
            ax.set_xlabel('x')
            ax.set_ylabel('ẋ')
            ax.grid(True)
            self.figures.append(fig)
            self.display_figure(fig)
        except Exception as e:
            messagebox.showerror("Ошибка", str(e))

    def solve_system(self, method, params, backward):
        alpha, beta, delta, gamma, x0, y0, t_max, atol, rtol, h = params

        def system(t, z):
            x, y = z
            dxdt = y
            dydt = gamma * x - beta * x**3 - delta * y - alpha * x**2 * y
            return [dxdt, dydt]

        y0 = [x0, y0]
        
        if backward:
            t_span = (0, -t_max)
            t_eval = np.arange(0, -t_max, -h)
        else:
            t_span = (0, t_max)
            t_eval = np.arange(0, t_max, h)

        sol = solve_ivp(system, t_span, y0, method=method, t_eval=t_eval, atol=atol, rtol=rtol)
        return sol

    def display_figure(self, fig):
        for widget in self.grid_slaves():
            if int(widget.grid_info()["column"]) > 2:
                widget.grid_forget()
        self.figure_canvas = FigureCanvasTkAgg(fig, self)
        self.figure_canvas.get_tk_widget().grid(row=1, column=3, rowspan=10, padx=10, pady=10)
        self.figure_canvas.draw()

        self.figure_canvas.mpl_connect("button_press_event", self.on_click)

    def clear_screen(self):
        for widget in self.grid_slaves():
            if int(widget.grid_info()["column"]) > 2:
                widget.grid_forget()
        self.figure, self.ax = plt.subplots(figsize=(self.width_var.get(), self.height_var.get()))
        self.figure_canvas = FigureCanvasTkAgg(self.figure, self)
        self.figure_canvas.get_tk_widget().grid(row=1, column=3, rowspan=10, padx=10, pady=10)

        self.figure_canvas.mpl_connect("button_press_event", self.on_click)

    def save_as_eps(self):
        file_path = filedialog.asksaveasfilename(defaultextension='.eps', filetypes=[('EPS files', '*.eps')])
        if file_path:
            self.figures[-1].savefig(file_path, format='eps')

    def undo_last_plot(self):
        if self.figures:
            self.figures.pop()
            self.clear_screen()
            for fig in self.figures:
                self.display_figure(fig)

    def on_click(self, event):
        if event.xdata is not None and event.ydata is not None:
            x0, y0 = event.xdata, event.ydata
            self.x0_entry.delete(0, tk.END)
            self.x0_entry.insert(0, str(x0))
            self.y0_entry.delete(0, tk.END)
            self.y0_entry.insert(0, str(y0))

if __name__ == "__main__":
    app = MainApplication()
    app.mainloop()
