import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk

# Активационная функция
def f_activ(x):
    return 1 / (1 + np.exp(-x))

# Производная активационной функции
def df_activ(x):
    return x * (1 - x)

# Классификация
def classification(x, y, metka, eta=0.01, epochs=10000):
    np.random.seed(0)  # Для воспроизводимости
    w3 = np.random.rand(3, 2)
    w2 = np.random.rand(2, 3)
    w1 = np.random.rand(2)

    min_mark = float('inf')
    best_w3, best_w2, best_w1 = None, None, None

    for e in range(epochs):
        mark = 0
        for i in range(len(x)):
            v3 = np.dot(w3, [x[i], y[i]])
            f3 = f_activ(v3)

            v2 = np.dot(w2, f3)
            f2 = f_activ(v2)

            v1 = np.dot(w1, f2)
            f1 = f_activ(v1)

            mark += abs(f1 - metka[i])

            # Обратное распространение
            b = (f1 - metka[i]) * df_activ(f1)
            w1 -= eta * b * f2

            b1 = b * w1 * df_activ(f2)
            w2 -= eta * np.outer(b1, f3)

            for j in range(3):
                b2 = df_activ(f3[j]) * np.dot(b1, w2[:, j])
                w3[j] -= eta * b2 * np.array([x[i], y[i]])

        if mark < min_mark and mark > 0:
            min_mark = mark
            best_w3, best_w2, best_w1 = w3.copy(), w2.copy(), w1.copy()
        if min_mark < 10:
            break

    w3, w2, w1 = best_w3, best_w2, best_w1

    # Прогноз
    result = []
    for i in range(len(x)):
        v3 = np.dot(w3, [x[i], y[i]])
        f3 = f_activ(v3)

        v2 = np.dot(w2, f3)
        f2 = f_activ(v2)

        v1 = np.dot(w1, f2)
        f1 = f_activ(v1)

        result.append(1 if f1 > 0.5 else 0)

    return result

# Регрессия
def regress(y, eta=0.001, max_iter=100000):
    w = np.array([0.0, 0.0, 0.0])
    best_w = w.copy()
    min_loss = float('inf')

    for _ in range(max_iter):
        total_loss = 0
        for i in range(len(y)):
            delta = y[i] - (w[1] * i + w[0])
            d = 1 if delta > 0 else -1
            w[0] += eta * d
            w[1] += eta * d * i
            total_loss += delta ** 2

        if total_loss < min_loss:
            min_loss = total_loss
            best_w = w.copy()

    return best_w

# Генерация случайных данных для классификации
def random_liner_yes(number):
    np.random.seed(0)
    x = np.random.randint(1, 100, number)
    y = np.random.randint(1, 100, number)

    metka = []
    for i in range(number):
        y_true = 0.2 * x[i] ** 2 - 20 * x[i] + 500
        metka.append(1 if y[i] > y_true else 0)

    return np.array(x), np.array(y), np.array(metka)

# Генерация данных для регрессии
def random_regress(number):
    np.random.seed(0)
    return 0.2 * np.arange(number) ** 2 - 20 * np.arange(number) + 500 + np.random.normal(0, 10, number)

# Функция для рисования графиков классификации
def paint_classification(canvas, figure, x, y, metka):
    figure.clear()
    ax = figure.add_subplot(111)
    for i in range(len(x)):
        color = 'red' if metka[i] == 1 else 'blue'
        ax.scatter(x[i], y[i], color=color)

    x_line = np.linspace(0, 100, 100)
    y_line = 0.2 * x_line ** 2 - 20 * x_line + 500
    ax.plot(x_line, y_line, color='black')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Классификация')
    canvas.draw()

# Функция для рисования графиков регрессии
def paint_regression(canvas, figure, y, a=0, b=0):
    figure.clear()
    ax = figure.add_subplot(111)
    ax.scatter(range(len(y)), y, color='red', label='Дата')
    x_line = np.arange(len(y))
    y_line = a * x_line + b
    ax.plot(x_line, y_line, color='black', label=f'Линия: {a:.2f}x + {b:.2f}')

    ax.set_xlabel('Индекс')
    ax.set_ylabel('Y')
    ax.set_title('Регресс')
    ax.legend()
    canvas.draw()

# Основной код для tkinter
def main():
    # Создаем окно
    root = tk.Tk()
    root.title("lab2")

    # Создаем область для графиков
    figure = plt.Figure(figsize=(6, 5), dpi=100)
    canvas = FigureCanvasTkAgg(figure, root)
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # Кнопка для классификации
    def on_classification():
        x, y, metka = random_liner_yes(100)
        classified_metka = classification(x, y, metka)
        paint_classification(canvas, figure, x, y, classified_metka)

    # Кнопка для регрессии
    def on_regression():
        y = random_regress(50)
        w = regress(y)
        paint_regression(canvas, figure, y, w[1], w[0])

    # Создаем кнопки
    btn_classification = tk.Button(root, text="Классификация", command=on_classification)
    btn_classification.pack(side=tk.LEFT, padx=5, pady=5)

    btn_regression = tk.Button(root, text="Регресс", command=on_regression)
    btn_regression.pack(side=tk.LEFT, padx=5, pady=5)

    # Запуск основного цикла приложения
    root.mainloop()

if __name__ == "__main__":
    main()
