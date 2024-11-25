import numpy as np
import matplotlib.pyplot as plt


def f_activation(x):
    return 1 / (1 + np.exp(-x))


# Derivative of f_activation
def df_activation(f):
    return f * (1 - f)


def sq_error(expect, actual, e=0):
    datasets = expect.shape[0]
    for counter in range(datasets):
        e+=(expect[counter]-actual[counter])*(expect[counter]-actual[counter])
    return e

class NeuralNetwork:
    def __init__(self, inp_count=2, w1_count=2, out_count=1):

        self.w1 = np.random.rand(w1_count, inp_count)
        self.b1 = np.random.rand(w1_count)
        self.w_out = np.random.rand(out_count, w1_count)
        self.b_out = np.random.rand(out_count)

    def feedforward(self, XY):
        self.v1 = np.dot(self.w1, XY) + self.b1
        self.v1_out = f_activation(self.v1);

        self.v_out = np.dot(self.w_out, self.v1_out) + self.b_out
        self.output = f_activation(self.v_out);
        return self.output

    def backprop(self, XY, true_out, learn_rate):
        self.feedforward(XY)
        out_corr = true_out - self.output #РѕС€РёР±РєР°

        ds = df_activation(self.output)
        sigm_out = learn_rate * out_corr * ds

        sigm_w1 = df_activation(self.v1_out) * \
                  np.dot(sigm_out, self.w_out)


        self.w_out += np.tensordot(
            sigm_out, self.v1_out, axes=0)
        self.b_out += sigm_out
        self.w1 += np.tensordot(sigm_w1, XY, axes=0)
        self.b1 += sigm_w1

    def train(self, XYs, Ys_true, XY_tests, C_test, learn_rate=0.1, epochs=1600):
        datasets = XYs.shape[0];
        loss = [[0] * epochs]
        for epoch in range(epochs):
            print(f'# Epoch: {epoch + 1}', end='\r')
            for counter in range(datasets):
                self.backprop(XYs[counter], Ys_true[counter], learn_rate)

            predict = np.apply_along_axis(self.feedforward, 1, XYs)
            loss[0][epoch] = sq_error(Ys_true, predict.T[0])
        datasets = XY_tests.shape[0];
        for counter in range(datasets):
            predict = np.apply_along_axis(self.feedforward, 1, XY_tests)
            loss_test = sq_error(C_test, predict.T[0])
        return (loss, epoch, loss_test)


def classifier(X):
    f = sepfunction(X[0])
    if (f - X[1] > 0):
        return 1
    return 0


def sepfunction(x):
    return 4 * x * x - 3 * x + 1  # РєРІР°РґСЂР°С‚РёС‡РЅР°СЏ


def generate_classified_data(sz):
    XY = np.random.rand(sz, 2)
    Ans = np.apply_along_axis(classifier, 1, XY)
    return (XY, Ans)


def sort_points(Points, Point_classes):
    # РљР»Р°СЃСЃРёС„РёС†РёСЂСѓРµРј РґР°РЅРЅС‹Рµ РІСЂСѓС‡РЅСѓСЋ
    ai = np.where(Point_classes > .5)
    bi = np.where(Point_classes <= .5)

    Ax = Points.T[0][ai]
    Ay = Points.T[1][ai]
    Bx = Points.T[0][bi]
    By = Points.T[1][bi]
    return (Ax, Ay, Bx, By)


if __name__ == "__main__":

    (XY_train, C_train) = generate_classified_data(100)
    (XY_test, C_test) = generate_classified_data(100)
    X_sepline = np.linspace(0, 1, 200)
    Y_sepline = np.apply_along_axis(sepfunction, 0, X_sepline)


    nw = NeuralNetwork()
    (loss, epoch, loss_test) = nw.train(XY_train, C_train, XY_test, C_test)
    print(f"Ошибка на обучающей выборке = {loss[0][epoch]}.")
    print(f"Ошибка на тестовой выборке = {loss_test}.")


    nw_res = np.apply_along_axis(nw.feedforward, 1, XY_test)
    (Ax, Ay, Bx, By) = sort_points(XY_train, C_train)

    plt.suptitle("Бинарная классификация")
    plt.subplot(1, 3, 1)
    plt.scatter(Ax, Ay, label='A')
    plt.scatter(Bx, By, label='B')
    plt.plot(X_sepline, Y_sepline)  # draw separation line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Обучающая выборка")


    (Ax, Ay, Bx, By) = sort_points(XY_test, nw_res.T[0])
    plt.subplot(1, 3, 2)
    plt.scatter(Ax, Ay, label='A')
    plt.scatter(Bx, By, label='B')
    plt.plot(X_sepline, Y_sepline)  # draw separation line
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.title("Тестовая выборка")
    plt.subplot(1, 3, 3)

    # РѕС€РёР±РєРё РѕС‚СЂРёСЃРѕРІС‹РІР°РµРј
    plt.plot(loss[0][:epoch])
    plt.title("Изменение ошибки")
    plt.xlim(0, epoch)
    plt.show()