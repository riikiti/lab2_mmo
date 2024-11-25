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
def error(a, b):
    return (a-b)*(a-b)
class NeuralNetwork:
    def __init__(self, inp_count=2, w1_count=3, out_count=1):

        self.w1 = np.random.rand(w1_count, inp_count)
        self.b1 = np.random.rand(w1_count)
        self.w_out = np.random.rand(out_count, w1_count)
        self.b_out = np.random.rand(out_count)

    def feedforward(self, XY):
        self.v1 = np.dot(self.w1, XY) + self.b1
        self.v1_out = f_activation(self.v1)

        self.v_out = np.dot(self.w_out, self.v1_out) + self.b_out
        self.output = f_activation(self.v_out)
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

    def train(self, XYs, Ys_true, XY_tests, C_test, learn_rate = 0.1, epochs = 1600):
        datasets = XYs.shape[0]
        loss = [[0] * epochs]
        for epoch in range(epochs):
            for counter in range(datasets):
                self.backprop(XYs[counter], Ys_true[counter], learn_rate)

            predict = np.apply_along_axis(self.feedforward, 1, XYs)
            loss[0][epoch] = sq_error(Ys_true, predict.T[0])
        predict = np.apply_along_axis(self.feedforward, 1, XY_tests)
        loss_test = sq_error(C_test, predict.T[0])
        return (loss, epoch, loss_test)


def classifier(X):
    f = sepfunction(X[0])
    if (f - X[1] > 0):
        return 1
    return 0


def sepfunction(x):
    return x*x*x

if __name__ == "__main__":
    X_train = np.zeros([50, 1])
    X_train.T[0] = np.linspace(0.0, 1.0, 50)
    Y_train = np.apply_along_axis(sepfunction, 1, X_train) + np.random.normal(0.0, 0.10, size=(50, 1))

    X_test = np.zeros([100, 1])
    X_test.T[0] = np.linspace(0, 1, 100)
    Y_test = np.apply_along_axis(sepfunction, 1, X_test)
    nw = NeuralNetwork(inp_count=1)
    (loss, epoch, loss_test) = nw.train(X_train, Y_train, X_test, Y_test)
    nw_res = np.apply_along_axis(nw.feedforward, 1, X_test)
    print(f"РћС€РёР±РєР° РЅР° РѕР±СѓС‡Р°Р±С‰РµР№ РІС‹Р±РѕСЂРєРµ= {loss[0][epoch]}.")
    print(f"РћС€РёР±РєР° РЅР° С‚РµСЃС‚РѕРІРѕР№ РІС‹Р±РѕСЂРєРµ= {loss_test}.")

    plt.suptitle("Р РµРіСЂРµСЃСЃРёСЏ")
    plt.subplot(1, 2, 1)
    plt.scatter(X_train, Y_train, label="A")
    plt.plot(X_test.T[0], Y_test.T[0], label="B")  # draw separation line
    plt.plot(X_test.T[0], nw_res.T[0], label="C")
    plt.title("Р РµРіСЂРµСЃСЃРёСЏ")

    plt.subplot(1, 2, 2)
    # РѕС€РёР±РєРё РѕС‚СЂРёСЃРѕРІС‹РІР°РµРј
    plt.plot(loss[0][:epoch])
    plt.title("РР·РјРµРЅРµРЅРёРµ РѕС€РёР±РєРё")
    plt.xlim(0, epoch)
    plt.show()