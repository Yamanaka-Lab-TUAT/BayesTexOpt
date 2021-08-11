import matplotlib.pyplot as plt
from dataset import load_data

# from tf_models.dnn2d.model import NetWork
from tf_models.dnn3d.model import NetWork


def learning_curve(hist):
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 10
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['ytick.direction'] = 'in'
    plt.rcParams['axes.linewidth'] = 1.0
    plt.rcParams['xtick.major.width'] = 1.0
    plt.rcParams['ytick.major.width'] = 1.0
    plt.rcParams['legend.fancybox'] = False
    plt.rcParams['legend.framealpha'] = 1
    plt.rcParams['legend.edgecolor'] = 'black'
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 4), dpi=200)

    loss = hist.history['loss']
    val_loss = hist.history['val_loss']
    epochs = range(len(loss))
    ax.plot(epochs, loss, 'b-', label='training loss')
    ax.plot(epochs, val_loss, 'b--', label='validation loss')

    ax.set_xlim([0., None])
    ax.set_ylim([0, None])
    ax.set_title('Training and Validation loss')
    plt.legend()
    plt.show()


def main():
    (x_train, y_train), (x_test, y_test) = \
        load_data(input_data_type=1, shuffle=True, use_cache=False)
    n = NetWork().build()
    n.training(x_train, y_train, x_test, y_test)
    n.saveHistory('history')
    n.saveParam()
    learning_curve(n.hist)


if __name__ == "__main__":
    main()
