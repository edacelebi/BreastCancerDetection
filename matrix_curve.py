import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

class matrix_curve:

    def plot_accuracy_loss(self, history):
        fig = plt.figure(figsize=(10, 5))

        plt.subplot(221)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title("model accuracy")
        plt.ylabel("accuracy")
        plt.xlabel("epochs")
        plt.legend(['train', 'test'], loc='upper left')

        plt.subplot(222)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.plot(np.argmin(history.history["val_loss"]), np.min(history.history["val_loss"]), marker="x", color="r",
                 label="best model")
        plt.title("model loss")
        plt.ylabel("loss")
        plt.xlabel("epochs")
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

    def plot_cofusion_matrix(self, cnf):
        info = [
            'benign',     # 0
            'malignant',  # 1
            'normal',     # 2

        ]
        plt.figure(figsize=(15, 15))
        ax = sns.heatmap(cnf, annot=True, square=True, xticklabels=info, yticklabels=info)
        ax.set_ylabel('Actual', fontsize=40)
        ax.set_xlabel('Predicted', fontsize=40)

        plt.show()

