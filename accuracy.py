from pandas_ml import ConfusionMatrix
import matplotlib.pyplot as plt

"""
    Find the relevant data-sets at:
        https://drive.google.com/drive/folders/1LGc_i2EyXArU_XBepYzPTIq141URtBl0?usp=sharing
"""


def get_accuracy(results, array):
    act_arr = array
    predicted_arr = results
    confusion_matrix = ConfusionMatrix(act_arr, predicted_arr)

    print("Confusion matrix:\n%s" % confusion_matrix)
    confusion_matrix.plot()
    plt.show()
    print("\n")
    a = perf_measure(act_arr, predicted_arr)
    b = close_measure(act_arr, predicted_arr)
    print("The Accuracy is: " + str(a * 100) + "%")
    print("The close range Accuracy is: " + str(b * 100) + "%")


def perf_measure(y_actual, y_predict):
    correct = 0
    wrong = 0
    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i]:
            correct += 1
        else:
            wrong += 1
    a = correct / (correct + wrong)
    return a


def close_measure(y_actual, y_predict):
    correct = 0
    wro = 0
    for i in range(len(y_predict)):
        if y_actual[i] == y_predict[i]:
            correct += 1
        elif y_actual[i] == y_predict[i] + 1 & y_actual[i] != 5:
            correct += 1
        elif y_actual[i] == y_predict[i] - 1 & y_actual[i] != 1:
            correct += 1
        else:
            wro += 1
    a = correct / (correct + wro)
    return a
