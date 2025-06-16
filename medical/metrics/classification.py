import itertools
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve 
from sklearn.metrics import roc_auc_score as AUC

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Purples):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

def confusion_matrixs(y_test, y_pred, savedir):
    cnf_matrix = confusion_matrix(y_test, y_pred)
    ## 限制两位小数
    np.set_printoptions(precision=2)
    plt.figure()
    plot_confusion_matrix(cnf_matrix, classes=[0, 1], title='Confusion matrix, without normalization')
    plt.savefig(savedir+"/confusion_matrix.png", dpi=300)
    
def AUC_curve(y_test, y_pred, savedir):
    FPR, recall, thresholds = roc_curve(y_test, y_pred)
    # 计算AUC面积
    area = AUC(y_test, y_pred)

    # 画图
    plt.figure()
    plt.plot(FPR,recall,label='ROC curve (area = %0.2f)' % area)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(savedir+"/auc_curve.png", dpi=300)
