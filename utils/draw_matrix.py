from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

from utils.params import LABELS_MAP

classes = list(LABELS_MAP.values())

def draw_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    cmap=plt.cm.Blues
    plt.figure(figsize=(8, 8))
    
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    tick_marks = np.arange(len(classes))
    
    plt.title('Confusion Matrix', fontsize=15)
    plt.xlabel('Predicted Category', fontsize=12)
    plt.ylabel('Target Category', fontsize=12)
    plt.tick_params(labelsize=12) 
    plt.xticks(tick_marks, classes, rotation=90) 
    plt.yticks(tick_marks, classes)
    
    threshold = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > threshold else "black",
                 fontsize=12)

    plt.tight_layout()

    plt.savefig('confusion_matrix.png')
    plt.show()