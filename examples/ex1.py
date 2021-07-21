import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from PMKLpy import PMKL
from myplots import *

X, y = PMKL.loadex1()


C = 1.0 

SVM = PMKL.PMKL( C=C, to_print = False) 
SVM.fit(X, y)
# yPred = SVM.predict(X)
# prediction = np.round(yPred)

models = [svm.SVC(kernel='linear', C=C), 
          svm.SVC(kernel='rbf', gamma=0.7, C=C),
          svm.SVC(kernel='poly', degree=3, gamma='auto', C=C)]

models = [clf.fit(X, y) for clf in models]
models.append(SVM)

# title for the plots
titles = ['SVC with linear kernel', 
          'SVC with RBF kernel',
          'SVC with polynomial (degree 3) kernel',
            'SVC with TKL kernel']

fig, sub = plt.subplots(1, 4, figsize = (25, 5))
plt.subplots_adjust(wspace=0.4, hspace=0.4)

X0, X1 = X[:, 0], X[:, 1]
xx, yy = make_meshgrid(X0, X1)

for clf, title, ax in zip(models, titles, sub.flatten()):
    plot_contour_and_points(ax, clf, xx, yy,  X0, X1, y, title, cmap=plt.cm.coolwarm, alpha=0.8)
    
plt.show()