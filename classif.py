import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn.apionly as sns
from sklearn import datasets, linear_model
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import numpy as np 
import pylab as pl
from sklearn import svm, datasets

iris = sns.load_dataset('iris')


#metodo dos minimos quadrados
dado_ajuste = iris[["petal_length", "petal_width"]].values
dado_x = dado_ajuste[:,0].reshape(-1,1)
dado_y = dado_ajuste[:,1].reshape(-1,1) 

reg = linear_model.LinearRegression()

reg.fit(dado_x, dado_y)
 

fig = plt.figure(figsize = (6,6))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('petal length (cm)', fontsize = 15)
ax.set_ylabel('petal width (cm)', fontsize = 15)

ax.set_title('Regressão linear', fontsize = 20)
targets = [0, 1, 2]
targetsLegend = ['setosa', 'versicolor', 'virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targetsLegend,colors):
    
    indicesToKeep = iris['species'] == target
    ax.scatter(iris.loc[indicesToKeep, 'petal_length']
               , iris.loc[indicesToKeep, 'petal_width']
               , c = color
               , s = 50)
ax.legend(targetsLegend)
ax.grid()
plt.plot(dado_x, reg.predict(dado_x), color='black', linewidth=2)
plt.savefig("LMS.png")

#pca

x = iris.drop('species', 1)
y = iris['species']



sc = StandardScaler()
x = sc.fit_transform(x)

pca = PCA(n_components=2)
x = pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_

print(explained_variance)

componentes_principais_pca = pd.DataFrame(data = x
             , columns = ['principal component 1', 'principal component 2'])


plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Componente principal - 1',fontsize=20)
plt.ylabel('Componente principal - 2',fontsize=20)
plt.title("PCA",fontsize=20)
targets = ['setosa', 'virginica','versicolor']
colors = ['r', 'g','c']
for target, color in zip(targets,colors):
    indicesToKeep = iris['species'] == target
    plt.scatter(componentes_principais_pca.loc[indicesToKeep, 'principal component 1']
               , componentes_principais_pca.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.savefig("PCA.png")

#lda

x = iris.drop('species', 1)
y = iris['species']
sc = StandardScaler()
x = sc.fit_transform(x)

lda = LinearDiscriminantAnalysis(n_components=2)
x = lda.fit(x, y).transform(x)

explained_variance = lda.explained_variance_ratio_

print(explained_variance)

componentes_principais_lda = pd.DataFrame(data = x
             , columns = ['principal component 1', 'principal component 2'])

plt.figure()
plt.figure(figsize=(10,10))
plt.xticks(fontsize=12)
plt.yticks(fontsize=14)
plt.xlabel('Componente principal - 1',fontsize=20)
plt.ylabel('Componente principal - 2',fontsize=20)
plt.title("LDA",fontsize=20)
targets = ['setosa', 'virginica','versicolor']
colors = ['r', 'g','c']
for target, color in zip(targets,colors):
    indicesToKeep = iris['species'] == target
    plt.scatter(componentes_principais_lda.loc[indicesToKeep, 'principal component 1']
               , componentes_principais_lda.loc[indicesToKeep, 'principal component 2'], c = color, s = 50)

plt.legend(targets,prop={'size': 15})
plt.savefig("LDA.png")


#SVM
	
iris = datasets.load_iris()
X = iris.data[:, 2:]  # we only take the last two features.
y = iris.target
C = 1.0  # SVM regularization parameter
 
# SVC with linear kernel
svc = svm.SVC(kernel='linear', C=C).fit(X, y)
# LinearSVC (linear kernel)
lin_svc = svm.LinearSVC(C=C).fit(X, y)
# SVC with RBF kernel
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=C).fit(X, y)
# SVC with polynomial (degree 3) kernel
poly_svc = svm.SVC(kernel='poly', degree=3, C=C).fit(X,y)
	
h = .02  # step size in the mesh
# create a mesh to plot in
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
	            np.arange(y_min, y_max, h))
# title for the plots
titles = ['SVC com kernel linear',
	  'LinearSVC (kernel linear)',
	  'SVC com kernel RBF',
	  'SVC com kernel polinomial']
 
 
for i, clf in enumerate((svc, lin_svc, rbf_svc, poly_svc)):
    
    plt.subplot(2, 2, i + 1)
    plt.subplots_adjust(wspace=0.4, hspace=0.4)
 
    Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
 
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
    plt.xlabel('Petal length')
    plt.ylabel('Petal width')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())
    plt.title(titles[i])
 
plt.savefig("SVM.png")


