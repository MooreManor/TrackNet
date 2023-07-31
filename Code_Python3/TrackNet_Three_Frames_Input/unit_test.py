# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sktime.classification.deep_learning.cnn import CNNClassifier
# from sklearn.preprocessing import StandardScaler
# # from sktime.classification.compose import TimeSeriesForestClassifier
# from sktime.classification.kernel_based import RocketClassifier
#
#
# # 随机生成速度和位置数据
# np.random.seed(0)
# Xs = np.random.randn(1000)
# Ys = np.random.randn(1000)
# Vxs = np.random.randn(1000)
# Vys = np.random.randn(1000)
#
# # 转换成20个时间长度的lag特征
# lag = 20
# # X = pd.DataFrame({'X': Xs, 'Y': Ys, 'Vx': Vxs, 'Vy': Vys})
# X = np.stack([Xs, Ys, Vxs, Vys], axis=1)
# # X_lagged = pd.concat([X.shift(i) for i in range(lag)], axis=1).dropna()
# X_lagged = np.stack([X[i:i-lag or None] for i in range(len(X))], axis=0)
# y = np.random.randint(0, 2, size=len(X_lagged))
#
# # 将数据集分成训练集和测试集
# X_train, X_test, y_train, y_test = train_test_split(X_lagged, y)
#
# # 定义管道：先对数据进行标准化再使用时间序列随机森林分类器
# clf = Pipeline([
#     ('transform', StandardScaler()),
#     ('classify', RocketClassifier(num_kernels=2000))
# ])
#
# # 拟合模型
# clf.fit(X_train, y_train)
#
# res = clf.predict(X_test)
# # 评估模型性能
# accuracy = clf.score(X_test, y_test)
# print(f"Accuracy: {accuracy}")

import numpy as np
# import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.datasets import load_basic_motions
from sklearn.metrics import accuracy_score
from sktime.transformations.panel.compose import ColumnConcatenator
from sktime.datasets import load_unit_test
X_train, y_train = load_unit_test(split="train", return_X_y=True)

motions_X, motions_Y = load_basic_motions(return_type="numpy3d")
motions_train_X, motions_train_y = load_basic_motions(
    split="train", return_type="numpy3d"
)
motions_test_X, motions_test_y = load_basic_motions(split="test", return_type="numpy3d")

from sktime.classification.interval_based import TimeSeriesForestClassifier
from sktime.classification.kernel_based import RocketClassifier

rocket = RocketClassifier(num_kernels=2000)
rocket.fit(motions_train_X, motions_train_y)
y_pred = rocket.predict(motions_test_X)
# steps = [('concatenate', ColumnConcatenator()),('classify', TimeSeriesForestClassifier(n_estimators=100))]
# clf = Pipeline(steps)
# clf.fit(motions_train_X, motions_train_y)
# res = clf.predict(motions_test_X)
res = clf.predict(motions_test_X)
# # # 评估模型性能
# accuracy = clf.score(res, motions_test_y)



# # ## Multivariate Classification
# #
# # Many classifiers, including ROCKET and HC2, are configured to work with multivariate input. For example:
#
# # In[ ]:
#
#
# from sktime.classification.kernel_based import RocketClassifier
#
# rocket = RocketClassifier(num_kernels=2000)
# rocket.fit(motions_train_X, motions_train_y)
# y_pred = rocket.predict(motions_test_X)
#
# accuracy_score(motions_test_y, y_pred)
#
#
# # In[ ]:
#
#
# from sktime.classification.hybrid import HIVECOTEV2
#
# HIVECOTEV2(time_limit_in_minutes=0.2)
# hc2.fit(motions_train_X, motions_train_y)
# y_pred = hc2.predict(motions_test_X)
#
# accuracy_score(motions_test_y, y_pred)
#
#
# # `sktime` offers two other ways of building estimators for multivariate time series problems:
# #
# # 1. Concatenation of time series columns into a single long time series column via `ColumnConcatenator` and apply a classifier to the concatenated data,
# # 2. Dimension ensembling via `ColumnEnsembleClassifier` in which one classifier is fitted for each time series column/dimension of the time series and their predictions are combined through a voting scheme.
# #
# # We can concatenate multivariate time series/panel data into long univariate time series/panel using a transform and then apply a classifier to the univariate data:
#
# # In[ ]:
#
#
# from sktime.classification.interval_based import DrCIF
# from sktime.transformations.panel.compose import ColumnConcatenator
#
# clf = ColumnConcatenator() * DrCIF(n_estimators=10, n_intervals=5)
# clf.fit(motions_train_X, motions_train_y)
# y_pred = clf.predict(motions_test_X)
#
# accuracy_score(motions_test_y, y_pred)
#
#
# # We can also fit one classifier for each time series column and then aggregate their predictions. The interface is similar to the familiar `ColumnTransformer` from `sklearn`.
#
# # In[ ]:
#
#
# from sktime.classification.compose import ColumnEnsembleClassifier
# from sktime.classification.interval_based import DrCIF
# from sktime.classification.kernel_based import RocketClassifier
#
# col = ColumnEnsembleClassifier(
#     estimators=[
#         ("DrCIF0", DrCIF(n_estimators=10, n_intervals=5), [0]),
#         ("ROCKET3", RocketClassifier(num_kernels=1000), [3]),
#     ]
# )
#
# col.fit(motions_train_X, motions_train_y)
# y_pred = col.predict(motions_test_X)
#
# accuracy_score(motions_test_y, y_pred)

# from sktime.datasets import (
#     load_arrow_head,
#     load_basic_motions,
#     load_japanese_vowels,
#     load_plaid,
# )
#
# plaid_train_X, plaid_train_y = load_plaid(split="train", return_type='numpy3D')
# plaid_test_X, plaid_test_y = load_plaid(split="test", return_type='numpy3D')
#
# from sktime.classification.feature_based import RandomIntervalClassifier
# from sktime.transformations.panel.padder import PaddingTransformer
#
# padded_clf = PaddingTransformer() * RandomIntervalClassifier(n_intervals=5)
# padded_clf.fit(plaid_train_X, plaid_test_y)
# y_pred = padded_clf.predict(plaid_test_X)
#
# accuracy_score(plaid_test_y, y_pred)


# ------------------------------------------------------------------------------------------------------
# from sktime.datasets import load_longley
# from sktime.forecasting.var import VAR
#
# _, y = load_longley()
#
# y = y.drop(columns=["UNEMP", "ARMED", "POP"])
#
# forecaster = VAR()
# forecaster.fit(y, fh=[1, 2, 3])
#
# y_pred = forecaster.predict()