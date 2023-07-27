import numpy as np
# import seaborn as sns
from sklearn.model_selection import GridSearchCV

from sktime.classification.deep_learning.cnn import CNNClassifier
from sktime.datasets import load_basic_motions
from sklearn.metrics import accuracy_score

motions_X, motions_Y = load_basic_motions(return_type="numpy3d")
motions_train_X, motions_train_y = load_basic_motions(
    split="train", return_type="numpy3d"
)
motions_test_X, motions_test_y = load_basic_motions(split="test", return_type="numpy3d")


from sktime.classification.kernel_based import RocketClassifier

rocket = RocketClassifier(num_kernels=2000)
rocket.fit(motions_train_X, motions_train_y)
y_pred = rocket.predict(motions_test_X)

accuracy_score(motions_test_y, y_pred)



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


