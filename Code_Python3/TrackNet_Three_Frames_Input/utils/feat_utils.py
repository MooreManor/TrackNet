import pandas as pd
from sktime.datatypes._panel._convert import from_2d_array_to_nested
import numpy as np

# def get_lag_feature(x, y, v):
#     test_df = pd.DataFrame({'x': x, 'y': y, 'V': v})
#     for i in range(20, 0, -1):
#         test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
#     for i in range(20, 0, -1):
#         test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
#     for i in range(20, 0, -1):
#         test_df[f'lagV_{i}'] = test_df['V'].shift(i, fill_value=0)
#     test_df.drop(['x', 'y', 'V'], 1, inplace=True)
#     Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
#                   'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
#                   'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
#                   'lagX_2', 'lagX_1']]
#     Xs = from_2d_array_to_nested(Xs.to_numpy())
#
#     Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
#                   'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
#                   'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
#                   'lagY_3', 'lagY_2', 'lagY_1']]
#     Ys = from_2d_array_to_nested(Ys.to_numpy())
#
#     Vs = test_df[['lagV_20', 'lagV_19', 'lagV_18',
#                   'lagV_17', 'lagV_16', 'lagV_15', 'lagV_14', 'lagV_13', 'lagV_12',
#                   'lagV_11', 'lagV_10', 'lagV_9', 'lagV_8', 'lagV_7', 'lagV_6', 'lagV_5',
#                   'lagV_4', 'lagV_3', 'lagV_2', 'lagV_1']]
#     Vs = from_2d_array_to_nested(Vs.to_numpy())
#     X = pd.concat([Xs, Ys, Vs], 1)
#     X = X.values
#     len_X = len(X)
#     X = np.stack([np.array(series.values) for series in X.reshape(-1)])
#     X = X.reshape(len_X, 20, 3)
#     X = X.reshape(len_X, 60)
#     return X


def get_lag_feature(x, y, vx, vy):
    test_df = pd.DataFrame({'x': x, 'y': y, 'Vx': vx, 'Vy': vy})
    for i in range(20, 0, -1):
        test_df[f'lagX_{i}'] = test_df['x'].shift(i, fill_value=0)
    for i in range(20, 0, -1):
        test_df[f'lagY_{i}'] = test_df['y'].shift(i, fill_value=0)
    for i in range(20, 0, -1):
        test_df[f'lagVx_{i}'] = test_df['Vx'].shift(i, fill_value=0)
    for i in range(20, 0, -1):
        test_df[f'lagVy_{i}'] = test_df['Vy'].shift(i, fill_value=0)
    test_df.drop(['x', 'y', 'Vx', 'Vy'], 1, inplace=True)
    Xs = test_df[['lagX_20', 'lagX_19', 'lagX_18', 'lagX_17', 'lagX_16',
                  'lagX_15', 'lagX_14', 'lagX_13', 'lagX_12', 'lagX_11', 'lagX_10',
                  'lagX_9', 'lagX_8', 'lagX_7', 'lagX_6', 'lagX_5', 'lagX_4', 'lagX_3',
                  'lagX_2', 'lagX_1']]
    Xs = from_2d_array_to_nested(Xs.to_numpy())

    Ys = test_df[['lagY_20', 'lagY_19', 'lagY_18', 'lagY_17',
                  'lagY_16', 'lagY_15', 'lagY_14', 'lagY_13', 'lagY_12', 'lagY_11',
                  'lagY_10', 'lagY_9', 'lagY_8', 'lagY_7', 'lagY_6', 'lagY_5', 'lagY_4',
                  'lagY_3', 'lagY_2', 'lagY_1']]
    Ys = from_2d_array_to_nested(Ys.to_numpy())

    Vxs = test_df[['lagVx_20', 'lagVx_19', 'lagVx_18',
                  'lagVx_17', 'lagVx_16', 'lagVx_15', 'lagVx_14', 'lagVx_13', 'lagVx_12',
                  'lagVx_11', 'lagVx_10', 'lagVx_9', 'lagVx_8', 'lagVx_7', 'lagVx_6', 'lagVx_5',
                  'lagVx_4', 'lagVx_3', 'lagVx_2', 'lagVx_1']]
    Vxs = from_2d_array_to_nested(Vxs.to_numpy())

    Vys = test_df[['lagVy_20', 'lagVy_19', 'lagVy_18',
                  'lagVy_17', 'lagVy_16', 'lagVy_15', 'lagVy_14', 'lagVy_13', 'lagVy_12',
                  'lagVy_11', 'lagVy_10', 'lagVy_9', 'lagVy_8', 'lagVy_7', 'lagVy_6', 'lagVy_5',
                  'lagVy_4', 'lagVy_3', 'lagVy_2', 'lagVy_1']]
    Vys = from_2d_array_to_nested(Vys.to_numpy())
    X = pd.concat([Xs, Ys, Vxs, Vys], 1)
    X = X.values
    len_X = len(X)
    X = np.stack([np.array(series.values) for series in X.reshape(-1)])
    X = X.reshape(len_X, 20, 4)
    # X = X.reshape(len_X, 4, 20)
    # X = X.reshape(len_X, 80)
    return X

def get_single_lag_feature(var, lag=20):
    test_df = pd.DataFrame({'var': var})
    for i in range(lag, 0, -1):
        test_df[f'lagVar_{i}'] = test_df['var'].shift(i, fill_value=0)

    test_df.drop(['var'], 1, inplace=True)
    Vars = test_df.filter(regex=r'^lagVar_', axis=1)
    Vars = Vars.reindex(columns=[f'lagVar_{i}' for i in range(lag, 0, -1)])
    Vars = from_2d_array_to_nested(Vars.to_numpy())
    X = Vars.values
    len_X = len(X)
    X = np.stack([np.array(series.values) for series in X.reshape(-1)])
    X = X.reshape(len_X, lag, 1)
    return X