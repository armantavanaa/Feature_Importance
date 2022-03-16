from scipy.stats import spearmanr
import pandas as pd
from sklearn.datasets import load_boston
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from sklearn.metrics import mean_squared_error
import shap
import math


def my_sort(cols, imp):
    res = [x for _, x in sorted(zip(imp, cols))]
    imp = sorted(imp)

    return res, imp


def my_plt(feat, cols, t, t_coord, st_dev=None, err=False):
    fig, ax = plt.subplots(figsize=(8, 6))
    if err == False:
        ax.barh(np.arange(len(feat)), feat, color='#7c8fad', height=0.1, align='center', alpha=0.6)
        plt.plot(feat, np.arange(len(feat)), marker="o", linestyle="", alpha=0.8, color="g")
    else:
        ax.barh(np.arange(len(feat)), feat, xerr=st_dev, color='#7c8fad', height=0.7, align='center', alpha=0.6)

    ax.set_yticks([i for i in range(len(cols))])
    ax.tick_params(axis='y', direction='out', pad=50, left=False)
    ax.set_yticklabels(cols, fontsize=12, horizontalalignment='left')

    fig.align_ylabels()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    ax.tick_params(axis='x', bottom=False)

    plt.grid(axis='x', color="#dfe5f0", linestyle='--')
    ax.annotate(t, (t_coord, 14.8), size=16, color='black', weight='bold', annotation_clip=False)

    plt.show()


def my_spearman(X, y):
    corrs = []
    for col in X:
        c, p = spearmanr(X[col], y)
        corrs.append(abs(c))

    return corrs


def pca(X):
    scalar = StandardScaler()
    X = scalar.fit_transform(X)
    cov = np.cov(X.T)
    eigen_val, eigen_vec = np.linalg.eigh(cov)
    max_idx = np.argmax(np.abs(eigen_val))
    eigen_vec = eigen_vec.T
    corrs = []
    for i, v in enumerate(eigen_vec[max_idx]):
        corrs.append(abs(v))
    return corrs


def permutation_importances(mod, X, y, metric):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = mod(n_estimators=40, random_state=20)
    reg.fit(X_train, y_train)
    pred = reg.predict(X_val)
    baseline = metric(y_val, pred)
    imp = []
    for col in X_train.columns:
        save = X_val[col].copy()
        X_val[col] = np.random.permutation(X_val[col])
        pred = reg.predict(X_val)
        m = metric(y_val, pred)
        X_val[col] = save
        diff = baseline - m
        imp.append(diff)
    return imp


def dropcol_importances(mod, X, y):
    reg = mod(n_estimators=40, random_state=678, oob_score=True)
    reg_ = clone(reg)
    reg_.fit(X, y)
    baseline = reg_.oob_score_
    imp = []
    for col in X.columns:
        X_ = X.drop(col, axis=1)
        reg_ = clone(reg)
        reg_.fit(X_, y)
        o = reg_.oob_score_
        diff = baseline - o
        imp.append(diff)
    return imp


def compare_methods(k, mod, X, y, feat_cols):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = mod(n_estimators=40, random_state=20)
    mse = []
    for i in range(k):
        reg_ = clone(reg)
        reg_.fit(X_train.loc[:, feat_cols[:i + 1]], y_train)
        pred = reg_.predict(X_val.loc[:, feat_cols[:i + 1]])
        mse.append(mean_squared_error(y_val, pred))

    return mse


def comp_method_plot(k, spear, pca, pi, di, shap):
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.plot([i for i in range(1, k + 1)], spear, label="Spearman's Rank", marker='o')
    ax.plot([i for i in range(1, k + 1)], pca, label="PCA", marker='s')
    ax.plot([i for i in range(1, k + 1)], pi, label="Permutation Importance", marker='*')
    ax.plot([i for i in range(1, k + 1)], di, label="Drop Column Importance", marker='x')
    ax.plot([i for i in range(1, k + 1)], shap, label="Shap", marker='v')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines["bottom"].set_color("gray")
    ax.spines["left"].set_color("gray")
    plt.xlabel('Number of Features (k)')
    plt.ylabel('MSE')
    plt.legend()
    plt.show()


def feat_selection(mod, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=42)
    reg = mod(n_estimators=40, random_state=678, oob_score=True)
    reg_ = clone(reg)
    feat_di = dropcol_importances(RandomForestRegressor, X, y)
    cols_di, feat_di_sorted = my_sort(X.columns, feat_di)
    reg_.fit(X_train, y_train)
    pred = reg_.predict(X_val)
    loss = mean_squared_error(y_val, pred)
    dropped = []
    X__ = X.copy()
    while True:
        dropped.append(cols_di[0])
        reg_ = clone(reg)
        X_train = X_train.drop(columns=cols_di[0])
        X_val = X_val.drop(columns=cols_di[0])
        X__ = X__.drop(columns=cols_di[0])
        reg_.fit(X_train, y_train)
        pred = reg_.predict(X_val)
        new_loss = mean_squared_error(y_val, pred)
        if new_loss > (loss + 0.2) and feat_di_sorted[0] > 0:
            dropped.remove(cols_di[0])
            break
        else:
            feat_di = dropcol_importances(RandomForestRegressor, X__, y)
            cols_di, feat_di_sorted = my_sort(X__.columns, feat_di)
        loss = new_loss

    return dropped


def my_standard_dev(n, model, X, y):
    all_imps = {}
    for col in X.columns:
        all_imps[col] = []
    for i in range(n):
        bootstrap = np.random.choice(len(X), size=len(X), replace=True)
        feat_pi = permutation_importances(RandomForestRegressor, X.iloc[bootstrap, :], y.iloc[bootstrap], r2_score)
        cols_pi, feat_pi_sorted = my_sort(X.columns, feat_pi)
        for j, c in enumerate(cols_pi):
            all_imps[c].append(feat_pi_sorted[j])

    res = {}
    for k in all_imps:
        res[k] = np.std(all_imps[k]) * 0.5

    return res


def p_val(n, model, X, y):
    baseline_feat_pi = permutation_importances(model, X, y, r2_score)
    baseline_cols_pi, baseline_feat_pi_sorted = my_sort(X.columns, baseline_feat_pi)

    baseline = {baseline_cols_pi[i]: baseline_feat_pi_sorted[i] for i in range(len(baseline_cols_pi))}
    shuff_imps = {}
    for col in X.columns:
        shuff_imps[col] = []
    for i in range(n):
        y_shuff = y.copy().sample(frac=1.0)
        shuff_feat_pi = permutation_importances(model, X, y_shuff, r2_score)
        shuff_cols_pi, shuff_feat_pi_sorted = my_sort(X.columns, shuff_feat_pi)
        for j, c in enumerate(shuff_cols_pi):
            shuff_imps[c].append(shuff_feat_pi_sorted[j])

    p_vals = {}
    for col in X.columns:
        temp = np.array(shuff_imps[col])
        p_vals[col] = np.sum(0 >= (baseline[col] - temp)) / n

    return shuff_imps, baseline, p_vals


def my_hist(shuff, null, title) -> list:
    fig, ax = plt.subplots(figsize=(12,8))
    ax.hist(shuff, bins = 20)
    ax.axvline(null, c='red')
    plt.title(title)
    plt.show()
