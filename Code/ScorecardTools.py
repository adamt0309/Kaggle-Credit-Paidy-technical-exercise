import numpy as np
from pandas.api.types import is_numeric_dtype
import pandas as pd

def get_bins(estimator):
    ''' takes a single variable decision tree and spits out the breakpoints.'''

    n_nodes = estimator.tree_.node_count
    children_left = estimator.tree_.children_left
    children_right = estimator.tree_.children_right
    feature = estimator.tree_.feature
    threshold = estimator.tree_.threshold

    node_depth = np.zeros(shape=n_nodes,dtype=np.int64)
    is_leaves = np.zeros(shape=n_nodes,dtype=bool)
    stack = [(0,-1)] # seed is the root node id and its patrent depth

    while len(stack)>0:
        node_id,parent_depth = stack.pop()
        node_depth[node_id] = parent_depth + 1

        # if we have test mode
        if (children_left[node_id])!=children_right[node_id]:
            stack.append((children_left[node_id],parent_depth+1))
            stack.append((children_right[node_id],parent_depth+1))
        else:
            is_leaves[node_id] = True

        boundaries = list()
        for i in range(n_nodes):
            if not is_leaves[i]:
                boundaries.append(threshold[i])
        boundaries.sort()
        return [-np.inf]+ boundaries + [np.inf]


def calc_mivs(X,target,pred_target,var,bins=None):
    X_ = X.copy()

    if is_numeric_dtype(X_[var]):
        X_['band'] = pd.cut(X_[var],bins)
    else:
        X_['band'] = X_[var].astype(str)

    x = X_.groupby('band')[[target,pred_target]].agg(['count','sum']).copy()

    x['goods'] = x[(target,'count')] - x[(target,'sum')]
    x['bads'] = x[(target,'sum')]
    x['perc_goods'] = x['goods']/x['goods'].sum()
    x['perc_bads'] = x['bads']/x['bads'].sum()

    x['exp_goods'] = x[(pred_target,'count')] - x[(pred_target,'sum')]
    x['exp_bads'] = x[(pred_target,'sum')]
    x['exp_perc_goods'] = x['exp_goods']/x['exp_goods'].sum()
    x['exp_perc_bads'] = x['exp_bads']/x['exp_bads'].sum()

    x['woe'] = np.log(x['perc_goods']/x['perc_bads'])
    x['iv_part'] = (x['perc_goods'] - x['perc_bads'])*x['woe']
    x['iv_part'] = np.where(x['iv_part']==np.inf,0,x['iv_part'])

    x['exp_woe'] = np.log(x['exp_perc_goods']/x['exp_perc_bads'])
    x['delta_score'] = (x['woe'] - x['exp_woe'])
    x['miv_part'] = (x['perc_goods'] - x['perc_bads'])*x['delta_score']
    x['miv_part'] = np.where(x['miv_part']==np.inf,0,x['miv_part'])

    return [x['iv_part'].sum(),x['miv_part'].sum(),bins]