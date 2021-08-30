import numpy as np
from scipy import stats
from tqdm import tqdm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.model_selection import train_test_split
from math import factorial

def auuc_sep_rel_prop1(y, t, u):
    """
    Compute separate relative AUUC using Prop. 1 of https://dl.acm.org/doi/abs/10.1145/3447548.3467395
    """
    y_t = y[t==1]
    y_c = y[t==0]
    
    u_t = u[t==1]
    u_c = u[t==0]
    
    auc_t = roc_auc_score(y_t, u_t)
    auc_c = roc_auc_score(y_c, u_c)
    
    lambda_t = np.mean(y_t)*(1 - np.mean(y_t))
    lambda_c = np.mean(y_c)*(1 - np.mean(y_c))
    
    return lambda_t*auc_t - lambda_c*auc_c + (np.mean(y_t)**2) / 2. - (np.mean(y_c)**2) / 2.

def calc_U(y_true, y_score):
    n1 = np.sum(y_true==1)
    n0 = len(y_score)-n1
    
    ## Calculate the rank for each observation
    # Get the order: The index of the score at each rank from 0 to n
    order = np.argsort(y_score)
    # Get the rank: The rank of each score at the indices from 0 to n
    rank = np.argsort(order)
    # Python starts at 0, but statistical ranks at 1, so add 1 to every rank
    rank += 1
    
    # If the rank for target observations is higher than expected for a random model,
    # then a possible reason could be that our model ranks target observations higher
    U1 = np.sum(rank[y_true == 1]) - n1*(n1+1)/2
    return U1
        

def rank_risk_emp_var(y,u):
    n = len(u)
    norm_coef = factorial(n) / factorial(n-4) #A_n^4
    R = calc_U(y, u) #WMW statistic
    
    order = np.argsort(u)[::-1]
    sum1 = sum2 = 0
    kappa_pos = kappa_neg = 0
    for i in y[order]:
        if i == 0:
            kappa_pos += 1
            sum2 += kappa_neg*(kappa_neg - 1)
        if i == 1:
            kappa_neg += 1
            sum1 += kappa_pos*(kappa_pos - 1)
    last_term = sum1 + sum2
    emp_var = (2./norm_coef)*(-(R**2) + (n**2-5*n+8)*R + 4*last_term)
    return emp_var
    

def auuc_test_set_bound_bern(y,t,u,delta):
    '''
    Test set Bernstein bound (using empirical variance).
    '''
    y_bar_t = np.mean(y[t==1])
    y_bar_c = np.mean(y[t==0])
    lambda_t = y_bar_t*(1 - y_bar_t)
    lambda_c = y_bar_c*(1 - y_bar_c)
    n_t = y[t==1].shape[0]
    n_c = y[t==0].shape[0]
    
    emp_var_t = rank_risk_emp_var(y[t==1],u[t==1])**2
    emp_var_c = rank_risk_emp_var(1-y[t==0],u[t==0])**2
    
    bound = lambda_t*(np.sqrt(4*emp_var_t*np.log(8./delta)/n_t) + np.log(8./delta)*(10./n_t)) + lambda_c*(np.sqrt(4*emp_var_c*np.log(8./delta)/n_c) + np.log(8./delta)*(10./n_c))
    return bound


def c2st(X, y, clf=LogisticRegression(), loss=log_loss, bootstraps=300):
    """
    Perform Classifier Two Sample Test (C2ST) [1].
    
    This test estimates if a target is predictable from features by comparing the loss of a classifier learning 
    the true target with the distribution of losses of classifiers learning a random target with the same average.
    
    The null hypothesis is that the target is independent of the features - therefore the loss a classifier learning 
    to predict the target should not be different from the one of a classifier learning independent, random noise.
    
    Input:
        - `X` : (n,m) matrix of features
        - `y` : (n,) vector of target - for now only supports binary target
        - `clf` : instance of sklearn compatible classifier (default: `LogisticRegression`)
        - `loss` : sklearn compatible loss function (default: `hamming_loss`)
        - `bootstraps` : number of resamples for generating the loss scores under the null hypothesis
    
    Return: (
        loss value of classifier predicting `y`, 
        loss values of bootstraped random targets, 
        p-value of the test
    )
    
    Usage:
    >>> emp_loss, random_losses, pvalue = c2st(X, y)
    
    Plotting H0 and target loss:
    >>>bins, _, __ = plt.hist(random_losses)
    >>>med = np.median(random_losses)
    >>>plt.plot((med,med),(0, max(bins)), 'b')
    >>>plt.plot((emp_loss,emp_loss),(0, max(bins)), 'r--')
    
    [1] Lopez-Paz, D., & Oquab, M. (2016). Revisiting classifier two-sample tests. arXiv preprint arXiv:1610.06545.
    """
#     np.random.seed(111)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    y_pred = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
    emp_loss = loss(y_test, y_pred)
    bs_losses = []
    y_bar = np.mean(y)
    for b in tqdm(range(bootstraps)):
        y_random = np.random.binomial(1, y_bar, size=y.shape[0])
        X_train, X_test, y_train, y_test = train_test_split(X, y_random)
        y_pred_bs = clf.fit(X_train, y_train).predict_proba(X_test)[:,1]
        bs_losses += [loss(y_test, y_pred_bs)]
    pc = stats.percentileofscore(sorted(bs_losses), emp_loss) / 100.
    pvalue = pc if pc < y_bar else 1 - pc
    return emp_loss, np.array(bs_losses), pvalue