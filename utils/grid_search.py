from sklearn.model_selection import GridSearchCV
from operator import itemgetter

def run_eval_gs(X,y,clf,cv,gs_params):

    gs = GridSearchCV(clf,gs_params,cv=cv,verbose=True)
    gs.fit(X,y)
    scores = gs.cv_results_['mean_test_score']
    ordered_scores = sorted(scores,reverse=True)[:30]
    print(ordered_scores)
