import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score


def build_model(train, test, target_col, model):
    tmp_target = train[target_col]
    tmp_train = train.drop(target_col, axis=1)

    model.fit(tmp_train, tmp_target)
    test[target_col] = model.predict(test.drop(target_col, axis=1))
    return test[target_col]


def rmse(actual, predict):
    return np.sqrt(np.mean((actual - predict)**2))


def x_validation(train, target, model, cv, err_func=accuracy_score):
    """
    Perform K fold cross validations
    :param train: Training data with no target column
    :param target: Training target column
    :param model: Model to train on
    :param cv: cv
    :param err_func: By default error function is accuracy score. "err_func" can be replaced by the your custom err_func.
    :return: avg_score from the function
    """
    #
    avg_score = 0
    for traincv, testcv in cv:
        tmp = model.fit(train[traincv], target[traincv])
        prediction = tmp.predict(train[testcv])
        avg_score += (err_func(target[testcv], prediction))
    avg_score /= len(cv)
    print "Average score is {}".format(avg_score)
    return avg_score


def get_train_test(data_enc, test_split=0.2, stratify=True):
    """

    :param data_enc: First columns should be Y
    :param train_split: split_size in percent
    :param stratify: If stratify it uses Y values
    :return:
    """
    from sklearn.model_selection import train_test_split
    X, Y = data_enc.iloc[:, 1:], data_enc.iloc[:, [0]]
    stratify = Y if stratify else None
    return train_test_split(X, Y, test_size=test_split, stratify=stratify)


def fit_predict_unbalanced_model(model, Xtest, ytest):
    """
    For unbalanced data finds optimum threshold for split of probabilities
    Runs prediction using optimum threshold from ROC curve
    """

    def get_optimum_threshold(fpr, sensitivity, thresholds):
        """
        Gets the best threshold based on maximum sum of sensitivity and
        specificity
        :param fpr: FPR is (1-Specificity)
        :param sensitivity: TPR
        :param thresholds: List of threshold values
        :return:
        """
        specificity = (1 - fpr)
        return thresholds[np.argmax(sensitivity + specificity)]

    from sklearn.metrics import roc_curve, auc, accuracy_score, f1_score
    probas_ = model.predict_proba(Xtest)
    # Compute ROC curve and area the curve
    fpr, tpr, thresholds = roc_curve(ytest, probas_[:, 1])
    roc_auc = auc(fpr, tpr)
    opt_threshold = get_optimum_threshold(fpr, tpr, thresholds)
    print "Opt Threshold : ", opt_threshold, "F1 Score - ", f1_score(ytest, probas_[:, 1] > opt_threshold)
    print "F1 Score 0.5 threshold", f1_score(ytest, probas_[:, 1] > 0.5)
    print "Accuracy Score 0.5 Threshold", accuracy_score(ytest, probas_[:, 1] > 0.5)
    print "Opt Threshold : ", opt_threshold, " Accuracy score is ", accuracy_score(ytest, probas_[:, 1] > opt_threshold)
    print "AUC score", roc_auc

    return probas_, opt_threshold


def build_categorized_models(train_data, test_data, category, target, catg_model):
    """
    Builds model for each categories of input data and develops prediciton results for each category.
    X-validation is used for each folds of category. No params grid is passed
    :param train_data: Training data
    :param test: Test data
    :category : Model on which to develop for
    :target: column to predict
    :param catg_model: Model developed from GridSearch for each Category
    :param cv: cv

    """
    from sklearn import cross_validation
    categories = train_data[category].unique()
    train_data['data_type'] = 'Train'
    test_data['data_type'] = 'Test'
    data = pd.concat([train_data, test_data])
    avg_score = 0

    for catg in categories:
        print "-------{}------".format(catg)
        catg_data = data[data[category] == catg]
        catg_data = catg_data.drop([category], axis=1)
        train = catg_data[catg_data.data_type == 'Train']
        test = catg_data[catg_data.data_type == 'Test']
        train = train.drop(['data_type'], axis=1)
        test = test.drop(['data_type'], axis=1)

        tmp_target = train[target]
        tmp_train = train.drop(target, axis=1)
        cv = cross_validation.StratifiedKFold(train["Loan_Status"], n_folds=10)
        avg_scores = x_validation(tmp_train, tmp_target, catg_model[catg], cv)
        data.loc[(data[category] == catg) & (data.data_type == 'Test'),
                 target] = catg_model.predict(test.drop(target, axis=1))
        # avg_scores += scores
    print "Average score is {}".format(avg_score / len(categories))
    return data[len(train_data):][target]


def build_categorized_models_cv(train_data, test_data, category, target, model_dict, folds):
    """
    Builds model for each categories of input data and develops prediciton results for each category.
    @param model_dict: can be dict of categories or single model
    GridSearchCV is used for each folds.
    """
    from sklearn import cross_validation



    categories = train_data[category].unique()

    train_data['data_type'] = 'Train'
    test_data['data_type'] = 'Test'
    data = pd.concat([train_data, test_data])
    avg_score = 0
    params = {}
    for catg in categories:
        print "-------{}------".format(catg)
        if isinstance(model_dict, dict):
            model = model_dict[catg]
        else:
            model = model_dict
        catg_data = data[data[category] == catg]
        catg_data = catg_data.drop([category], axis=1)
        train = catg_data[catg_data.data_type == 'Train']
        test = catg_data[catg_data.data_type == 'Test']
        train = train.drop(['data_type'], axis=1)
        test = test.drop(['data_type'], axis=1)
        # Simple K-Fold cross validation. 10 folds.
        cv = cross_validation.StratifiedKFold(
            train["Loan_Status"], n_folds=folds)
        model.cv = cv
        tmp_target = train[target]
        tmp_train = train.drop(target, axis=1)
        model.fit(tmp_train, tmp_target)

        print model.best_params_
        print model.best_score_
        avg_score += model.best_score_
        params[catg] = model
        data.loc[(data[category] == catg) & (data.data_type == 'Test'),
                 target] = model.predict(test.drop(target, axis=1))
    print "Average score is {}".format(avg_score / len(categories))
    return data[len(train_data):][target], params
