__author__ = 'sshetty'

import preprocess_util,model_util
import pandas as pd
import numpy as np
reload(preprocess_util)
class Imputerator:
    def __init__(self, train, test, model):
        """

        :param train:
        :param test:
        :param model: "reg", "knn", "prob_dist"
        :return:
        """
        self.train = train
        self.test = test
        self.model = model
    def date_impute(self, df, ts_col, y_col, max_date, freq='M'):
        median_y = df[y_col].median()
        min_date = x[ts_col].min()
        date_df = (
            pd.DataFrame(pd.date_range(min_date, max_date, freq=freq), columns=[ts_col])
            )
        return df.merge(date_df, how='right').fillna(median_y)

    def impute(self, column_name):

        if self.model == 'reg':
            return self._call_regression(column_name)
        elif self.model =='catg_dist':
            return self._call_categorical_distribution(column_name)
        elif self.model == 'bayes':
            return self._call_bayes(column_name)
        elif self.model == 'rf':
            return self._call_rf(column_name)
        elif self.model == 'rf_reg':
            return self._call_rf_reg(column_name)
        elif self.model == 'gbm_regr':
            return self._call_gbm_regr(column_name)
        elif self.model == 'gbm':
            return self._call_gbm(column_name)
        else:
            self.model(column_name)

    def _call_categorical_distribution(self, predictor_variable):
        from scipy import stats
        import numpy as np
        counts = self.train[predictor_variable].value_counts()
        dist = stats.rv_discrete(values=(np.arange(counts.shape[0]),
                                         counts/counts.sum()))
        fill_idxs = dist.rvs(size=self.test.shape[0])
        return counts.iloc[fill_idxs].index.values

    def _call_regression(self, predictor_variable):
        from sklearn.linear_model import LinearRegression
        tmp = preprocess_util.create_dummies(pd.concat(self.train, self.test, keys=['train','test'])
        # Preprocess
        self.train = tmp.ix["train"].reset_index(drop=True)
        self.test = tmp.ix["test"].reset_index(drop=True)

        # Create linear regression object
        regr = LinearRegression()

        Y = self.train[predictor_variable]
        X = self.train.select(lambda x: x!= predictor_variable, axis=1)
        Xtest = self.test.select(lambda x: x!= predictor_variable, axis=1)

        # Train the model using the training sets
        try:
            # results, regr = preprocess_util.kFold_validate(X,Y,10, regr,lambda  y, ypred: np.mean(np.abs(y-ypred)/y))
            # print np.mean(results)
            regr.fit(X[Xtest.columns], Y)
            Ytest = regr.predict(Xtest)
        except Exception as e:
            print "_call_regression: Error in prediction", e
            return e
        return Ytest

    def _call_bayes(self, predictor_variable):
        from sklearn.naive_bayes import GaussianNB
        from sklearn.metrics import zero_one_loss
        # Pre process
        Y = self.train[predictor_variable]
        X = self.train.select(lambda x: x != predictor_variable, axis=1)
        Xtest = self.test.select(lambda x: x != predictor_variable, axis=1)

        X = preprocess_util.create_dummies(X)
        Xtest = preprocess_util.create_dummies(Xtest)

        # Create Bayes object
        gnb = GaussianNB()
        # results, fit = preprocess_util.kFold_validate(X[Xtest.columns], Y, 10, gnb, zero_one_loss)
        # print (1 - np.mean(results))
        fit = gnb.fit(X[Xtest.columns], Y)

        y_pred = fit.predict(Xtest)
        y_prob = pd.concat([pd.DataFrame(y_pred, columns=['output']), pd.DataFrame(fit.predict_proba(Xtest))], axis=1)

        return y_pred

    def _call_rf(self, predictor_variable):
        import pandas as pd
        from sklearn.metrics import zero_one_loss
        # Pre process
        Y = self.train[predictor_variable]
        X = self.train.select(lambda x: x != predictor_variable, axis=1)
        Xtest = self.test.select(lambda x: x != predictor_variable, axis=1)

        X = preprocess_util.create_dummies(X)
        Xtest = preprocess_util.create_dummies(Xtest)

        # Create RF object
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=500, random_state=50, n_jobs=-1, oob_score=True)
        fit = model.fit(X[Xtest.columns], Y)
        y_pred = fit.predict(Xtest)
        print model.oob_score_
        m = model.feature_importances_
        t= pd.DataFrame(dict([(Xtest.columns[i], m[i]) for i in xrange(0, len(m))]).items(), columns=['column','Imp'])
        t.sort(['Imp'])
        print t

        y_prob = pd.concat([pd.DataFrame(y_pred, columns=['output']), pd.DataFrame(fit.predict_proba(Xtest))], axis=1)
        return y_pred

    def _call_rf_reg(self, predictor_variable):
        from sklearn.ensemble import RandomForestRegressor
        # Pre process
        Y = self.train[predictor_variable]
        X = self.train.select(lambda x: x != predictor_variable, axis=1)
        Xtest = self.test.select(lambda x: x != predictor_variable, axis=1)

        X = preprocess_util.create_dummies(X)
        Xtest = preprocess_util.create_dummies(Xtest)

        # Create RF object
        model = RandomForestRegressor(n_estimators=500, random_state=50, n_jobs=-1, oob_score=True)
        # results, fit= preprocess_util.kFold_validate(X,Y,10, model,lambda y, ypred: np.mean(np.abs(y-ypred)/y))
        # print np.mean(results)
        fit = model.fit(X[Xtest.columns], Y)
        print model.oob_score_
        m = model.feature_importances_
        t= pd.DataFrame(dict([(Xtest.columns[i], m[i]) for i in xrange(0, len(m))]).items(), columns=['column','Imp'])
        print t.sort(['Imp'])
        y_pred = fit.predict(Xtest)
        return y_pred

    def _call_knn(self, predictor_variable):
        from sklearn.neighbors import KNeighborsClassifier
        # Pre process
        self.train = preprocess_util.create_dummies(self.train)
        self.test = preprocess_util.create_dummies(self.test)

    def _call_gbm_regr(self, predictor_variable):
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.grid_search import GridSearchCV

        # Pre process
        Y = self.train[predictor_variable]
        X = self.train.select(lambda x: x != predictor_variable, axis=1)
        Xtest = self.test.select(lambda x: x != predictor_variable, axis=1)

        X = preprocess_util.create_dummies(X)
        Xtest = preprocess_util.create_dummies(Xtest)
        # Create CV
        from sklearn import cross_validation
        cv = cross_validation.KFold(len(X), n_folds=10)
        # Create RF object
        model = GridSearchCV(GradientBoostingRegressor(n_estimators =100,max_depth=4, learning_rate=0.13),cv=cv,refit=True,
                        param_grid={})
        model.fit(X[Xtest.columns],Y)
        print model.best_score_
        print model.best_params_
#        m = model.feature_importances_
#        t= pd.DataFrame(dict([(Xtest.columns[i], m[i]) for i in xrange(0, len(m))]).items(), columns=['column','Imp'])
#        print t.sort(['Imp'])

        y_pred = model.predict(Xtest)
        return y_pred

    def _call_gbm(self, predictor_variable):
        from sklearn.ensemble import GradientBoostingClassifier
        from sklearn.grid_search import GridSearchCV

        # Pre process
        Y = self.train[predictor_variable]
        X = self.train.select(lambda x: x != predictor_variable, axis=1)
        Xtest = self.test.select(lambda x: x != predictor_variable, axis=1)

        X = preprocess_util.create_dummies(X)
        Xtest = preprocess_util.create_dummies(Xtest)
        # Create CV
        from sklearn import cross_validation
        cv = cross_validation.KFold(len(X), n_folds=10)
        # Create RF object
        model = GridSearchCV(GradientBoostingClassifier(n_estimators =100,max_depth=4, learning_rate=0.13),cv=cv,refit=True,
                        param_grid={})
        model.fit(X[Xtest.columns],Y)
        print model.best_score_
        print model.best_params_

        y_pred = model.predict(Xtest)
        return y_pred

