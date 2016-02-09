import calendar
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from core.algo.features import FeatureGenerator
import numpy as np
from sklearn import cross_validation
from sklearn.externals import joblib
# from core.database.models import Model
from datetime import datetime
import os

class NoModelException(Exception):
    pass

class Manager(object):
    def __init__(self, question):
        self.question = question

    def score_essay(self, text,MODEL_PATH):
        model_obj = joblib.load(MODEL_PATH)
        return model_obj.predict(text)

    def get_latest_model(self,models):
        if len(models) == 0:
            raise NoModelException

        model = models[-1]
        return model

    def create_model(self,id, text, scores, MODEL_PATH):
        # text = [e.text for e in self.question.essays if e.actual_score is not None]
        # scores = [e.actual_score for e in self.question.essays if e.actual_score is not None]
        scorer = Scorer(text, scores)
        scorer.train()
        time = datetime.utcnow()
        timestamp = calendar.timegm(time.utctimetuple())
        path_string = "{0}_{1}.pickle".format(id, timestamp)
        # model = Model(
        #     question=self.question,
        #     error=scorer.cv_score,
        #     path=path_string
        # )
        joblib.dump(scorer, os.path.join(MODEL_PATH, path_string), compress=9)
        return os.path.join(MODEL_PATH, path_string)

class Scorer(object):
    classification_max = 4
    cv_folds = 2

    def __init__(self, text, scores):
        self.text = text
        self.scores = scores
        self.feature_generator = FeatureGenerator()
        self.classifier = RandomForestRegressor(
            n_estimators=100,
            min_samples_split=4,
            min_samples_leaf=3,
            random_state=1
        )

        unique_scores = set(scores)
        if len(unique_scores) <= self.classification_max:
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                min_samples_split=4,
                min_samples_leaf=3,
                random_state=1
            )

        self.fit_feats()
        self.fit_done = False

    def fit_feats(self):
        self.feature_generator.fit(self.text, self.scores)

    def get_features(self):
        feats = []
        for t in self.text:
            feats.append(self.feature_generator.get_features(t))

        feat_mat = np.vstack(feats)
        return feat_mat

    def train(self):
        feats = self.get_features()
        scores = np.array(self.scores)

        # Compute error metrics for the estimator.
        self.cv_scores = cross_validation.cross_val_score(self.classifier, feats, scores)
        self.cv_score = self.cv_scores.mean()
        self.cv_dev = self.cv_scores.std()

        self.classifier.fit(feats, scores)
        self.fit_done = True

    def predict(self, text):
        feats = self.feature_generator.get_features(text)
        return self.classifier.predict(feats)