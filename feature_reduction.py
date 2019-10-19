from sklearn import decomposition
from sklearn.ensemble import RandomForestClassifier
import scipy.stats as ss
import numpy as np
import pandas as pd

class FeatureReduction:

    def __init__(self, data_X, data_y, feature_label, no_of_features = 10):
        self.data_X = data_X
        self.data_y = data_y
        self.feature_label = feature_label
        self.no_of_features = no_of_features

    def pca(self, keep_info):

        pca = decomposition.PCA(keep_info)
        pca.fit(np.nan_to_num(self.data_X, nan = 0))

        feature_selected = []
        for a, b in zip(pca.components_, pca.explained_variance_ratio_):
            feature_selected.append([self.feature_label[list(ss.rankdata(np.abs(1/a))).index(i+1)] for i in range(self.no_of_features)])
            break

        return feature_selected[0]

    def random_forest(self):
        rf = RandomForestClassifier()
        rf.fit(self.data_X, self.data_y)

        feature_with_impurity =  (sorted(zip(map(lambda x: round(x, 4), rf.feature_importances_), self.feature_label), reverse=True))

        feature_selected = feature_with_impurity[:self.no_of_features]


        return feature_selected

    def feature_selection(self, keep_info = 0.95):

        feature_selected =[
        self.pca(keep_info = keep_info),
        self.random_forest()
        ]

        feat_sel = []

        i = 0
        for method in feature_selected:
            for feature in method:
                if i == 1:
                    if feature[1] not in feat_sel:
                        feat_sel.append(feature[1])
                else:
                    if feature not in feat_sel:
                        feat_sel.append(feature)
            i += 1

        print('\n\n\nIMPORTANT FEATURES:\n\n', feat_sel, '\n\n\n')

        indices = [self.feature_label.index(feature_) for feature_ in feat_sel]

        selected_data = np.array([self.data_X[:,i] for i in indices])

        reduced_feature_set = np.array([selected_data[:,i] for i in range(selected_data.shape[1])])

        return reduced_feature_set, feat_sel




