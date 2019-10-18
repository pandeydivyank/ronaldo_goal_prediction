from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from feature_reduction import FeatureReduction

class DatasetGeneration:

    def __init__(self,
        data,
        split_ratio = 0.2, random_state = 0, max_features = 10, keep_info = 0.95,
        list_of_removals = None, nan_removal_method = 'by_mean'):

        self.data = data
        self.split_ratio = split_ratio
        self.random_state = random_state
        self.nan_removal_method = nan_removal_method
        self.max_features = max_features
        self.keep_info = keep_info

        if list_of_removals is None:
            self.list_of_removals = [
                'match_event_id',
                'is_goal',
                'shot_id_number',
                'match_id',
                'team_id',
                'game_season',
                'team_name',
                'date_of_game',
                'lat/lng',
                'home/away'
            ]
        else:
            self.list_of_removals = list_of_removals

    def target_creation(self):
        return self.data['is_goal'].to_numpy()

    def dataset_creation(self):
        return self.data.drop(self.list_of_removals, axis = 1)

    def nan_removal(self, method = 'by_mean'):

        data = self.dataset_creation()

        if method is 'by_mean':
            data = data.fillna(data.mean())
        else:
            pass

        return data

    def categorical_conversion(self):

        data = self.nan_removal(method = self.nan_removal_method)

        categorical_variable = [key for key in dict(data.dtypes)
             if dict(data.dtypes)[key] in ['object'] ]

        print(categorical_variable)

        for feature in categorical_variable:
            print('FEATURE: ', feature)
            one_hot = pd.get_dummies(data[feature])
            # Drop column B as it is now encoded
            data = data.drop(feature, axis = 1)
            print('DATA SHAPE (BO): ', data.shape)
            # Join the encoded df
            data = data.join(one_hot)
            print('DATA SHAPE (AO): ', data.shape)

        return data

    def reduce_features(self, keep_info, max_features):

        data_reduction = FeatureReduction(
            data_X = self.categorical_conversion().to_numpy(),
            data_y = self.target_creation(),
            feature_label = self.feature_labels(),
            no_of_features = max_features)

        reduced_feature_set, selected_features = data_reduction.feature_selection(keep_info = keep_info)

        return reduced_feature_set, selected_features

    def train_test_split(self):

        reduced_feature_set, selected_features = self.reduce_features(keep_info = self.keep_info, max_features = self.max_features)


        X_train, X_test, y_train, y_test = train_test_split(reduced_feature_set, self.target_creation(), test_size = self.split_ratio, random_state = self.random_state)

        obj = StandardScaler()

        train_data = {}

        train_data['X'] = X_train
        train_data['y'] = y_train

        train_data['X'] = obj.fit_transform(train_data['X'])

        validation_data = {}

        validation_data['X'] = X_test
        validation_data['y'] = y_test

        validation_data['X'] = obj.fit_transform(validation_data['X'])

        return train_data, validation_data, selected_features

    def feature_labels(self):
        return list(self.categorical_conversion().columns)

    def dataset_generation(self):
        train_data, validation_data, feature_label = self.train_test_split()
        return train_data, validation_data, feature_label


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv('./data.csv')
    obj = DatasetGeneration(data = data)
    train_data, validation_data, feature_label = obj.dataset_generation()

    print('TRAIN DATA SHAPE: ', train_data.shape, 'VALIDATION DATA SHAPE: ', validation_data.shape, 'FEATURE LABELS: ', feature_label)




