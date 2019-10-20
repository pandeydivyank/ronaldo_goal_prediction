from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from feature_reduction import FeatureReduction
import pandas as pd

class DatasetGeneration:


    def __init__(self,
        data,
        split_ratio = 0.2, random_state = 0, max_features = 10, keep_info = 0.95,
        list_of_removals = None, nan_removal_method = 'by_mean'):

        self.data = data[pd.notnull(data['is_goal'])]
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
                'home/away',
                'Unnamed: 0'
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

        #print(categorical_variable)

        one_hots = []

        for feature in categorical_variable:
            one_hot = pd.get_dummies(data[feature])
            one_hots.append(one_hot)
            # Drop column B as it is now encoded
            data = data.drop(feature, axis = 1)
            # Join the encoded df

        # ss = StandardScaler()
        mms = MinMaxScaler()
        data_scaled = pd.DataFrame(mms.fit_transform(data))


        for feature, one_hot in zip(categorical_variable, one_hots):
            if feature == 'type_of_combined_shot':
                one_hot.columns = [str(col) + '_x' for col in one_hot.columns]
            data = data.join(one_hot)
            data_scaled = data_scaled.join(one_hot)


        feature_labels = list(data.columns)

        del data

        return data_scaled.to_numpy(), feature_labels

    def reduce_features(self, keep_info, max_features):

        data_X, feature_labels = self.categorical_conversion()

        data_reduction = FeatureReduction(
            data_X = data_X,
            data_y = self.target_creation(),
            feature_label = feature_labels,
            no_of_features = max_features)

        reduced_feature_set, selected_features = data_reduction.feature_selection(keep_info = keep_info)

        return reduced_feature_set, selected_features

    def train_test_split(self):

        reduced_feature_set, selected_features = self.reduce_features(keep_info = self.keep_info, max_features = self.max_features)

        X_train, X_test, y_train, y_test = train_test_split(reduced_feature_set, self.target_creation(), test_size = self.split_ratio, random_state = self.random_state)


        train_data = {}

        train_data['X'] = X_train
        train_data['y'] = y_train


        validation_data = {}

        validation_data['X'] = X_test
        validation_data['y'] = y_test


        return train_data, validation_data, selected_features

    # def feature_labels(self):
    #     return list(self.categorical_conversion().columns)

    def dataset_generation(self):
        train_data, validation_data, feature_label = self.train_test_split()
        return train_data, validation_data, feature_label


if __name__ == "__main__":
    import pandas as pd

    data = pd.read_csv('./data.csv')
    obj = DatasetGeneration(data = data)
    train_data, validation_data, feature_label = obj.dataset_generation()

    print('TRAIN DATA SHAPE: ', train_data['X'].shape, 'VALIDATION DATA SHAPE: ', validation_data['X'].shape, 'FEATURE LABELS: ', feature_label)




