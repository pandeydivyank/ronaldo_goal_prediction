from dataset_generation import DatasetGeneration
from model_architectures import ModelArchitectures
from import_export import ImportExport
import numpy as np

class Main:
    """
                            Raw Data
                                |
                          Import Export (Import Data)
                                |
                         Data Engineering
                                |
                       Feature Engineering
                                |
        Feature Selection       |
                |               |
                |_______________|
                                |
    Dimensionality Reduction    |                  -------> Model Hyperparameter Tuning
                |               |                 |                     |
                |_______________|                 |                     |
                                |                 |                     |
                        Dataset Generation (Processed Data)     Model Architectures
                                        |        |       ________________|
                                        |        |      |
                                        |         --> Model
                                        |               |     Training
                                        |               |________|
                                        |               |
                                        |__             |     Testing
                                           |            |________|
            Import Export (Export Results) |            |
                          |_____________   |  __________|
                                        |  | |
                                        Main (run)


    """
    @staticmethod
    def run():
        data = ImportExport.import_csv(file_location = './data_1.csv')

        obj = DatasetGeneration(data = data, max_features = 6)
        train_data, validation_data, feature_label = obj.dataset_generation()

        print('\n\n')

        for i in range(train_data['X'].shape[1]):

            query = "{} | MIN: {} | MAX: {} | MEAN: {} | STD: {} |".format(feature_label[i], np.min(train_data['X'][:,i]), np.max(train_data['X'][:,i]), np.mean(train_data['X'][:,i]), np.std(train_data['X'][:,i]))
            print(query)

        print('\n\n')

        print("TRAINING DATA [ X: {}, y: {}], VALIDATION DATA [ X: {}, y: {}] ".format(train_data['X'].shape, train_data['y'].shape,
            validation_data['X'].shape, validation_data['y'].shape))

        model_arch = ModelArchitectures(
            train_data = train_data,
            validation_data = validation_data,
            verbose = True,
            grid_search = False
            )

        svm_matrix = model_arch.svm_classifier()
        rf_matrix, _ = model_arch.random_forest_classifier()

if __name__ == "__main__":
    Main.run()
