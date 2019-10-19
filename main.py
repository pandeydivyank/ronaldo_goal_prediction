from dataset_generation import DatasetGeneration
from model_architectures import ModelArchitectures
from import_export import ImportExport

class Main:
    @staticmethod
    def run():
        data = ImportExport.import_csv(file_location = './data.csv')

        obj = DatasetGeneration(data = data)
        train_data, validation_data, feature_label = obj.dataset_generation()

        model_arch = ModelArchitectures(
            train_data = train_data,
            validation_data = validation_data
            )

        svm_matrix = model_arch.svm_classifier()
        rf_matrix, _ = model_arch.random_forest_classifier()

if __name__ == "__main__":
    Main.run()





