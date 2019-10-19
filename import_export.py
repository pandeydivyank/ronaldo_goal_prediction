import pandas as pd

class ImportExport:

    @staticmethod
    def import_csv(file_location):
        return pd.read_csv(file_location)

if __name__ == "__main__":
    data = ImportExport.import_csv('./data.csv')
    print("DATA SHAPE: ", data.shape)

