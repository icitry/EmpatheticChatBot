import pandas as pd


class InputParser:
    def __init__(self):
        self._dataset = None

    def read_from_file(self, csv_paths, csv_columns):
        self._dataset = pd.DataFrame(columns=csv_columns)

        for path in csv_paths:
            curr_dataset = pd.read_csv(path['path'], usecols=csv_columns, delimiter=path['delimiter'])
            curr_dataset.drop(index=0, inplace=True)
            self._dataset = pd.concat([self._dataset, curr_dataset], ignore_index=True)

        self._dataset.dropna(subset=csv_columns, inplace=True)  # remove invalid entries

    def get_organized_data(self, training_percent):
        self._dataset = self._dataset.sample(frac=1)  # shuffle the input
        training_entries = int(len(self._dataset) * training_percent)

        return self._dataset[:training_entries], self._dataset[training_entries + 1:]
