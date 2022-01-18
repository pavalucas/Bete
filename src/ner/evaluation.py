"""
This module contains an algorithm to evaluate proposed models.
Author: Lucas Pavanelli
"""
import pandas as pd
from sklearn.metrics import accuracy_score
from seqeval.metrics import classification_report


class Evaluation:
    """
    Evaluates proposed models.

    Parameters
    ----------
    output_folder : str
        Path to output folder

    Attributes
    ----------
    output_folder : str
        Path to output folder
    """
    def __init__(self, output_folder):
        self.output_folder = output_folder

    @staticmethod
    def convert_output_to_text(y, out_id2w):
        """
        Converts output list containing integers to a text list.

        Parameters
        ----------
        y : list
            Output list containing integers.
        out_id2w : dict
            Map of index to word.

        Returns
        -------
        list
            Text list.
        """
        result = []
        for indexes in y:
            result.append([out_id2w[index] for index in indexes])
        return result

    @staticmethod
    def get_single_output_id_list(y):
        """
        Converts a list of lists into a single list.

        Parameters
        ----------
        y : list
            Output list.

        Returns
        -------
        list
            Single list.
        """
        return [index for indexes in y for index in indexes]

    def evaluate(self, num_experiment, y_true, y_pred):
        """
        Evaluates model's predictions using classification_report, generates a csv containing classification report,
        and return micro avg f1-score

        Parameters
        ----------
        num_experiment : int
            Current experiment number
        y_true : list
            True output list.
        y_pred : list
            Predicted output list.

        Returns
        -------
        float
            F1 score of current experiment
        """
        print(classification_report(y_true, y_pred))
        class_report = classification_report(y_true, y_pred, output_dict=True)
        df = pd.DataFrame(class_report).transpose()
        df.to_csv(f'{self.output_folder}classification_report_experiment_{num_experiment}.csv')
        return class_report['micro avg']['f1-score']

    def generate_output_csv(self, file_name, y_true, y_pred, test_tokens):
        """
        Generates a csv containing test results.

        Parameters
        ----------
        file_name : str
            Name of csv file.
        y_true : list
            True output list.
        y_pred : list
            Predicted output list.
        test_tokens : list
            List of tokens in test data
        """
        csv_dict = {'id': [], 'token': [], 'true_tag': [], 'pred_tag': []}
        test_id = 0
        for cur_tokens, cur_y_true, cur_y_pred in zip(test_tokens, y_true, y_pred):
            for token, true_tag, pred_tag in zip(cur_tokens, cur_y_true, cur_y_pred):
                csv_dict['id'].append(test_id)
                csv_dict['token'].append(token)
                csv_dict['true_tag'].append(true_tag)
                csv_dict['pred_tag'].append(pred_tag)
            test_id += 1
        df = pd.DataFrame(csv_dict, columns=['id', 'token', 'pred_tag', 'true_tag'])
        df.to_csv(f'{self.output_folder}{file_name}.csv', index=False)