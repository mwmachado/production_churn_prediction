'''
Logging and testing functions from Predict Customer Churn project.

Author: Udacity, Matheus
Date: December 2021
'''

import warnings
import os
import logging
from glob import glob
import joblib
warnings.filterwarnings('ignore')

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


class TestChurn():
    '''Class to implent tests for Predict Custormer Churn project'''

    def __init__(self):
        '''Object initialization'''
        self.data = None
        self.datasets = None
        self.rfc_model = None

    def test_import(self, import_data, pth):
        '''
        test function import_data
        '''
        try:
            assert isinstance(pth, str)
            assert len(pth) > 0
        except AssertionError as err:
            logging.error(
                'test_import: ERROR: Constant pth must be string and not empty')
            raise err
        else:
            logging.info('test_import: SUCCESS: pth=%s', pth)

        try:
            self.data = import_data(pth)
        except FileNotFoundError as err:
            logging.error("test_import: ERROR: The file wasn't found")
            raise err
        else:
            logging.info("test_import: SUCCESS: Data imported")

        try:
            assert self.data.shape[0] > 0
            assert self.data.shape[1] > 0
        except AssertionError as err:
            logging.error(
                "test_import: ERROR: The file doesn't appear to have rows and columns")
            raise err
        else:
            logging.info(
                "test_import: SUCCESS: Dataframe shape=%s",
                self.data.shape)

    def test_eda(self, perform_eda, pth, n_png):
        '''
        test perform eda function
        '''
        try:
            assert isinstance(pth, str)
            assert len(pth) > 0
        except AssertionError as err:
            logging.error('test_eda: ERROR: path must be string and not empty')
            raise err
        else:
            logging.info('test_eda: SUCCESS: pth=%s', pth)

        try:
            perform_eda(self.data, pth)
        except KeyError as err:
            logging.error("test_eda: ERROR: Failed to execute perform_eda")
            raise err
        else:
            logging.info("test_eda: SUCCESS: EDA performed")

        try:
            images = len(glob(pth + '/*.png'))
            assert images == n_png
        except AssertionError as err:
            logging.error(
                "test_eda: ERROR: A different number (%s) of png images were found at %s",
                images,
                pth)
            raise err
        else:
            logging.info(
                "test_eda: SUCCESS: %s png images found at %s as expected",
                n_png,
                pth)

    def test_encoder_helper(self, encoder_helper, category_lst, result_lst):
        '''
        test encoder helper
        '''
        try:
            assert isinstance(category_lst, list)
            assert isinstance(result_lst, list)
            assert len(category_lst) > 0
            assert len(result_lst) > 0
            assert len(result_lst) == len(category_lst)
        except AssertionError as err:
            logging.error(
                'test_encoder_helper: ERROR: category_lst and result_lst must be lists,'
                ' not empty, with the same length')
            raise err
        else:
            logging.info(
                'test_encoder_helper: SUCCESS: loaded category_lst and result_lst'
                ' with %s elements each', len(result_lst))

        try:
            assert category_lst <= self.data.columns.tolist()
        except AssertionError as err:
            logging.error(
                'test_encoder_helper: ERROR: one or more columns in category_lst is'
                ' not present in the dataset')
            raise err
        else:
            logging.info(
                'test_encoder_helper: SUCCESS: all columns in category_lst are present'
                ' in the dataset')

        try:
            self.data = encoder_helper(self.data, category_lst)
        except KeyError as err:
            logging.error(
                "test_encoder_helper: ERROR: Failed to execute encoder_helper")
            raise err
        else:
            logging.info(
                "test_encoder_helper: SUCCESS: Category features encoded")

    def test_perform_feature_engineering(
            self, perform_feature_engineering, keep_cols):
        '''
        test perform_feature_engineering
        '''
        try:
            assert isinstance(keep_cols, list)
            assert len(keep_cols) > 0
        except AssertionError as err:
            logging.error(
                'test_perform_feature_engineering: ERROR: keep_cols must be a list and not empty')
            raise err
        else:
            logging.info(
                'test_perform_feature_engineering: SUCCESS: loaded keep_cols with %s elements',
                len(keep_cols))

        try:
            assert keep_cols <= self.data.columns.tolist()
        except AssertionError as err:
            logging.error(
                'test_perform_feature_engineering: ERROR: one or more columns in keep_cols is'
                ' not present in the dataset')
            raise err
        else:
            logging.info(
                'test_perform_feature_engineering: SUCCESS: all columns in keep_cols are present'
                'in the dataset')

        try:
            self.datasets = perform_feature_engineering(
                self.data, KEEP_COLS, TEST_SPLIT, RANDOM_SEED)
        except KeyError as err:
            logging.error(
                "test_perform_feature_engineering: ERROR: Failed to execute"
                " perform_feature_engineering"
            )
            raise err
        else:
            logging.info(
                "test_perform_feature_engineering: SUCCESS: Feature engineering performed")
            logging.info(
                "test_perform_feature_engineering: SUCCESS: 2 datasets for training and"
                " 2 for test were created")

    def test_train_models(self, train_models, img_pth, model_pth, params):
        '''
        test train_models
        '''
        try:
            assert isinstance(img_pth, str)
            assert isinstance(model_pth, str)
            assert isinstance(params, dict)
            assert len(img_pth) > 0
            assert len(model_pth) > 0
        except AssertionError as err:
            logging.error(
                'test_train_models: ERROR: img_pth and model_pth must be a strings and not empty')
            logging.error('test_train_models: ERROR: params must be a dict')
            raise err
        else:
            logging.info(
                'test_train_models: SUCCESS: img_pth=%s model_pth=%s',
                img_pth,
                model_pth)
            logging.info('test_train_models: SUCCESS: params=%s', params)

        try:
            train_models(*self.datasets, img_pth, model_pth, params)
            self.rfc_model = joblib.load(model_pth + '/rfc_model.pkl')
        except KeyError as err:
            logging.error(
                "test_train_models: ERROR: Failed to execute train_models")
            raise err
        else:
            logging.info(
                "test_train_models: SUCCESS: Models trained and save to %s",
                model_pth)
            logging.info(
                "test_train_models: SUCCESS: Model scores were saved to %s",
                img_pth)

    def test_feature_importance_plot(self, feature_importance_plot, pth):
        '''
        test feature_importance_plot
        '''
        try:
            feature_importance_plot(self.rfc_model, self.datasets[0], pth)
        except KeyError as err:
            logging.error(
                "test_feature_importance_plot: ERROR: Failed to execute feature_importance_plot")
            raise err
        else:
            logging.info(
                "test_feature_importance_plot: SUCCESS: Feature importance plotted")

        try:
            assert 'feature_importance.png' in os.listdir(pth)
        except AssertionError as err:
            logging.error(
                "test_feature_importance_plot: ERROR: feature_importance.png were not found in %s",
                pth)
            raise err
        else:
            logging.info(
                "test_feature_importance_plot: SUCCESS: feature_importance.png is in %s", pth)


if __name__ == '__main__':
    import churn_library as clb
    from constants import DATA_PTH, IMG_EDA_PTH, CAT_COLUMNS
    from constants import KEEP_COLS, TEST_SPLIT, RANDOM_SEED
    from constants import IMG_RESULTS_PTH, MODEL_PTH
    from constants import EDA_IMAGES, ENCODED_COLUMNS

    integration_test = TestChurn()
    integration_test.test_import(clb.import_data, DATA_PTH)
    integration_test.test_eda(clb.perform_eda, IMG_EDA_PTH, EDA_IMAGES)
    integration_test.test_encoder_helper(
        clb.encoder_helper, CAT_COLUMNS, ENCODED_COLUMNS)
    integration_test.test_perform_feature_engineering(
        clb.perform_feature_engineering, KEEP_COLS)
    integration_test.test_train_models(
        clb.train_models, IMG_RESULTS_PTH, MODEL_PTH, {})
    integration_test.test_feature_importance_plot(
        clb.feature_importance_plot, IMG_RESULTS_PTH)
    del integration_test
