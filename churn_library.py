'''
Function for Predict Customer Churn project.

Author: Udacity, Matheus
Date: December 2021
'''

import warnings
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings('ignore')



matplotlib.use('Agg')

sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth
    input:
            pth: a path to the csv
    output:
            dataframe: pandas dataframe
    '''

    print('Reading csv...')
    dataframe = pd.read_csv(pth)
    print('\rDataframe ready!')
    return dataframe


def perform_eda(dataframe, pth):
    '''
    perform eda on dataframe and save figures to images folder
    input:
            dataframe: pandas dataframe
            pth: path where images will be saved
    output:
            None
    '''

    print('Performing EDA...')
    # Churn Histogram
    dataframe['Churn'] = dataframe['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    plt.figure(figsize=(10, 5))
    dataframe['Churn'].hist()
    plt.savefig(pth + '/churn_histogram.png')

    # Customer Age Histogram
    plt.figure(figsize=(10, 5))
    dataframe['Customer_Age'].hist()
    plt.savefig(pth + '/customer_age_histogram.png')

    # Marital Status Bar
    plt.figure(figsize=(10, 10))
    dataframe.Marital_Status.value_counts('normalize').plot(kind='bar')
    plt.savefig(pth + '/marital_status_bar.png')

    # Total Trans CT Distribution
    plt.figure(figsize=(10, 5))
    sns.distplot(dataframe['Total_Trans_Ct'])
    plt.savefig(pth + '/total_trans_ct_distplot.png')

    # Correlation Heatmap
    plt.figure(figsize=(20, 20))
    sns.heatmap(dataframe.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(pth + '/correlation_heatmap.png')

    print(f'\rAll EDA images saved to {pth}')


def encoder_helper(dataframe, category_lst):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            dataframe: pandas dataframe
            category_lst: list of columns that contain categorical features
    output:
            dataframe: pandas dataframe with new columns for
    '''

    print('Encoding category features...')
    for col in category_lst:
        groups = dataframe.groupby(col).mean()['Churn']
        dataframe[col + '_Churn'] = dataframe[col].map(groups.to_dict())

    print('\rCategory features enconded!')
    return dataframe


def perform_feature_engineering(
        dataframe,
        keep_cols,
        test_size=0.3,
        random_state=42):
    '''
    filter dataframe with selected features and split into train and test parts.
    input:
              dataframe: pandas dataframe
              keep_cols: selected columns to filter in dataframe
              test_size: proportion of dataframe separated for test
              random_state: random seed to guarantee reproducibility
    output:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
    '''

    print('Performing feature engineering...')
    label = dataframe['Churn']
    features = dataframe[keep_cols].copy()

    print('\rTransformed dataframe ready!')
    return train_test_split(features, label, test_size=test_size,
                            random_state=random_state)


def train_models(
        x_train,
        x_test,
        y_train,
        y_test,
        img_pth,
        model_pth,
        param_grid,
        random_state=42,
        cross_val=5):
    '''
    train, store model results: images + scores, and store models
    input:
              x_train: x training data
              x_test: x testing data
              y_train: y training data
              y_test: y testing data
              img_pth: path where images will be saved
              model_pth: path where models will be saved
              param_grid: dictionary with parameters names and values
              random_state: random seed to guarantee reproducibility
              cv: number of folds for cross-validation
    output:
              None
    '''

    print('Training models...')
    # Model Definition
    rfc = RandomForestClassifier(random_state=random_state)
    lrc = LogisticRegression()

    # Model Training
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=cross_val)
    cv_rfc.fit(x_train, y_train)

    lrc.fit(x_train, y_train)
    print('Models Trained!')
    # Model Predictions
    y_train_preds_rf = cv_rfc.best_estimator_.predict(x_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(x_test)

    y_train_preds_lr = lrc.predict(x_train)
    y_test_preds_lr = lrc.predict(x_test)

    # Model Scoring
    # Random Forest Classification Report
    plt.figure(figsize=(10, 10))
    plt.text(0, 1, str('Random Forest Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0, .8, str(
            classification_report(
                y_train, y_train_preds_rf)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0, .6, str('Random Forest Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(0, .4, str(classification_report(y_test, y_test_preds_rf)), {
             'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(img_pth + '/rfc_report.png')

    # Logistic Regression Classification Report
    plt.figure(figsize=(10, 10))
    plt.text(0, 1, str('Logistic Regression Train'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0, 0.8, str(
            classification_report(
                y_train, y_train_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0, 0.6, str('Logistic Regression Test'), {
             'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0, 0.4, str(
            classification_report(
                y_test, y_test_preds_lr)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(img_pth + '/logistic_report.png')

    # ROC Curve
    lrc_plot = plot_roc_curve(lrc, x_test, y_test)
    plt.figure(figsize=(15, 8))

    axis = plt.gca()
    plot_roc_curve(
        cv_rfc.best_estimator_,
        x_test,
        y_test,
        ax=axis,
        alpha=0.8)
    lrc_plot.plot(ax=axis, alpha=0.8)
    plt.savefig(img_pth + '/roc_curve.png')
    print(f'All training images saved to {img_pth}')

    # Model Persistency
    joblib.dump(cv_rfc.best_estimator_, model_pth + '/rfc_model.pkl')
    joblib.dump(lrc, model_pth + '/logistic_model.pkl')
    print(f'Models saved to {model_pth}')


def feature_importance_plot(model, x_data, img_pth):
    '''
    creates and stores the feature importances in pth
    input:
            model: model object containing feature_importances_
            x_data: pandas dataframe of x values
            img_pth: path to store the figure

    output:
             None
    '''
    print('Generating feature importance...')
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [x_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(10, 20))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(x_data.shape[1]), importances[indices])

    # Add feature names as x-axis labels
    plt.xticks(range(x_data.shape[1]), names, rotation=90)

    # Save figure
    plt.savefig(img_pth + '/feature_importance.png')
    print(f'Result saved to {img_pth}')


if __name__ == '__main__':
    from constants import DATA_PTH, IMG_EDA_PTH, CAT_COLUMNS
    from constants import KEEP_COLS, TEST_SPLIT, RANDOM_SEED
    from constants import IMG_RESULTS_PTH, MODEL_PTH, CV

    data = import_data(DATA_PTH)
    perform_eda(data, IMG_EDA_PTH)
    data = encoder_helper(data, CAT_COLUMNS)
    datasets = perform_feature_engineering(
        data, KEEP_COLS, TEST_SPLIT, RANDOM_SEED)
    train_models(
        *datasets,
        IMG_RESULTS_PTH,
        MODEL_PTH,
        {},
        RANDOM_SEED,
        CV)
    rfc_model = joblib.load(MODEL_PTH + '/rfc_model.pkl')
    feature_importance_plot(rfc_model, datasets[0], IMG_RESULTS_PTH)
