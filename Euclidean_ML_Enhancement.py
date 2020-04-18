import pandas as pd
from scipy import spatial


# enter training set, testing set
# train set and test set should include the target variable
# test set should include the predicted values from your model
# make sure each row in test set corresponds to the appropriate prediction value (list)
# in the function we drop the target variable from test set and use prediction values instead

def df_values_to_tuples(train_set, test_set, test_prediction, y_variable):

    #train_set, test_set, train_tuples = df_with_predictions(train_set, test_set, train_prediction, test_prediction, y_variable)

    pd.options.mode.chained_assignment = None

    #train_set, test_set, train_tuples = df_values_to_tuples(train_set, test_set, test_prediction, y_variable)

    train_tuples = [tuple(x) for x in train_set.to_numpy()]
    #train_set['Coordinates'] = train_tuples

    test_set['Prediction'] = test_prediction
    test_tuples = [tuple(x) for x in test_set.drop(columns=[y_variable]).to_numpy()]
    test_set['Coordinates'] = test_tuples

    return train_set, test_set, train_tuples


# enter training set, testing set, values of the model from prediction of train set,
# prediction values of test set, and name of the target variable (string)

# e.g. if used sklearn, run model.predict(x_train_set) where x_train_set does not have target variable
# store the above in a variable and use that variable for "train_prediction" as argument in df_with_predictions

def df_with_predictions(train_set, test_set, train_prediction, test_prediction, y_variable):

    train_set, test_set, train_tuples = df_values_to_tuples(train_set, test_set, test_prediction, y_variable)

    train_set['Model_Output'] = train_prediction
    train_set['Difference'] = train_set[y_variable] - train_set['Model_Output']
    train_set = train_set.reset_index()
    train_set = train_set.drop(columns=['index'])

    '''
    test_set['Prediction'] = test_predictions
    test_set['Difference'] = test_set[y_variable] - test_set['Prediction']
    '''
    test_set = test_set.reset_index()
    test_set = test_set.drop(columns=['index'])

    return train_set, test_set, train_tuples


def closest_points_and_difference(train_set, test_set, train_prediction, test_prediction, y_variable, radius):

    train_set, test_set, train_tuples = df_with_predictions(train_set, test_set, train_prediction, test_prediction, y_variable)

    closest_points_by_indices = []
    tree = spatial.KDTree(train_tuples)
    for i in range(len(test_set['Coordinates'])):
        point = test_set['Coordinates'][i]
        # radius = df_test_2['std_difference'][i]
        closest_points_by_indices.append(tree.query_ball_point(point, radius))

    test_set['Closest Points By Indices'] = closest_points_by_indices

    add_to_prediction = []
    for i in range(len(test_set['Closest Points By Indices'])):
        indices_sum = 0
        for index in test_set['Closest Points By Indices'][i]:
            indices_sum += train_set['Difference'][index]
        avg_indices_sum = 0
        if len(test_set['Closest Points By Indices'][i]) > 0:
            avg_indices_sum = indices_sum / (len(test_set['Closest Points By Indices'][i]))
        else:
            avg_indices_sum = 0
        add_to_prediction.append(avg_indices_sum)

    test_set['Delta_Prediction'] = add_to_prediction
    test_set['New_Prediction'] = test_set['Prediction'] + test_set['Delta_Prediction']

    return test_set


############################################################################################################
############################################################################################################
############################################################################################################

from sklearn.datasets import load_boston
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR

boston = load_boston()
df_boston = pd.DataFrame(boston.data, columns=boston.feature_names)
df_boston['target'] = boston['target']
train_set, test_set = train_test_split(df_boston, test_size=0.10, random_state=7)
x_train_set = train_set.drop(columns=['target'])
y_train = train_set[['target']]
x_test_set = test_set.drop(columns=['target'])
y_test = test_set[['target']]

svr = SVR()
svr.fit(x_train_set, y_train.values.ravel())
prediction_test = svr.predict(x_test_set)
prediction_train = svr.predict(x_train_set)


test_set = closest_points_and_difference(train_set, test_set, prediction_train, prediction_test, 'target', 1)
print(test_set)

############################################################################################################
############################################################################################################
############################################################################################################