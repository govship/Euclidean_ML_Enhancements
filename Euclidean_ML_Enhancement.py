import pandas as pd
from scipy import spatial


class Euclidean_Neighbors:

    def __init__(self, train_set, test_set, test_prediction, target_variable, model):
        self.train_set = train_set
        self.test_set = test_set
        self.test_prediction = test_prediction
        self.target_variable = target_variable
        self.model = model

    # enter training set, testing set
    # train set and test set should include the target variable
    # test set should include the predicted values from your model
    # make sure each row in test set corresponds to the appropriate prediction value (list)
    # in the function we drop the target variable from test set and use prediction values instead

    def df_values_to_tuples(self):

        pd.options.mode.chained_assignment = None

        train_dup = self.train_set.copy()
        test_dup = self.test_set.copy()

        train_tuples = [tuple(x) for x in train_dup.to_numpy()]
        # train_dup['Coordinates'] = train_tuples

        test_dup['Prediction'] = self.test_prediction
        test_tuples = [tuple(x) for x in test_dup.drop(columns=[self.target_variable]).to_numpy()]
        test_dup['Coordinates'] = test_tuples

        return train_dup, test_dup, train_tuples

    # enter training set, testing set, values of the model from prediction of train set,
    # prediction values of test set, and name of the target variable (string)

    # e.g. if used sklearn, run model.predict(x_train_set) where x_train_set does not have target variable
    # store the above in a variable and use that variable for "train_prediction" as argument in df_with_predictions

    def df_with_predictions(self):

        train_dup = self.df_values_to_tuples()[0]
        test_dup = self.df_values_to_tuples()[1]

        x_train_set = train_dup.drop(columns=[self.target_variable])

        train_dup['Model_Output'] = self.model.predict(x_train_set)
        train_dup['Difference'] = train_dup[self.target_variable] - train_dup['Model_Output']
        train_dup = train_dup.reset_index()
        train_dup = train_dup.drop(columns=['index'])

        '''
        test_dup['Prediction'] = test_predictions
        test_dup['Difference'] = test_dup[target_variable] - test_dup['Prediction']
        '''
        test_dup = test_dup.reset_index()
        test_dup = test_dup.drop(columns=['index'])

        return train_dup, test_dup

    def closest_points_and_difference(self, radius):

        train_tuples = self.df_values_to_tuples()[2]
        train_dup = self.df_with_predictions()[0]
        test_dup = self.df_with_predictions()[1]

        closest_points_by_indices = []
        tree = spatial.KDTree(train_tuples)
        for i in range(len(test_dup['Coordinates'])):
            point = test_dup['Coordinates'][i]
            # radius = df_test_2['std_difference'][i]
            closest_points_by_indices.append(tree.query_ball_point(point, radius))

        test_dup['Closest Points By Indices'] = closest_points_by_indices

        add_to_prediction = []
        for i in range(len(test_dup['Closest Points By Indices'])):
            indices_sum = 0
            for index in test_dup['Closest Points By Indices'][i]:
                indices_sum += train_dup['Difference'][index]
            avg_indices_sum = 0
            if len(test_dup['Closest Points By Indices'][i]) > 0:
                avg_indices_sum = indices_sum / (len(test_dup['Closest Points By Indices'][i]))
            else:
                avg_indices_sum = 0
            add_to_prediction.append(avg_indices_sum)

        test_dup['Delta_Prediction'] = add_to_prediction
        test_dup['New_Prediction'] = test_dup['Prediction'] + test_dup['Delta_Prediction']

        return test_dup