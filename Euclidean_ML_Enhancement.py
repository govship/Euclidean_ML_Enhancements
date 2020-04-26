import pandas as pd
from scipy import spatial
from sklearn.metrics import mean_squared_error


class Euclidean_Neighbors:

    # train_set and test set in pandas dataframe (do not drop the target variable)
    # test_prediction is your output after running [model_name].predict() on your model
    # target_variable is the string name of the target variable in your data
    # model is the variable you stored your model in (e.g. if you have linreg = LinearRegression() then use linreg)
    def __init__(self, train_set, test_set, test_prediction, target_variable, model, radius):
        self.train_set = train_set
        self.test_set = test_set
        self.test_prediction = test_prediction
        self.target_variable = target_variable
        self.model = model
        self.radius = radius

    # creates a list of tuples for both train and test dataframe; stored as new columns in respective dataframe
    # each tuple (coordinates) represents a data instance (row) of the dataframe
    # adds prediction to the test_set dataframe
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

    # calculates the values of the model you used on the train set
    # calculates the difference between model values and target variable values
    def df_with_predictions(self):

        train_dup = self.df_values_to_tuples()[0]
        test_dup = self.df_values_to_tuples()[1]

        x_train_set = train_dup.drop(columns=[self.target_variable])

        # assumes you are using Sklearn or Tensorflow since they use the predict() method
        # in future I will remove this
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

    # checks the training coordinates that are within a certain radius from each predicted point
    # evaluates a new predicted point which brings it closer to those neighbors
    def closest_points_and_difference(self):

        train_tuples = self.df_values_to_tuples()[2]
        train_dup = self.df_with_predictions()[0]
        test_dup = self.df_with_predictions()[1]

        closest_points_by_indices = []
        tree = spatial.KDTree(train_tuples)
        for i in range(len(test_dup['Coordinates'])):
            point = test_dup['Coordinates'][i]
            # radius = df_test_2['std_difference'][i]
            closest_points_by_indices.append(tree.query_ball_point(point, self.radius))

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

    def model_compare(self):

        df = self.closest_points_and_difference()

        original_model = mean_squared_error(df['Prediction'], df[self.target_variable], squared=False)
        new_model = mean_squared_error(df['New_Prediction'], df[self.target_variable], squared=False)

        results = 'RMSE of old predictions is ' + str(original_model) + '.\nRMSE of new predictions is ' + str(new_model)

        if original_model <= new_model:
            results = results + '.\nOriginal predictions are more accurate.'
        else:
            results = results + '.\nNew predictions are more accurate.'

        return results