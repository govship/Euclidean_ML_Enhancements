from Euclidean_ML_Enhancement import Euclidean_Neighbors
import pandas as pd
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
#prediction_train = svr.predict(x_train_set)

euclidean = Euclidean_Neighbors(train_set, test_set, prediction_test, 'target', svr)
prediction_final = euclidean.closest_points_and_difference(25)
print(prediction_final)

print(mean_squared_error(prediction_final['Prediction'], prediction_final['target'], squared=False))
print(mean_squared_error(prediction_final['New_Prediction'], prediction_final['target'], squared=False))