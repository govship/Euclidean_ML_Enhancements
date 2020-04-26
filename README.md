# Euclidean_ML_Enhancements
Using Euclidean distances to increase variance in a predominantly high bias ML model.


You ran your model and you made predictions. There is a chance that your model can be more accurate.
This tool uses your current model and makes a slight change to it the following way:

1. Set A - Consider your predictions as coordinates in an n-dimensional space
2. Set B - Consider the model your trained as coordinates as well (also in n-dimensional space)
3. Set C - Consider the coordinates of the original data set (also in n-dimensional space)

**Notice Sets A and B are the outputs of the model you ran**

4. Plot Set B in an n-dimensional space that contains points from your training data set
5. The algorithm traverses each point in Set B and considers points that belong to Set C that are within R, radius, distance away from that point;
   The identified points in Set C are considered as neighbors of the point from Set B
6. The algorithm then brings the point from Set B closer to those neighbors
7. You can compare if your original prediction is more or less accurate than the new predictions by running model_compare()


The initial phase of this tool assumes your are running a model from Sklearn or Tensorflow since both have the predict() method.
Future phases will be agnostic of this assumption.