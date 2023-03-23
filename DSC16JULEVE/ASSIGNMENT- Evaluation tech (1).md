Model evaluation is very important in data science. It helps you to understand the performance of your model and makes it easy to present your model to other people.

MSE-
MSE is calculated by the sum of square of prediction error which is real output minus predicted output and then divide by the number of data points.It gives you an absolute number on how much your predicted results deviate from the actual number. You cannot interpret many insights from one single result but it gives you a real number to compare against other model results and help you select the best regression model.

RMSE-
It gives you an absolute number on how much your predicted results deviate from the actual number. You cannot interpret many insights from one single result but it gives you a real number to compare against other model results and help you select the best regression model.

MAE-
Mean Absolute Error(MAE) is similar to Mean Square Error(MSE). However, instead of the sum of square of error in MSE, MAE is taking the sum of the absolute value of error.
MAE is a more direct representation of sum of error terms. MSE gives larger penalization to big prediction error by squaring it while MAE treats all errors the same.


```python
import pandas as pd
import numpy as np
```


```python
df=pd.DataFrame([[25,45000],[28,55000],[32,100000],[20,92000],[21,30000],[33,33000],[24,35000],[22,42000]],
                columns=["Age","Salary"])
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Age</th>
      <th>Salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>25</td>
      <td>45000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>28</td>
      <td>55000</td>
    </tr>
    <tr>
      <th>2</th>
      <td>32</td>
      <td>100000</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20</td>
      <td>92000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>21</td>
      <td>30000</td>
    </tr>
    <tr>
      <th>5</th>
      <td>33</td>
      <td>33000</td>
    </tr>
    <tr>
      <th>6</th>
      <td>24</td>
      <td>35000</td>
    </tr>
    <tr>
      <th>7</th>
      <td>22</td>
      <td>42000</td>
    </tr>
  </tbody>
</table>
</div>




```python
#assigning x and y for train test split
x=df[["Age"]]
y=df[["Salary"]]
```


```python
x=np.array(x)
x=x.reshape(-1,1)
print(x.ndim)
y=np.array(y)
y=y.reshape(-1,1)
print(y.ndim)
```

    2
    2
    


```python
#applying standard scaling
from sklearn.preprocessing import StandardScaler
sta = StandardScaler()
xst = sta.fit_transform(x)
print(xst)

yst = sta.fit_transform(y)
print(yst)
```

    [[-0.13563141]
     [ 0.51539936]
     [ 1.38344038]
     [-1.22068269]
     [-1.00367243]
     [ 1.60045064]
     [-0.35264167]
     [-0.78666218]]
    [[-0.35423738]
     [ 0.03935971]
     [ 1.8105466 ]
     [ 1.49566893]
     [-0.94463301]
     [-0.82655388]
     [-0.74783447]
     [-0.47231651]]
    


```python
from sklearn.model_selection import train_test_split
xtrain, xtest, ytrain, ytest =train_test_split(xst, yst, test_size=0.2)
print(xtrain.shape)
print(ytrain.shape)
print(xtest.shape)
print(ytest.shape)

```

    (6, 1)
    (6, 1)
    (2, 1)
    (2, 1)
    


```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(xtrain, ytrain)
```




    LinearRegression()




```python
lr.score(xtest,ytest)
```




    -1.135797167846731




```python
ypred= lr.predict(xtest)
ypred
```




    array([[0.12466585],
           [0.04345633]])




```python
print(len(xtest))
print(len(ytest))
print(len(ypred))
```

    2
    2
    2
    


```python
changes = list(ypred-ytest)
```


```python
changes
```




    [array([0.08530614]), array([0.39769371])]



MSE


```python
from sklearn.metrics import mean_squared_error
```


```python
mse =mean_squared_error(ytest,ypred)
mse
```




    0.08271871292977269



MAE


```python
from sklearn.metrics import mean_absolute_error
```


```python
mae = mean_absolute_error(ytest, ypred)
mae
```




    0.24149992757925948



RMSE


```python
rmse = mean_squared_error(ytest, ypred, squared = False)
rmse
```




    0.2876086106669491



r2score


```python
from sklearn.metrics import r2_score
```


```python
r2score = r2_score(ytest,ypred)
r2score
```




    -1.135797167846731



Ihe above exapmle RMSE is giving the best result out of other 3 so here rmse is best if used. As RMSE is the root of the high error value which gives a lower value as compared to MSE and hence easier to evaluate.
