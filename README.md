# anti-spam_model_project
Parsing and preprocessing data from .eml files, using them to train some models, and analyzing results
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 1595 entries, 0 to 1594
Data columns (total 6 columns):
| # |  Column     | Non-Null Count | Dtype 
|---|  ------     | -------------- | ----- 
| 0 |  Unnamed: 0 | 1595 non-null  | int64 
| 1 |  From       | 1594 non-null  | object
| 2 |  To         | 1539 non-null  | object
| 3 |  Date       | 1594 non-null  | object
| 4 |  Text       | 1578 non-null  | object
| 5 |  Mark       | 1595 non-null  | object

dtypes: int64(1), object(5)
memory usage: 74.9+ KB

Mark:
spam        995 |
not spam    600

Name: count, dtype: int64

![image](https://github.com/user-attachments/assets/3f2e18d1-90bd-481d-a5ec-c24d35f14e26)

<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Unnamed: 0</th>
      <th>From</th>
      <th>To</th>
      <th>Date</th>
      <th>Text</th>
      <th>Mark</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1595.000000</td>
      <td>1594</td>
      <td>1539</td>
      <td>1594</td>
      <td>1578</td>
      <td>1595</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1090</td>
      <td>549</td>
      <td>1546</td>
      <td>1451</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>boingboing &lt;rssfeeds@example.com&gt;</td>
      <td>yyyy@example.com</td>
      <td>Fri, 29 Mar 2002 05:01:01 +0000</td>
      <td>dear paypal member account randomly flagged sy...</td>
      <td>spam</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>31</td>
      <td>142</td>
      <td>4</td>
      <td>8</td>
      <td>995</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>797.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>460.581155</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>398.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>797.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1195.500000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1594.000000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

![image](https://github.com/user-attachments/assets/251eefb3-f096-4cfb-a87b-05afc8ca4c7c)

![image](https://github.com/user-attachments/assets/ce0e6777-3adf-4117-8247-282ad3203ee0)

Neural Network (MLP) Performance:
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|----------
|            0 |      0.95 |     0.94 |     0.95 |      125
|            1 |      0.96 |     0.97 |     0.97 |      191
|     accuracy |           |          |     0.96 |      316
|    macro avg |      0.96 |     0.96 |     0.96 |      316
| weighted avg |      0.96 |     0.96 |     0.96 |      316

Logistic Regression Performance:
|              |  precision |    recall | f1-score |  support
|--------------|------------|-----------|----------|----------              
|            0 |       0.97 |      0.94 |     0.96 |      125
|            1 |       0.96 |      0.98 |     0.97 |      191
|     accuracy |            |           |     0.97 |      316
|    macro avg |       0.97 |      0.96 |     0.96 |      316
| weighted avg |       0.97 |      0.97 |     0.97 |      316

Support Vector Machine (SVM) Performance:
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|----------
|            0 |      0.97 |     0.96 |     0.96 |      125
|            1 |      0.97 |     0.98 |     0.98 |      191
|     accuracy |           |          |     0.97 |      316
|    macro avg |      0.97 |     0.97 |     0.97 |      316
| weighted avg |      0.97 |     0.97 |     0.97 |      316

Bernoulli Naive Bayes Performance:
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|----------
|            0 |      0.85 |     0.98 |     0.91 |      125
|            1 |      0.99 |     0.89 |     0.94 |      191
|     accuracy |           |          |     0.93 |      316
|    macro avg |      0.92 |     0.94 |     0.93 |      316
| weighted avg |      0.94 |     0.93 |     0.93 |      316



