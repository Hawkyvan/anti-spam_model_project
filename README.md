# anti-spam_model_project
Parsing and preprocessing data from .eml files, using them to train some models, and analyzing results

RangeIndex: 1600 entries, 0 to 1599
Data columns (total 6 columns):
| # |  Column     | Non-Null Count | Dtype 
|---|-------------|----------------|------ 
| 0 |  Unnamed: 0 | 1600 non-null  | int64 
| 1 |  From       | 1599 non-null  | object
| 2 |  To         | 1544 non-null  | object
| 3 |  Date       | 1599 non-null  | object
| 4 |  Text       | 1583 non-null  | object
| 5 |  Mark       | 1600 non-null  | object

dtypes: int64(1), object(5)
memory usage: 75.1+ KB

Mark:
spam        1000 |
not spam     600

Name: count, dtype: int64

![image](https://github.com/user-attachments/assets/15ae23a6-f9c4-43c1-8990-9236c85467fe)

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
      <td>1600.00000</td>
      <td>1599</td>
      <td>1544</td>
      <td>1599</td>
      <td>1583</td>
      <td>1600</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>1094</td>
      <td>550</td>
      <td>1551</td>
      <td>1455</td>
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
      <td>1000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>799.50000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>462.02453</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.00000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>399.75000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>799.50000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1199.25000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>1599.00000</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>

![image](https://github.com/user-attachments/assets/c19e5f9c-4304-40a9-89f4-184e34082281)

![image](https://github.com/user-attachments/assets/e3fdf455-53d6-40bc-85cc-af2682cc4190)

Neural Network (MLP) Performance:
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|----------
|            0 |      0.96 |     0.95 |     0.95 |      128
|            1 |      0.96 |     0.97 |     0.97 |      189
|     accuracy |           |          |     0.96 |      317
|    macro avg |      0.96 |     0.96 |     0.96 |      317
| weighted avg |      0.96 |     0.96 |     0.96 |      317

Logistic Regression Performance:
|              | precision |  recall | f1-score |  support
|--------------|-----------|---------|----------|----------
|            0 |      0.98 |    0.93 |     0.95 |      128
|            1 |      0.95 |    0.98 |     0.97 |      189
|     accuracy |           |         |     0.96 |      317
|    macro avg |      0.96 |    0.96 |     0.96 |      317
| weighted avg |      0.96 |    0.96 |     0.96 |      317

SVM Performance:
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|----------
|            0 |      0.95 |     0.97 |     0.96 |      128
|            1 |      0.98 |     0.97 |     0.97 |      189
|     accuracy |           |          |     0.97 |      317
|    macro avg |      0.97 |     0.97 |     0.97 |      317
| weighted avg |      0.97 |     0.97 |     0.97 |      317

Bernoulli Naive Bayes Performance:
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|----------
|            0 |      0.86 |     0.95 |     0.90 |      128
|            1 |      0.97 |     0.89 |     0.93 |      189
|     accuracy |           |          |     0.92 |      317
|    macro avg |      0.91 |     0.92 |     0.92 |      317
| weighted avg |      0.92 |     0.92 |     0.92 |      317
