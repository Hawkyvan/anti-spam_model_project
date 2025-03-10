# anti-spam_model_project
Parsing and preprocessing data from .eml files, using them to train some models, and analyzing 

Данный проект создан в целях обучения основам Machine Learning и Data Science

Возможности данного проекта:
- автоматизированный сбор данных (отправитель, получатель, дата письма, текст письма, метка спам/не спам) из файлов с расширением .eml (parsing)
- преобразование данных в .CSV (рекомендуется проверка целостности и точности переноса данных после преобразования)
- предподготовка данных для машинного обучения, включая векторизацию
- анализ преобразованных данных
- обучение моделей, основанных на логической регрессии, нейронной сети, методе опорных векторов, Байесовском алгоритме Бернулли
- вывод тестовых результатов каждой модели

## Инструкция запуска 
**Необходима версия Python не ниже 3.12.7**

1. Загрузить архив и распаковать в удобную директорию
2. Для сбора данных и сохранения их в формат .CSV запустить *parser.py*
3. Для анализа уже готового набора данных *eml_dataset.csv* запустить *analysis.ipynb*
4. Для запуска процесса машинного обучения моделей следует запустить *models.ipynb*

## Анализ готового набора данных *eml_dataset.csv*

*eml_dataset.csv* содержит основные сведения, необходимые для анализа данных и машинного обучения:
- отправитель: "From"
- получатель: "To"
- дата письма: "Date"
- текст письма: "Text"
- метка письма (спам/не спам): "Mark"

### Результаты анализа *eml_dataset.csv*

Основной статистический анализ
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

По результатам основного анализа, можно выделить следующие ключевые элементы:
- общее кол-во файлов: 1600
- 

![image](https://github.com/user-attachments/assets/15ae23a6-f9c4-43c1-8990-9236c85467fe)

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
