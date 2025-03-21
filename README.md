# anti-spam_model_project
Parsing and preprocessing data from .eml files, using them to train some models, and analyzing 

Данный проект создан в целях изучения основ Machine Learning и Data Science

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
- "From":
  - Кол-во заполненных ячеек: 1599
  - Уникальных значений: 1094
  - Самое частое значение: boingboing <rssfeeds@example.com>
  - Кол-во вхождений самого частого значения: 31
- "To":
  - Кол-во заполненных ячеек: 1544
  - Уникальных значений: 550
  - Самое частое значение: yyyy@example.com
  - Кол-во вхождений самого частого значения: 142
- "Date":
  - Кол-во заполненных ячеек: 1599
  - Уникальных значений: 1551
  - Самое частое значение: Fri, 29 Mar 2002 05:01:01 +0000
  - Кол-во вхождений самого частого значения: 4
- "Text":
  - Кол-во заполненных ячеек: 1583
  - Уникальных значений: 1455
  - Самое частое значение: dear paypal member account randomly flagged sy...	
  - Кол-во вхождений самого частого значения: 8
- "Mark":
  - Кол-во заполненных ячеек: 1600
  - Уникальных значений: 2
  - Самое частое значение: spam
  - Кол-во вхождений самого частого значения: 1000
 
Cравнение кол-ва писем спам/не спам в наборе данных

![image](https://github.com/user-attachments/assets/15ae23a6-f9c4-43c1-8990-9236c85467fe)

Данный график отображает соотношение кол-ва обработанных писем спам/не спам, находящихся в файле *eml_dataset.csv*:
- spam: 1000
- not spam: 600

Самые частые отправители

![image](https://github.com/user-attachments/assets/c19e5f9c-4304-40a9-89f4-184e34082281)

На этом графике отображены сведения по 10 самым частым отправителям (отправитель, кол-во сообщений):
1. boingboing <rssfeeds@example.com>: 31 письмо
2. guardian <rssfeeds@example.com>: 27 писем
3. Tom <tomwhore@slack.net>: 22 письма
4. PayPal <service@paypal.com>: 17 писем
5. Gary Lawrence Murphy <garym@canada.cim>: 14 писем
6. tim.one@comcast.net: 14 писем
7. gamasutra <rssfeeds@example.com>: 13 писем
8. bugzilla-daemon@hughes-family.org: 13 писем
9. fark <rssfeeds@example.com>: 13 писем
10. Tim Chapman <timc@2ubh.com>: 11 писем

Рсапределние писем по частоте во времени

![image](https://github.com/user-attachments/assets/e3fdf455-53d6-40bc-85cc-af2682cc4190)

Этот график иллюстрирует частоту отправлений во временном диапазоне:
Год самых частых отправлений писем, вне зависимости от их статуса - 2002: около 190 писем

## Результаты обучения различных моделей

Было создано две выборки: 75/25 и 80/20, также подготовленны гиперпараметры для проведения сравнения результатов обучения моделей.

В таблицах представленных ниже not spam соответствует классификатору "0", а spam - "1".

### 1) Анализ результатов обучения моделей с выборкой 75%/25% без настройки гиперпараметров

**Neural Network (MLP) performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.95 |     0.97 |     0.96 |      144
|            1 |      0.98 |     0.97 |     0.98 |      252
|     accuracy |           |          |     0.97 |      396
|    macro avg |      0.96 |     0.97 |     0.97 |      396
| weighted avg |      0.97 |     0.97 |     0.97 |      396

Общая точность определения среди всех параметров (accuracy): 97%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 95%, spam - 98%

Доля истинно положительных объектах во всех положительных (recall): not spam - 97%, spam - 97%

Среднее гармоническое между presicion и recall (f1-score): not spam - 96%, spam - 98%


**Logistic Regression performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.90 |     0.95 |     0.93 |      144
|            1 |      0.97 |     0.94 |     0.96 |      252
|     accuracy |           |          |     0.94 |      396
|    macro avg |      0.94 |     0.95 |     0.94 |      396
| weighted avg |      0.95 |     0.94 |     0.94 |      396

Общая точность определения среди всех параметров (accuracy): 94%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 90%, spam - 97%

Доля истинно положительных объектах во всех положительных (recall): not spam - 95%, spam - 94%

Среднее гармоническое между presicion и recall (f1-score): not spam - 93%, spam - 96%


**SVM performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.87 |     0.98 |     0.92 |      144
|            1 |      0.99 |     0.91 |     0.95 |      252
|     accuracy |           |          |     0.94 |      396
|    macro avg |      0.93 |     0.95 |     0.93 |      396
| weighted avg |      0.94 |     0.94 |     0.94 |      396

Общая точность определения среди всех параметров (accuracy): 94%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 87%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 98%, spam - 91%

Среднее гармоническое между presicion и recall (f1-score): not spam - 92%, spam - 95%


**Bernoulli Naive Bayes performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.81 |     0.99 |     0.89 |      144
|            1 |      1.00 |     0.87 |     0.93 |      252
|     accuracy |           |          |     0.91 |      396
|    macro avg |      0.90 |     0.93 |     0.91 |      396
| weighted avg |      0.93 |     0.91 |     0.91 |      396

Общая точность определения среди всех параметров (accuracy): 91%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 81%, spam - 100%

Доля истинно положительных объектах во всех положительных (recall): not spam - 99%, spam - 87%

Среднее гармоническое между presicion и recall (f1-score): not spam - 89%, spam - 93%



### 2) Анализ обучения моделей с выборкой 80%/20% без настройки гиперпараметров

**Neural Network (MLP) performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|--------- 
|            0 |      0.94 |     0.97 |     0.96 |      119
|            1 |      0.98 |     0.96 |     0.97 |      198
|     accuracy |           |          |     0.97 |      317
|    macro avg |      0.96 |     0.97 |     0.97 |      317
| weighted avg |      0.97 |     0.97 |     0.97 |      317

Общая точность определения среди всех параметров (accuracy): 97%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 94%, spam - 98%

Доля истинно положительных объектах во всех положительных (recall): not spam - 97%, spam - 96%

Среднее гармоническое между presicion и recall (f1-score): not spam - 96%, spam - 97%


**Logistic Regression performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.91 |     0.96 |     0.93 |      119
|            1 |      0.97 |     0.94 |     0.96 |      198 
|     accuracy |           |          |     0.95 |      317
|    macro avg |      0.94 |     0.95 |     0.95 |      317
| weighted avg |      0.95 |     0.95 |     0.95 |      317

Общая точность определения среди всех параметров (accuracy): 95%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 91%, spam - 97%

Доля истинно положительных объектах во всех положительных (recall): not spam - 96%, spam - 94%

Среднее гармоническое между presicion и recall (f1-score): not spam - 93%, spam - 96%


**SVM performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.88 |     0.98 |     0.93 |      119
|            1 |      0.99 |     0.92 |     0.95 |      198
|     accuracy |           |          |     0.94 |      317
|    macro avg |      0.93 |     0.95 |     0.94 |      317
| weighted avg |      0.95 |     0.94 |     0.94 |      317

Общая точность определения среди всех параметров (accuracy): 94%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 88%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 98%, spam - 92%

Среднее гармоническое между presicion и recall (f1-score): not spam - 93%, spam - 95%


**Bernoulli Naive Bayes performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.81 |     0.99 |     0.89 |      119
|            1 |      0.99 |     0.86 |     0.92 |      198
|     accuracy |           |          |     0.91 |      317
|    macro avg |      0.90 |     0.93 |     0.91 |      317
| weighted avg |      0.92 |     0.91 |     0.91 |      317

Общая точность определения среди всех параметров (accuracy): 91%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 81%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 99%, spam - 86%

Среднее гармоническое между presicion и recall (f1-score): not spam - 89%, spam - 92%



### 3) Анализ обучения моделей с выборкой 75%/25% с настройкой гиперпараметров

**Neural Network (MLP) performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.95 |     0.94 |     0.95 |      144
|            1 |      0.97 |     0.97 |     0.97 |      252
|     accuracy |           |          |     0.96 |      396
|    macro avg |      0.96 |     0.96 |     0.96 |      396
| weighted avg |      0.96 |     0.96 |     0.96 |      396

Общая точность определения среди всех параметров (accuracy): 96%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 95%, spam - 97%

Доля истинно положительных объектах во всех положительных (recall): not spam - 94%, spam - 97%

Среднее гармоническое между presicion и recall (f1-score): not spam - 95%, spam - 97%


**Logistic Regression performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.97 |     0.97 |     0.97 |      144
|            1 |      0.98 |     0.98 |     0.98 |      252
|     accuracy |           |          |     0.98 |      396
|    macro avg |      0.98 |     0.98 |     0.98 |      396
| weighted avg |      0.98 |     0.98 |     0.98 |      396

Общая точность определения среди всех параметров (accuracy): 98%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 97%, spam - 98%

Доля истинно положительных объектах во всех положительных (recall): not spam - 97%, spam - 98%

Среднее гармоническое между presicion и recall (f1-score): not spam - 97%, spam - 98%


**SVM performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.97 |     0.98 |     0.97 |      144
|            1 |      0.99 |     0.98 |     0.98 |      252
|     accuracy |           |          |     0.98 |      396
|    macro avg |      0.98 |     0.98 |     0.98 |      396
| weighted avg |      0.98 |     0.98 |     0.98 |      396

Общая точность определения среди всех параметров (accuracy): 98%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 97%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 98%, spam - 98%

Среднее гармоническое между presicion и recall (f1-score): not spam - 97%, spam - 98%


**Bernoulli Naive Bayes performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.93 |     0.94 |     0.93 |      144
|            1 |      0.97 |     0.96 |     0.96 |      252
|     accuracy |           |          |     0.95 |      396
|    macro avg |      0.95 |     0.95 |     0.95 |      396
| weighted avg |      0.95 |     0.95 |     0.95 |      396

Общая точность определения среди всех параметров (accuracy): 95%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 93%, spam - 97%

Доля истинно положительных объектах во всех положительных (recall): not spam - 94%, spam - 96%

Среднее гармоническое между presicion и recall (f1-score): not spam - 93%, spam - 96%



### 4) Анализ обучения моделей с выборкой 80%/20% с настройкой гиперпараметров

**Neural Network (MLP) performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.95 |     0.97 |     0.96 |      119
|            1 |      0.98 |     0.97 |     0.98 |      198
|     accuracy |           |          |     0.97 |      317
|    macro avg |      0.97 |     0.97 |     0.97 |      317
| weighted avg |      0.97 |     0.97 |     0.97 |      317

Общая точность определения среди всех параметров (accuracy): 97%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 95%, spam - 98%

Доля истинно положительных объектах во всех положительных (recall): not spam - 97%, spam - 97%

Среднее гармоническое между presicion и recall (f1-score): not spam - 96%, spam - 98%


**Logistic Regression performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.97 |     0.98 |     0.97 |      119
|            1 |      0.99 |     0.98 |     0.98 |      198
|     accuracy |           |          |     0.98 |      317
|    macro avg |      0.98 |     0.98 |     0.98 |      317
| weighted avg |      0.98 |     0.98 |     0.98 |      317

Общая точность определения среди всех параметров (accuracy): 98%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 97%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 98%, spam - 98%

Среднее гармоническое между presicion и recall (f1-score): not spam - 97%, spam - 98%


**SVM performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.97 |     0.98 |     0.97 |      119
|            1 |      0.99 |     0.98 |     0.98 |      198
|     accuracy |           |          |     0.98 |      317
|    macro avg |      0.98 |     0.98 |     0.98 |      317
| weighted avg |      0.98 |     0.98 |     0.98 |      317

Общая точность определения среди всех параметров (accuracy): 98%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 97%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 98%, spam - 98%

Среднее гармоническое между presicion и recall (f1-score): not spam - 97%, spam - 98%


**Bernoulli Naive Bayes performance:**
|              | precision |   recall | f1-score |  support
|--------------|-----------|----------|----------|---------
|            0 |      0.85 |     0.99 |     0.91 |      119
|            1 |      0.99 |     0.89 |     0.94 |      198
|     accuracy |           |          |     0.93 |      317
|    macro avg |      0.92 |     0.94 |     0.93 |      317
| weighted avg |      0.94 |     0.93 |     0.93 |      317

Общая точность определения среди всех параметров (accuracy): 93%

Доля истинно положительных объектов, среди всех распознанных как положительные (precision): not spam - 85%, spam - 99%

Доля истинно положительных объектах во всех положительных (recall): not spam - 99%, spam - 89%

Среднее гармоническое между presicion и recall (f1-score): not spam - 91%, spam - 94%


## Итоги

На основании данных эксперементов следует вывод в необходимости использования гиперпараметров при обучении моделей, а также грамотном распределении данных на обучающую и тестовую выборки.

Единственной моделью, не показавшей прироста при изменении выборки и(или) добавлении гипрепараметров, оказалась _Neural Network (MLP)_.

Модели _Logistic Regression_ и _SVM_ стабильно улучшались и достигли показателя в 98% общей точности.

_Bernoulli Naive Bayes_ при увеличении обучающей выборки показала уменьшение точности определения по тестовым данным примерно на 2%.

Краткая сводка минимальных и максимальных показателей обшей точности (accuracy) по результатам тестов:

|                         | Мин. показатели | Макс. показатели
|-------------------------|-----------------|-----------------
|  _Neural Network (MLP)_ |       0.96      |      0.97
|   _Logistic Regression_ |       0.94      |      0.98
|                   _SVM_ |       0.94      |      0.98
| _Bernoulli Naive Bayes_ |       0.91      |      0.95   

Таким образом моделями, наилучшим образом показавшими свои возможности в задаче классификации писем на спам/не спам, оказались _Logistic Regression_ и _SVM_ с показателем в 98% и приростом 4% в процессе обучения и настройки.
