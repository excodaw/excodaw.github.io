---
layout: single
title: "머신러닝 모델을 활용하여 웹 앱 만들기"
---


# 머신러닝 모델을 사용하여 웹 앱 만들기

NUFORC의 데이터베이스에서 얻은 데이터 셋에 대한 머신러닝 모델을 훈련하면서

- 훈련된 모델을 'Pickle'하는 법

- Flask 앱에서 모델을 사용하는 법

을 알아볼 것이다.

## 앱 만들기

앱을 만들기 전 다음과 같은 고려사항을 검토해야 한다
- 웹 앱인지 모바일 앱인지 
- 모델은 어디에 저장될 것인지
- 오프라인 동작을 지원할 것인지
- 모델을 훈련시키는데 어떤 기술이 사용될 것인지

또한 웹 브라우저에서 모델 자체를 학습 가능한 Flask웹 앱을 만들 수 있다. 이 작업은 JavaScript 컨택스트에서 TensorFlow.js를 사용하여 수행할 수 있다.

여기서는 Python 기반으로 모델을 학습할 것이기 때문에 Python에서 빌드한 웹 앱에서 읽을 수 있는 형식으로 내보내는데 필요한 단계를 살펴볼 것이다.

## 필요한 도구

작업을 위해서는 Flask와 Pickle이라는 두 가지 도구가 필요하며 둘 다 파이썬에서 실행 가능하다.

- Flask는 'micro-framework'로 정의되었다. 이것은 Python과 템플릿 엔진을 사용하여 웹 페이지를 빌드하는 웹 프레임 워크의 기본 기능을 제공한다.
- Pickle은 Python 객체 구조를 직렬화와 역직렬화하는 것이 가능한 Python 모듈이다. 모델을 'Pickle'하면 웹에서 사용할 수 있도록 구조를 직렬화하거나 병합한다.

  주의: Pickle은 안전하지 않으므로 파일을 'un-pickle' 하라는 메시지가 표시되면 주의해야 한다. pickle된 파일에는 `.pk1`이라는 접미사가 붙는다.

## 데이터 정리

[NUFORC](https://nuforc.org/)에서 수집 한 80,000 UFO 목격 데이터를 사용한다. 이 데이터에는 UFO 목격에 대한 몇 가지 흥미로운 설명이 있다.
- "한 남자가 밤에 풀밭에서 비추는 빛의 광선에서 나와 텍사스 인스트루먼트 주차장을 향해 달려갑니다"같은 긴 예제 설명이 존재한다
- "불빛이 우리를 쫓아다녔다" 같은 짧은 예제 설명이 존재한다.

[ufos.csv](https://github.com/codingalzi/ML-For-Beginners/blob/main/3-Web-App/1-Web-App/data/ufos.csv) 스프레드 시트에는 `city` `state` `country` `shape` `latitude` `longitude`같은 목격된 위치에 대한 정보가 열에 포함되어 있다.



1. 데이터를 가져온다. 


```python
import pandas as pd
import numpy as np

ufos = pd.read_csv('https://raw.githubusercontent.com/codingalzi/ML-For-Beginners/main/3-Web-App/1-Web-App/data/ufos.csv')
ufos.head()
```





  <div id="df-39abd0ab-526b-4f61-9dae-9582344638a2">
    <div class="colab-df-container">
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
      <th>datetime</th>
      <th>city</th>
      <th>state</th>
      <th>country</th>
      <th>shape</th>
      <th>duration (seconds)</th>
      <th>duration (hours/min)</th>
      <th>comments</th>
      <th>date posted</th>
      <th>latitude</th>
      <th>longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>10/10/1949 20:30</td>
      <td>san marcos</td>
      <td>tx</td>
      <td>us</td>
      <td>cylinder</td>
      <td>2700.0</td>
      <td>45 minutes</td>
      <td>This event took place in early fall around 194...</td>
      <td>4/27/2004</td>
      <td>29.883056</td>
      <td>-97.941111</td>
    </tr>
    <tr>
      <th>1</th>
      <td>10/10/1949 21:00</td>
      <td>lackland afb</td>
      <td>tx</td>
      <td>NaN</td>
      <td>light</td>
      <td>7200.0</td>
      <td>1-2 hrs</td>
      <td>1949 Lackland AFB&amp;#44 TX.  Lights racing acros...</td>
      <td>12/16/2005</td>
      <td>29.384210</td>
      <td>-98.581082</td>
    </tr>
    <tr>
      <th>2</th>
      <td>10/10/1955 17:00</td>
      <td>chester (uk/england)</td>
      <td>NaN</td>
      <td>gb</td>
      <td>circle</td>
      <td>20.0</td>
      <td>20 seconds</td>
      <td>Green/Orange circular disc over Chester&amp;#44 En...</td>
      <td>1/21/2008</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>10/10/1956 21:00</td>
      <td>edna</td>
      <td>tx</td>
      <td>us</td>
      <td>circle</td>
      <td>20.0</td>
      <td>1/2 hour</td>
      <td>My older brother and twin sister were leaving ...</td>
      <td>1/17/2004</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>4</th>
      <td>10/10/1960 20:00</td>
      <td>kaneohe</td>
      <td>hi</td>
      <td>us</td>
      <td>light</td>
      <td>900.0</td>
      <td>15 minutes</td>
      <td>AS a Marine 1st Lt. flying an FJ4B fighter/att...</td>
      <td>1/22/2004</td>
      <td>21.418056</td>
      <td>-157.803611</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-39abd0ab-526b-4f61-9dae-9582344638a2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-39abd0ab-526b-4f61-9dae-9582344638a2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-39abd0ab-526b-4f61-9dae-9582344638a2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




2. ufos 데이터를 새로운 제목의 작은 데이터 프레임으로 변환한다. 필드에서 `Country`의 고유 값을 확인한다.


```python
ufos = pd.DataFrame({'Seconds': ufos['duration (seconds)'], 'Country': ufos['country'],'Latitude': ufos['latitude'],'Longitude': ufos['longitude']})

ufos.Country.unique()
```




    array(['us', nan, 'gb', 'ca', 'au', 'de'], dtype=object)



이제 `null`값을 삭제하고 1 ~ 60초 사이의 목격만 추출하여 처리할 데이터의 양을 줄일 수 있다.


```python
ufos.dropna(inplace=True)

ufos = ufos[(ufos['Seconds'] >= 1) & (ufos['Seconds'] <= 60)]

ufos.info()
```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 25863 entries, 2 to 80330
    Data columns (total 4 columns):
     #   Column     Non-Null Count  Dtype  
    ---  ------     --------------  -----  
     0   Seconds    25863 non-null  float64
     1   Country    25863 non-null  object 
     2   Latitude   25863 non-null  float64
     3   Longitude  25863 non-null  float64
    dtypes: float64(3), object(1)
    memory usage: 1010.3+ KB
    

Scikit-learn의 `LabelEncoder` 라이브러리를 가져와 국가의 텍스트 값을 숫자로 변환한다. 해당 라이브러리는 데이터를 알파벳순으로 인코딩한다.


```python
from sklearn.preprocessing import LabelEncoder

ufos['Country'] = LabelEncoder().fit_transform(ufos['Country'])

ufos.head()
```

    /usr/local/lib/python3.7/dist-packages/ipykernel_launcher.py:3: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      This is separate from the ipykernel package so we can avoid doing imports until
    





  <div id="df-208f68b4-1828-4c3b-853f-5f0970a8f1b2">
    <div class="colab-df-container">
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
      <th>Seconds</th>
      <th>Country</th>
      <th>Latitude</th>
      <th>Longitude</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>20.0</td>
      <td>3</td>
      <td>53.200000</td>
      <td>-2.916667</td>
    </tr>
    <tr>
      <th>3</th>
      <td>20.0</td>
      <td>4</td>
      <td>28.978333</td>
      <td>-96.645833</td>
    </tr>
    <tr>
      <th>14</th>
      <td>30.0</td>
      <td>4</td>
      <td>35.823889</td>
      <td>-80.253611</td>
    </tr>
    <tr>
      <th>23</th>
      <td>60.0</td>
      <td>4</td>
      <td>45.582778</td>
      <td>-122.352222</td>
    </tr>
    <tr>
      <th>24</th>
      <td>3.0</td>
      <td>3</td>
      <td>51.783333</td>
      <td>-0.783333</td>
    </tr>
  </tbody>
</table>
</div>
      <button class="colab-df-convert" onclick="convertToInteractive('df-208f68b4-1828-4c3b-853f-5f0970a8f1b2')"
              title="Convert this dataframe to an interactive table."
              style="display:none;">

  <svg xmlns="http://www.w3.org/2000/svg" height="24px"viewBox="0 0 24 24"
       width="24px">
    <path d="M0 0h24v24H0V0z" fill="none"/>
    <path d="M18.56 5.44l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94zm-11 1L8.5 8.5l.94-2.06 2.06-.94-2.06-.94L8.5 2.5l-.94 2.06-2.06.94zm10 10l.94 2.06.94-2.06 2.06-.94-2.06-.94-.94-2.06-.94 2.06-2.06.94z"/><path d="M17.41 7.96l-1.37-1.37c-.4-.4-.92-.59-1.43-.59-.52 0-1.04.2-1.43.59L10.3 9.45l-7.72 7.72c-.78.78-.78 2.05 0 2.83L4 21.41c.39.39.9.59 1.41.59.51 0 1.02-.2 1.41-.59l7.78-7.78 2.81-2.81c.8-.78.8-2.07 0-2.86zM5.41 20L4 18.59l7.72-7.72 1.47 1.35L5.41 20z"/>
  </svg>
      </button>

  <style>
    .colab-df-container {
      display:flex;
      flex-wrap:wrap;
      gap: 12px;
    }

    .colab-df-convert {
      background-color: #E8F0FE;
      border: none;
      border-radius: 50%;
      cursor: pointer;
      display: none;
      fill: #1967D2;
      height: 32px;
      padding: 0 0 0 0;
      width: 32px;
    }

    .colab-df-convert:hover {
      background-color: #E2EBFA;
      box-shadow: 0px 1px 2px rgba(60, 64, 67, 0.3), 0px 1px 3px 1px rgba(60, 64, 67, 0.15);
      fill: #174EA6;
    }

    [theme=dark] .colab-df-convert {
      background-color: #3B4455;
      fill: #D2E3FC;
    }

    [theme=dark] .colab-df-convert:hover {
      background-color: #434B5C;
      box-shadow: 0px 1px 3px 1px rgba(0, 0, 0, 0.15);
      filter: drop-shadow(0px 1px 2px rgba(0, 0, 0, 0.3));
      fill: #FFFFFF;
    }
  </style>

      <script>
        const buttonEl =
          document.querySelector('#df-208f68b4-1828-4c3b-853f-5f0970a8f1b2 button.colab-df-convert');
        buttonEl.style.display =
          google.colab.kernel.accessAllowed ? 'block' : 'none';

        async function convertToInteractive(key) {
          const element = document.querySelector('#df-208f68b4-1828-4c3b-853f-5f0970a8f1b2');
          const dataTable =
            await google.colab.kernel.invokeFunction('convertToInteractive',
                                                     [key], {});
          if (!dataTable) return;

          const docLinkHtml = 'Like what you see? Visit the ' +
            '<a target="_blank" href=https://colab.research.google.com/notebooks/data_table.ipynb>data table notebook</a>'
            + ' to learn more about interactive tables.';
          element.innerHTML = '';
          dataTable['output_type'] = 'display_data';
          await google.colab.output.renderOutput(dataTable, element);
          const docLink = document.createElement('div');
          docLink.innerHTML = docLinkHtml;
          element.appendChild(docLink);
        }
      </script>
    </div>
  </div>




## 모델 빌드

이제 데이터를 훈련과 테스트 그룹으로 나누었기 때문에 모델을 훈련할 수 있다.

1. 학습하길 원하는 세 가지 특성을 가진 X 벡터와 y벡터를 선택한다. 입력 가능하고 국가 id를 리턴할 수 있다.


```python
from sklearn.model_selection import train_test_split

Selected_features = ['Seconds','Latitude','Longitude']

X = ufos[Selected_features]
y = ufos['Country']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

2. 로지스틱 회귀를 이용하여 모델을 훈련한다.


```python
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

print(classification_report(y_test, predictions))
print('Predicted labels: ', predictions)
print('Accuracy: ', accuracy_score(y_test, predictions))
```

                  precision    recall  f1-score   support
    
               0       1.00      1.00      1.00        41
               1       0.79      0.17      0.28       288
               2       1.00      0.90      0.95        10
               3       0.99      1.00      1.00       134
               4       0.95      1.00      0.97      4700
    
        accuracy                           0.95      5173
       macro avg       0.95      0.81      0.84      5173
    weighted avg       0.94      0.95      0.94      5173
    
    Predicted labels:  [4 4 4 ... 4 4 1]
    Accuracy:  0.9508988981248792
    

    /usr/local/lib/python3.7/dist-packages/sklearn/linear_model/_logistic.py:818: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      extra_warning_msg=_LOGISTIC_SOLVER_CONVERGENCE_MSG,
    

정확도는 95%정도로 나쁘지 않다. `Country`와 `LaLatitude/Longitude`간의 상관관계가 존재한다.

제작한 모델은 그 모델에서 a를 추론할 수 있어야하므로 혁신적이진 않지만 웹 응용 프로그램에서 이 모델을 정리 후 내보낸 다음 사용하는 원시 데이터에서 학습하는 것이 좋다.

### 모델 'Pickle'

이제 모델을 'Pickle'할 차례이다. Pickle된 모델을 로드하여 시간, 위도 및 경도 값을 포함하는 샘플 데이터 배열에 대해 테스트 해본다.


```python
import pickle
model_filename = 'ufo-model.pkl'
pickle.dump(model, open(model_filename,'wb'))

model = pickle.load(open('ufo-model.pkl','rb'))
print(model.predict([[50,44,-12]]))
```

    [3]
    

    /usr/local/lib/python3.7/dist-packages/sklearn/base.py:451: UserWarning: X does not have valid feature names, but LogisticRegression was fitted with feature names
      "X does not have valid feature names, but"
    

해당 모델은 영국의 국가 코드인 3을 반환한다.

### Flask 앱 만들기

Flask 앱을 빌드하여 모델을 호출하고 시각화 한다.
1. web-app이라는 폴더를 만들고 그 안에 notebook.ipynb, ufo-model.pkl파일을 넣는다
2. 이 폴더에서 세 개의 폴더를 만든다. 웹 앱 안 static 폴더 안 css안 templates이 있어야 한다.

```
web-app/
  static/
    css/
  templates/
notebook.ipynb
ufo-model.pkl
```

3. web-app 폴더에서 만드는 첫 번째 파일은 requirements.txt이다. 이 파일에는 앱에 필요한 종속성이 나열된다. requirements.txt에서 줄을 추가한다.

```
scikit-learn
pandas
numpy
flask
```

4. 이제 웹 앱으로 이동하여 이 파일을 실행한다.

```
cd web-app
```

5. 터미널 유형에서 요구 사항에 나열된 라이브러리를 설치하려면 다음을 줄이 필요하다

```
pip install -r requirements.txt
```

6. 이제 앱을 완료하기 위해 세 개의 파일을 더 만들어야 한다.
  
  - 루트에 app.py를 만든다
  - templates 디렉토리에 index.html을 만든다
  - static/css 디렉토리에 styles.css를 만든다

7. styles.css 파일을 구성한다

```css
body {
	width: 100%;
	height: 100%;
	font-family: 'Helvetica';
	background: black;
	color: #fff;
	text-align: center;
	letter-spacing: 1.4px;
	font-size: 30px;
}

input {
	min-width: 150px;
}

.grid {
	width: 300px;
	border: 1px solid #2d2d2d;
	display: grid;
	justify-content: center;
	margin: 20px auto;
}

.box {
	color: #fff;
	background: #2d2d2d;
	padding: 12px;
	display: inline-block;
}
```

8. 다음으로 index.html 파일을 구성한다.

```html
<!DOCTYPE html>
<html>
  <head>
    <meta charset="UTF-8">
    <title>🛸 UFO Appearance Prediction! 👽</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/styles.css') }}">
  </head>

  <body>
    <div class="grid">

      <div class="box">

        <p>According to the number of seconds, latitude and longitude, which country is likely to have reported seeing a UFO?</p>

        <form action="{{ url_for('predict')}}" method="post">
          <input type="number" name="seconds" placeholder="Seconds" required="required" min="0" max="60" />
          <input type="text" name="latitude" placeholder="Latitude" required="required" />
          <input type="text" name="longitude" placeholder="Longitude" required="required" />
          <button type="submit" class="btn">Predict country where the UFO is seen</button>
        </form>

        <p>{{ prediction_text }}</p>

      </div>

    </div>

  </body>
</html>
```

이 파일에서 앱에서 제공할 변수 주변의 `{{}}` 구문을 확인한다. 경로에 대한 예측을 게시하는 `/predict`양식도 있다.

9. `app.py` 파일을 다음과 같이 추가한다.

```python
import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)

model = pickle.load(open("./ufo-model.pkl", "rb"))


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():

    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = prediction[0]

    countries = ["Australia", "Canada", "Germany", "UK", "US"]

    return render_template(
        "index.html", prediction_text="Likely country: {}".format(countries[output])
    )


if __name__ == "__main__":
    app.run(debug=True)
```

`python app.py`나 `python3 app.py`로 웹 서버를 실행하거나 로컬에서 시작하면 UFO가 발견 된 위치에 대한 질문에 답변을 얻을 수 있다.

그 전에 `app.py`의 다음 부분을 살펴본다
1. 종속성이 먼저 로드되고 앱이 시작된다.
2. 그 다음 모델을 가져온다.
3. 그 다음 index.html이 홈 경로에서 랜더링 된다.


`/predict` 경로에서 양식이 게시될 때 몇 가지 일이 발생한다.
1. 양식 변수가 수집되어 numpy 배열로 변환된다. 그 다음 모델로 전송되고 예측이 반환된다
2. 표시할 국가는 예측된 국가 코드에서 읽을 수 있는 텍스트로 다시 랜더링되고 헤당 값은 인텍스로 다시 전송되어 html 템플릿에서 랜더링된다.

다음과 같은 웹-앱을 만들 수 있다.

![image.png](data:image/png;base64,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kExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkExMAgCQTEwCAJBMTAIAkOx9xdfPKkMAALgyZiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgmJgEASCYmAQBIJiYBAEgU8f8HtCOQFBtzXwAAAABJRU5ErkJggg==)

Flask와 Pickle된 모델을 사용하여 이러한 방식으로 모델을 사용하는 것은 비교적 간단하다. 가장 어려운 것은 예측을 얻기 위해 모델로 보내야 하는 데이터의 모양을 이해하는 것이다. 이 모든 것은 모델이 어떻게 훈련되었는지에 달려 있다.

웹-앱에 대한 파일은 [여기](https://github.com/excodaw/excodaw.github.io/tree/main/downloads/wep-app)에서 확인 가능하다
