import pandas as pd
from sklearn.model_selection import train_test_split
import nltk
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import SnowballStemmer
nltk.download('punkt')
nltk.download('stopwords')
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import precision_score, recall_score, precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
import numpy as np
from sklearn.model_selection import GridSearchCV
import joblib

if __name__ == '__main__':
    # загружаем данные
    df = pd.read_csv("labeled.csv", sep=",")

    df["toxic"] = df["toxic"].apply(int)

    train_df, test_df = train_test_split(df, test_size=500)

    snowball = SnowballStemmer(language="russian")
    russian_stop_words = stopwords.words("russian")


    # def tokenize_sentence(sentence: str, remove_stop_words: bool = True):
    #     tokens = word_tokenize(sentence, language="russian")
    #     tokens = [i for i in tokens if i not in string.punctuation]
    #     if remove_stop_words:
    #         tokens = [i for i in tokens if i not in russian_stop_words]
    #     tokens = [snowball.stem(i) for i in tokens]
    #     return tokens

    # векторизуем посты с применением токинайзера
    vectorizer = TfidfVectorizer(tokenizer=word_tokenize)

    features = vectorizer.fit_transform(train_df["comment"])

    # обучаем модель логистической регрессии
    model = LogisticRegression(random_state=0)
    model.fit(features, train_df["toxic"])

    model_pipeline = Pipeline([

        ("vectorizer",

         TfidfVectorizer(tokenizer=word_tokenize)),

        ("model", LogisticRegression(random_state=0))
    ]
    )

    model_pipeline.fit(train_df["comment"], train_df["toxic"])

    # смотрим метрики на точность и полноту
    precision_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict(test_df["comment"]))

    recall_score(y_true=test_df["toxic"], y_pred=model_pipeline.predict(test_df["comment"]))

    prec, rec, thresholds = precision_recall_curve(y_true=test_df["toxic"],
                                                   probas_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1])

    # рисуем график
    plot_precision_recall_curve(estimator=model_pipeline, X=test_df["comment"], y=test_df["toxic"])

    # берём трешхолд по нижней границе в точности > 0.95
    ar1 = np.where(prec > 0.95)

    precision_score(y_true=test_df["toxic"],
                    y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds[ar1[0][0]])

    recall_score(y_true=test_df["toxic"],
                 y_pred=model_pipeline.predict_proba(test_df["comment"])[:, 1] > thresholds[ar1[0][0]])

    # ищем лучший коэффициент регуляризации
    grid_pipeline = Pipeline([
        ("vectorizer", TfidfVectorizer(tokenizer=word_tokenize)),
        ("model",
         GridSearchCV(
             LogisticRegression(random_state=0),
             param_grid={'C': [0.1, 1, 10.]},
             cv=3,
             verbose=4
         )
         )
    ])

    grid_pipeline.fit(train_df["comment"], train_df["toxic"])

    model_pipeline_c_10 = Pipeline([
        ("vectorizer", TfidfVectorizer(tokenizer=word_tokenize)),
        ("model", LogisticRegression(random_state=0, C=10.))
    ]
    )

    # обучаем модель с лучшими параметрами
    model_pipeline_c_10.fit(train_df["comment"], train_df["toxic"])

    prec_c_10, rec_c_10, thresholds_c_10 = precision_recall_curve(y_true=test_df["toxic"],
                                                                  probas_pred=model_pipeline_c_10.predict_proba(
                                                                      test_df["comment"])[:, 1])

    # так же смотрим с порогом в нижней границе точности > 0.95
    ar2 = np.where(prec_c_10 > 0.95)

    precision_score(y_true=test_df["toxic"],
                    y_pred=model_pipeline_c_10.predict_proba(test_df["comment"])[:, 1] > thresholds_c_10[ar2[0][0]])

    recall_score(y_true=test_df["toxic"],
                 y_pred=model_pipeline_c_10.predict_proba(test_df["comment"])[:, 1] > thresholds_c_10[ar2[0][0]])

    joblib.dump(model_pipeline_c_10, "./model_pipeline_c_10.joblib")
