import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

# создание сервиса и загрузка модели
app = FastAPI()

model = joblib.load("./model_pipeline_c_10.joblib")


class ToxicComments(BaseModel):
    """
    Ввод данных для модели
    """
    sentence: str

@app.post('/predict')
def predict(comment: ToxicComments):
    """
    :param comment: ввод данных из post request
    :return: предсказание комментария
    """
    features = [comment.sentence]
    prediction = model.predict(features).tolist()[0]
    return {
        "prediction": prediction
    }


if __name__ == '__main__':
    # Run server using given host and port
    uvicorn.run(app, host='127.0.0.1', port=80)
