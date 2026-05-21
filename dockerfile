FROM python:3.8

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app/app.py .
COPY models/transformer_classifier_7_layers.h5 .
COPY models/transformer_classifier_3_layers.h5 .
COPY models/transformer_classifier_5_layers.h5 .

EXPOSE 8051

CMD ["streamlit", "run", "app.py", "--server.port=8051", "--server.address=0.0.0.0"]


