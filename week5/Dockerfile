FROM svizor/zoomcamp-model:3.10.12-slim
RUN pip install pipenv

WORKDIR /app
COPY ['Pipefile', 'Pipfile.lock', './']

RUN pipenv install --system --deploy
COPY ['predict_client.py', 'predict_client_request.py', './']
EXPOSE 8000
ENTRYPOINT ["gunicorn", "--bind 0.0.0.0:8000", "predict_client:app"]