#Image: jstubbs/ml-clothes-api

FROM python:3.10

RUN pip install Flask==3.0
RUN pip install tensorflow==2.15

COPY models /models
COPY api.py /api.py


CMD ["python", "api.py"]