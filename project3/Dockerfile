#Image: jstubbs/ml-clothes-api

FROM python:3.11

RUN pip install Flask==3.0
RUN pip install tensorflow

COPY models /models
COPY api.py /api.py


CMD ["python", "api.py"]