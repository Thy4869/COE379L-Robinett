#Image: jstubbs/ml-clothes-api

# Python version used: 3.10
FROM python:3.10

# Install the required packages
RUN pip install Flask==3.0
RUN pip install tensorflow==2.15

# Copy the files into the container
COPY models /models
COPY api.py /api.py

# Run the API
CMD ["python", "api.py"]