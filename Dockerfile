FROM python:3.6.7
COPY requirements.txt .
COPY app.py .
COPY state.pt .
COPY saved_model /saved_model
RUN pip install -r requirements.txt
EXPOSE 80
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]