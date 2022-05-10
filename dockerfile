FROM python:3.9.6

WORKDIR /app

COPY . .

RUN pip install Flask
RUN pip install pandas
RUN pip install matplotlib
RUN pip install sklearn

CMD ["python", "./notebooks/api.py"]