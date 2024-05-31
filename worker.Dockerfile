FROM python:3.10-slim

ENV APP_HOME /app
WORKDIR $APP_HOME

# Removes output stream buffering, allowing for more efficient logging
ENV PYTHONUNBUFFERED 1

# Install dependencies
COPY requirements.txt .
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy local code to the container image.
COPY . .

CMD exec python3 manage.py rqworker-pool default --num-workers 3
