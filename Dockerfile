# One image for the API and the Streamlit frontend (SE). Data and results are mounted, not copied.
FROM python:3.11-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
# commit of the code, used to name the graph version (.git is not copied into the image)
ARG GIT_COMMIT=""
ENV GIT_COMMIT=$GIT_COMMIT
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
EXPOSE 8000 8501
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
