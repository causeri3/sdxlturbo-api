FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /sdxlturbo-api
COPY . /sdxlturbo-api

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
