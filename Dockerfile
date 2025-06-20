# Its deployed on RunPod
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

# default RunPod working directory
WORKDIR /workspace

RUN git clone https://github.com/causeri3/sdxlturbo-api.git
WORKDIR /workspace/sdxlturbo-api

RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
