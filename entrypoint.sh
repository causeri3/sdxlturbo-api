#!/bin/bash
set -e
cd /workspace/sdxlturbo-api
git pull
uvicorn api:app --host 0.0.0.0 --port 8000
