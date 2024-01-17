FROM python:3.11-slim

WORKDIR /app

COPY . /app

RUN pip install poetry
RUN poetry config virtualenvs.create false
RUN poetry install --no-interaction --no-ansi --only main
RUN pip install -r requirements.txt
# RUN pip install huggingface_hub faiss-cpu geocoder
# RUN pip install transformers einops numpy
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118


CMD exec uvicorn app.server:app --host 0.0.0.0 --port $PORT
