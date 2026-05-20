FROM python:3.11-slim

RUN apt-get update && apt-get install -y \
        openmpi-bin \
        libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV N_RANKS=4
ENV SOURCE=newsgroups
ENV DOCS=500
ENV PORT=5555

EXPOSE 5555

CMD mpiexec --allow-run-as-root -n ${N_RANKS} \
    python server.py \
    --source ${SOURCE} \
    --docs   ${DOCS} \
    --port   ${PORT}
