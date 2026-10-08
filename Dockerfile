FROM python:3.12-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY src/ src/
COPY api/ api/
COPY data/samples/ data/samples/
COPY configs/application_sample_split.json configs/application_sample_split.json
COPY configs/llm/demo.json configs/llm/demo.json
COPY scripts/bootstrap_workflow.py scripts/bootstrap_workflow.py
RUN python -m scripts.bootstrap_workflow

ENV TICKET_MODEL_PATH=models/transformer_demo.json
ENV TICKET_LLM_PATH=models/llm_demo.pt
ENV TICKET_REVIEW_DB=data/runtime/tickets.sqlite3
VOLUME ["/app/data/runtime"]
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--no-access-log"]
