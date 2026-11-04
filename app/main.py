from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1.document.router import DOCUMENT_ROUTER
import logging

from app.core.docs import register_docs
from app.core.logging import configure_logging, queue_logs_enabled

logger = logging.getLogger(__name__)
configure_logging(use_queue=queue_logs_enabled())

app = FastAPI(docs_url=None, title="Inzozi cafe backend")


app.include_router(DOCUMENT_ROUTER)
app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static",
)
register_docs(app)


@app.get("")
def main():
    return "Hello"
