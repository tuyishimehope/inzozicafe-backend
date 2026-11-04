from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.v1 import router as api_v1
from app.core.docs import register_docs
from app.core.logging import configure_logging, queue_logs_enabled
from app.core.request_log import RequestLogMiddleware

configure_logging(use_queue=queue_logs_enabled())

app = FastAPI(title="Inzozi Cafe API", docs_url=None)
app.add_middleware(RequestLogMiddleware)
app.mount("/static", StaticFiles(directory="static"), name="static")
app.include_router(api_v1)
register_docs(app)


@app.get("/health", include_in_schema=False)
async def health() -> dict[str, str]:
    return {"status": "ok"}
