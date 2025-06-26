import json
from pathlib import Path
from modal import Secret, asgi_app, Image
from vrag.app import app
from vrag.colpali import ColPaliModel

static_path = Path(__file__).with_name("frontend").joinpath("dist").resolve()

img = (
    Image.debian_slim(python_version="3.11")
    .pip_install(
        "openai==1.44.1",
        "opencv_python_headless==4.10.0.84",
        "pydantic==2.9.1",
        "pypdfium2==4.30.0",
        "fastapi==0.114.2",
        "qdrant_client==1.11.2",
        "sse-starlette==2.1.3",
    )
    .pip_install("numpy==2.1.1")
    .add_local_python_source("vrag")
    .add_local_dir(static_path, remote_path="/assets")
)

colpali = ColPaliModel()


@app.function(
    image=img,
    secrets=[Secret.from_dotenv()],
    concurrency_limit=1,
    container_idle_timeout=300,
    timeout=600,
    allow_concurrent_inputs=10,
)
@asgi_app()
def web():
    from uuid import UUID
    import uuid
    from pydantic import BaseModel
    from fastapi import FastAPI, UploadFile, File
    from sse_starlette.sse import EventSourceResponse, ServerSentEvent
    from fastapi.staticfiles import StaticFiles
    from fastapi.middleware.cors import CORSMiddleware
    from vrag.vrag import VRAG
    from vrag.qdrant_client import InMemoryQdrant

    class SearchRequest(BaseModel):
        query: str
        instance_id: UUID
        count: int = 3

    web_app = FastAPI()

    origins = [
        "http://localhost",
        "http://localhost:5173",
    ]

    web_app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    qdrant = InMemoryQdrant()

    VisionRAG = VRAG(colpali=colpali, qdrant=qdrant)

    @web_app.post("/collections")
    async def create_collection(files: list[UploadFile] = File(...)):
        name = str(uuid.uuid4())
        filenames = []
        byte_files = []

        async def read_files():
            for file in files:
                content = await file.read()
                filenames.append(file.filename or "file has no name")
                byte_files.append((name, file.filename or "file has no name", content))

        await read_files()

        async def event_generator():
            yield ServerSentEvent(
                data=json.dumps({"message": f"Indexing {len(byte_files)} files"})
            )
            for idx, byte_file in enumerate(byte_files):
                yield ServerSentEvent(
                    data=json.dumps(
                        {"message": f"Indexing file {idx + 1} / {len(byte_files)}"}
                    )
                )
                try:
                    async for state in VisionRAG.add_pdf(*byte_file):
                        yield state
                except Exception as e:
                    yield json.dumps({"error": str(e)})
            yield ServerSentEvent(
                data=json.dumps({"id": name, "filenames": filenames}), event="complete"
            )

        return EventSourceResponse(event_generator())

    @web_app.post("/search")
    async def search(query: SearchRequest):
        can_query = await qdrant.does_collection_exist(str(query.instance_id))

        async def event_generator():
            if not can_query:
                yield ServerSentEvent(
                    data=json.dumps(
                        {
                            "message": "The index has been deleted or does not exist. Please re-add the files."
                        }
                    )
                )
                return
            async for stage in VisionRAG.run_vrag(str(query.instance_id), query.query, query.count):
                yield stage

        return EventSourceResponse(event_generator())

    web_app.mount("/", StaticFiles(directory="/assets", html=True))
    return web_app
