from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from vrag.vrag import VRAG

load_dotenv()

app = FastAPI()
VisionRAG = VRAG()


class SearchRequest(BaseModel):
    query: str


@app.post("/files")
async def create_upload_file(file: UploadFile):
    content = file.file.read()
    name = file.filename if file.filename else "default"
    await VisionRAG.add_pdf(name, content)
    return {"id": file.filename}


@app.post("/search")
async def search(query: SearchRequest):

    async def event_generator():
        async for stage in VisionRAG.run_vrag(query.query):
            yield stage

    return StreamingResponse(event_generator(), media_type="text/event-stream")
