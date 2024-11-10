import logging
import time
import traceback
import uvicorn
from fastapi import FastAPI, status, HTTPException, Request, Response
from loguru import logger
from pydantic import BaseModel

import logging_config as cfg
from main import Config, RAGSystem


logging.getLogger("uvicorn.access").addFilter(cfg.EndpointFilter())
cfg.add_loguru(logger)

config = Config()
rag_system = RAGSystem(config)

app = FastAPI()


class InputPromptModel(BaseModel):
    prompt: str
    
    
@app.get("/healthcheck")
def healthcheck():
    return Response(status_code=200)
    
    
@app.post("/getResponse")
async def get_response(input: InputPromptModel) -> str:
    start_time = time.time()
    try:
        context, docs, confidence = rag_system.get_relevant_content(input.prompt)
        response = rag_system.generate_response(input, context)
        logger.info(f"elapsed time: {(time.time() - start_time)} seconds")
        return response
    except (HTTPException, Exception) as exc:
        msg = traceback.format_exc()
        logger.error(f'function=get_response message="{str(msg)}"')
        return Response(msg, status_code=500)
    
if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=5000)
