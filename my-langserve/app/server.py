from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from langserve import add_routes
from my_app.chain import chain as my_app_chain

app = FastAPI()

add_routes(app, my_app_chain, path="/my-app")
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
