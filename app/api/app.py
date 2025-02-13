from fastapi import FastAPI

from app.api.view import iris_router

def create_app() -> FastAPI:
    app = FastAPI()
    
    app.include_router(iris_router, prefix="/iris")
    
    @app.get("/")
    async def landing_api():
        return {"api": [route.path for route in app.routes]}
    
    return app
    
    