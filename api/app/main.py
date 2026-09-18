from fastapi import FastAPI

from routers.calculation import router as calculation_router

app = FastAPI()


app.include_router(calculation_router)


@app.get("/")
def root():
    return {"message": "API is running"}
