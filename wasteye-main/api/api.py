from  fastapi import FastAPI

app = FastAPI()


@app.get("/")
def index ():
    #load a pretrained model
    return{"response ok" : True}

@app.get("/predict")
def predict(image_placeholder):
    # YOLO pretrained model
    return{'mode predicted..'}