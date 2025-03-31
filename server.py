import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile
import io
from ultralytics import YOLO
from typing import List

app = FastAPI()

model = YOLO("yolo11n-seg.pt")


@app.get("/")
def health_check():
    return {"Hello": "World"}


@app.post("/inference")
async def run_inference(file: UploadFile = File(...)):
    print("Received file:", file.filename)
    image = Image.open(io.BytesIO(await file.read()))

    # run inference
    results = model.predict(image, iou=0.3)

    masks = results[0].masks.data.cpu().numpy()
    classes = results[0].boxes.cls.cpu().numpy().tolist()
    names = results[0].names
    classes = [names[int(cls)] for cls in classes]

    # convert mask to original image size
    scale_factor = results[0].orig_shape / masks.shape[1:]

    # area of each mask
    area = np.sum(masks, axis=(1, 2))

    return {"area": area.tolist(), "classes": classes}


@app.post("/batch_inference")
async def run_batch_inference(files: List[UploadFile] = File(...)):
    batch_size = 16
    images = []
    filenames = []

    for file in files:
        print("Received file:", file.filename)
        filenames.append(file.filename)
        images.append(Image.open(io.BytesIO(await file.read())))

    results_list = []
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        batch_filenames = filenames[i:i + batch_size]

        # run inference on batch
        results = model.predict(batch_images, iou=0.3)

        for j, result in enumerate(results):
            masks = result.masks.data.cpu().numpy()
            classes = result.boxes.cls.cpu().numpy().tolist()
            names = result.names
            classes = [names[int(cls)] for cls in classes]

            # area of each mask
            area = np.sum(masks, axis=(1, 2))

            # convert mask to original image size
            scale_xy = results[j].orig_shape[0] / masks.shape[1], results[j].orig_shape[1] / masks.shape[2]
            scale_factor = scale_xy[0] * scale_xy[1]
            area = area * scale_factor

            results_list.append({"filename": batch_filenames[j], "area": area.tolist(), "classes": classes})

    return {"results": results_list}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)