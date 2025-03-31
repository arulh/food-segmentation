## Setup

- Clone repo and download `yolo11n-seg.pt` into root of project.
- Server can be setup via docker or by installing requirements in vitrual environment.

## Report

The model used for the server is the default weights from the yolo11n-seg which contains all 80 classes from the COCO dataset (classes 46-55 are foods). The only paramater I changed on the model is the iou threshold because I noticed from my tests that there was quite a bit of overlap in the segmentations. The endpoint `/batch_inference` is designed to handle multiple requests and occupy as much resources on a given server as needed. The returned JSON structure contains a list of each image with their detections/segmentations and the corresponding area in pixels. I decided to set the batch size to 16 just to prevent any potential memory problems on my laptop.

In terms of scalablity, I implemented a very simple load balancer that distributes the images to seperate servers. The example I included in `client.py` uses two different servers on the local area network. The load balancer was designed to make use of horizantil scaling by dispatching the different server requests in parallel.

The server can be improved mainly by developing a model dedicated to foods, which can be accomplished by finetuning or training end-to-end on a food dataset. This will add more food classes to the model and remove the unwanted classes from the COCO dataset.