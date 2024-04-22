from ultralytics import YOLO
from wandb.integration.ultralytics import add_wandb_callback
import wandb

wandb.init(project="yolo_train", job_type="training")
model_name = "yolov8n"
dataset_name = "data.yaml" # load the dataset and create a data.yaml file as shown below: 

'''train: ../train/images
val: ../valid/images
test: ../test/images

# nc: 1
# names: ['person']

names: 
  0: 'person'(base)'''

model = YOLO(f"{model_name}.pt")
add_wandb_callback(model, enable_model_checkpointing=True)
model.train(project="yolo_train", data=dataset_name, epochs=100, imgsz=640,single_cls=True)
model.val()
wandb.finish()


