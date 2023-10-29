from ultralytics import YOLO
from time import perf_counter, strftime
from datetime import datetime
######################################### Main Training Script #######################################################

# Insert dataset path (the yaml file):
DATASET_YAML_PATH = "./datasets/FDD/fdd.yaml"

#Predict data path:
PREDICT_DIR =  '/'.join(DATASET_YAML_PATH.split('/')[:-1]) + "/test/images/"

# Choose the model (config.yaml or pretrained weights.pt or a checkpoint.pt):
MODEL_CONFIG = 'runs/detect/train/weights/best.pt'
#MODEL_CONFIG =  "yolov8s.pt"

### HYPERPARAMETERS:
EPOCHS = 10
BATCH_SIZE = 16
COSINE_LEARNING_RATE = True

##########
# choose what you want to do:
TRAIN = True # starts a trining process
EVAL =  True # do the model evaluation on val and test sets
PREDICT = True # Predict on the test images (interference on whole dataset takes time)

# If you want to print coco styled evaluation metrics:
COCO_EVAL = True  #( the dataset needs to have instances_set.json (a yolified coco format annotations))

###################
# Measure time
start_time = perf_counter()

# Load a model
model = YOLO(MODEL_CONFIG)

# Train the model:
if TRAIN:
    results = model.train(data=DATASET_YAML_PATH, epochs=EPOCHS,batch=BATCH_SIZE,save_json=True, cos_lr=COSINE_LEARNING_RATE)

if EVAL:
    # evaluate model performance on the validation set
    results = model.val(save_json=True,plots=True)


# predict on test images
if PREDICT:
    results = model(PREDICT_DIR,data=DATASET_YAML_PATH,save=True,show_labels=False)

# Measure time
end_time = datetime.fromtimestamp(perf_counter() - start_time).strftime("%H hours %M minutes %S seconds")
print(f"Yolo Script concluded without a fuss. It took {end_time}. Have a nice day! 😄")


#%%