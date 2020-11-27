# yolo2_tf2_coco80_plus_self_trained_model
Step by step build Yolo2 by TF2 with tunered model for fun
Yolo2 is very good example for developer to understand the object detection, then you may jump to the high level TF Object Detection API
#### Download the coco 80 classes pretrained weight first

- Download pretrained weights [here](https://pjreddie.com/media/files/yolov2.weights). Place this weights file in the directory "./models/yolo_v2/pretrained weights/"

#### if tuner new model, set the Model class instance at the bottom of yolo_v2.py

- the subclass function gonna create all kinds of folders for you

#### if train without GPU

-  uncomment the os.environ["CUDA_VISIBLE_DEVICES"]="-1"
-  comment out physical_devices = tf.config.experimental.list_physical_devices('GPU')
               tf.config.experimental.set_memory_growth(physical_devices[0], True)
               
#### Using pretrained coco weights to do prediction

- running_model(coco_y2 or other instance,
              create_model=True,
              model_load=False,
              load_tra_val_data=False,
              trainning=False,
              prediction=True)
- Set the SCORE_THRESHOLD & IOU_THRESHOLD at the beginning of yolo_v2.py
- First time need to use create_model load the pretrained weights, set create_model=True, model_load=False
-Then next time set create_model=False, model_load=True            
               
#### Train the model with your own sample, you need set up the subclass instance

- running_model(sugarbeet_y2 or other instance,
              create_model=True,
              model_load=False,
              load_tra_val_data=True,
              trainning=True,
              prediction=False)
- after you trained the model, next time set create_model=False, model_load=True
- Set the TRAIN_BATCH_SIZE,VAL_BATCH_SIZE,EPOCHS at the beginning of yolo_v2.py

#### Use trained model to make prediction

- running_model(sugarbeet_y2 or other instance,
              create_model=False,
              model_load=True,
              load_tra_val_data=False,
              trainning=False,
              prediction=True)
- when do prediction, put the image into the prediction input folder, no need to load_tra_val_data, load_tra_val_data is just for trainning. 

#### Credits

Many thanks to these great repositories:

https://github.com/experiencor/keras-yolo2

https://github.com/jmpap/YOLOV2-Tensorflow-2.0
