import time
from dl_utils import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import tensorflow as tf
# Run purely on CPU....
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"
physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

from tensorflow import keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Concatenate, concatenate, Dropout, LeakyReLU, Reshape, Activation, Conv2D, Input
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Flatten, Dense, Lambda, Add

# Parameters
# py_base_name = yolo_v2
py_full_name = os.path.basename(__file__)
py_base_name = py_full_name[:-3]
extensions = [".jpg", ".jpeg", ".png"]

IMAGE_H, IMAGE_W = 512, 512
GRID_H,  GRID_W  = 16, 16 # GRID size = IMAGE size / 32
BOX              = 5
ANCHORS          = [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828]
SCORE_THRESHOLD  = 0.3
IOU_THRESHOLD    = 0.3
TRAIN_BATCH_SIZE = 4
VAL_BATCH_SIZE   = 4
EPOCHS           = 30
LAMBDA_NOOBJECT  = 1
LAMBDA_OBJECT    = 5
LAMBDA_CLASS     = 1
LAMBDA_COORD     = 1
# max_annot        = 0

# anchors numpy.array
anchors = np.array(ANCHORS)
# anchors.shape = (5, 2)
anchors = anchors.reshape(len(anchors) // 2, 2)

# set a class instance at this py bottom for start...
class Yolo2model:
    def __init__(self, datasets_name):
        self.datasets_name = datasets_name

        pretrained_model_path = f"./models/{py_base_name}/{self.datasets_name}/pretrained/"
        if os.path.exists(pretrained_model_path) == False:
            os.makedirs(pretrained_model_path)
            print(f"{pretrained_model_path} has been created...")
        self.pretrained_model_path = pretrained_model_path

        updated_model_path = f"./models/{py_base_name}/{self.datasets_name}/updated/"
        if os.path.exists(updated_model_path) == False:
            os.makedirs(updated_model_path)
            print(f"{updated_model_path} has been created...")
        self.updated_model_path = updated_model_path

        chkpoint_path = f"./models/{py_base_name}/{self.datasets_name}/checkpoint/"
        if os.path.exists(chkpoint_path) == False:
            os.makedirs(chkpoint_path)
            print(f"{chkpoint_path} has been created...")
        self.model_chkpoint_path = chkpoint_path

        log_path = f"./logs/fit/{py_base_name}/{self.datasets_name}/"
        if os.path.exists(log_path) == False:
            os.makedirs(log_path)
            print(f"{log_path} has been created...")
        self.log_path = log_path

        self.pred_dir = f"./datasets/{py_base_name}/{self.datasets_name}/predict/"
        if os.path.exists(self.pred_dir) == False:
            os.makedirs(self.pred_dir)
            print(f"{self.pred_dir} has been created...")

        self.pred_input_path = f"{self.pred_dir}input/"
        if os.path.exists(self.pred_dir + "input/") == False:
            os.makedirs(self.pred_dir + "input/")
            print(f"{self.pred_dir} input/ has been created...")
        # e.g. pred_img_input_names is ['0004.jpg', '0051.jpg']
        self.pred_img_input_names = [f for f in os.listdir(self.pred_dir + "input/") if os.path.splitext(f)[1] in extensions]

        self.pred_output_path = f"{self.pred_dir}output/"
        if os.path.exists(self.pred_output_path) == False:
            os.makedirs(self.pred_output_path)
            print(f"{self.pred_output_path} has been created...")

        self.label_cls_len = len(self.label_cls_names())
        # random color pick for each label classes, list element is like (num1, num2, num3)
        self.color_list = generate_colors(self.label_cls_names())

    def label_cls_names(self):
        with open(f"./models/{py_base_name}/{self.datasets_name}/{self.datasets_name}_labels.txt", 'r') as file:
            # labels_data is a very long string with 80 labels
            labels_data = file.read()
            # Split the string into a tuple, 80 classes
        # class_names and LABELS are list with 80 strings elements
        self.class_names = list(filter(None, labels_data.split('\n')))
        # e.g. coco_y2 class_names and LABELS are list with 80 strings elements
        return self.class_names

class Y2_Diy(Yolo2model):
    def __init__(self, datasets_name):
        super().__init__(datasets_name)
        self.train_ann_dir = f"./datasets/{py_base_name}/{self.datasets_name}/train/annotation/"
        self.train_img_dir = f"./datasets/{py_base_name}/{self.datasets_name}/train/image/"
        self.val_ann_dir = f"./datasets/{py_base_name}/{self.datasets_name}/val/annotation/"
        self.val_img_dir = f"./datasets/{py_base_name}/{self.datasets_name}/val/image/"


def running_model(cls_ins, create_model=False, model_load=True, load_tra_val_data=False, trainning=False, prediction=False):
    # This is main def to control model operation
    #### Argument name can not be the same with globe function name ####
    if create_model == True:
        model = build_model(cls_ins)
        print(f"model has been built, saved it to {cls_ins.pretrained_model_path}")

    if model_load == True:
        model = model_loading(cls_ins)
        print(f"{py_base_name} {cls_ins.datasets_name} model has been loaded...")
        print(f"{py_base_name} {cls_ins.datasets_name} model has {cls_ins.label_cls_len} categories labels.")

    if load_tra_val_data == True:
        # tra_gen_len is the trainning datesets size
        tra_gen, val_gen, tra_gen_len, val_gen_len= load_tra_val_datasets(cls_ins)
        print(f"{cls_ins.datasets_name} datasets have been loaded....")

    if trainning == True:
        assert (load_tra_val_data == True), "must load the data first"
        training_data(cls_ins, model, tra_gen, val_gen, tra_gen_len, val_gen_len)
        print("trainning has been done")

    if prediction == True:
        assert (model_load == True), "must load the model first"
        prediction_display(cls_ins, model, SCORE_THRESHOLD, IOU_THRESHOLD)

# Custom Keras layer
# input tensor is (none, 38, 38, 64), after SpaceToDepth, shape would be (none, 19, 19, 256)
class SpaceToDepth(keras.layers.Layer):

    def __init__(self, block_size, **kwargs):
        super(SpaceToDepth, self).__init__(**kwargs)
        self.block_size = block_size

    def call(self, inputs):
        x = inputs
        batch, height, width, depth = K.int_shape(x)
        batch = -1
        reduced_height = height // self.block_size
        reduced_width = width // self.block_size
        y = K.reshape(x, (batch, reduced_height, self.block_size,
                          reduced_width, self.block_size, depth))
        z = K.permute_dimensions(y, (0, 1, 3, 2, 4, 5))
        t = K.reshape(z, (batch, reduced_height, reduced_width, depth * self.block_size ** 2))
        return t

    def compute_output_shape(self, input_shape):
        shape = (input_shape[0], input_shape[1] // self.block_size, input_shape[2] // self.block_size,
                 input_shape[3] * self.block_size ** 2)
        return tf.TensorShape(shape)


def build_model(cls_ins):

    input_image = tf.keras.layers.Input((IMAGE_H, IMAGE_W, 3), dtype='float32')
    # Layer 1
    x = Conv2D(32, (3,3), strides=(1,1), padding='same', name='conv_1', use_bias=False)(input_image)
    x = BatchNormalization(name='norm_1')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 2
    x = Conv2D(64, (3,3), strides=(1,1), padding='same', name='conv_2', use_bias=False)(x)
    x = BatchNormalization(name='norm_2')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 3
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_3', use_bias=False)(x)
    x = BatchNormalization(name='norm_3')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 4
    x = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_4', use_bias=False)(x)
    x = BatchNormalization(name='norm_4')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 5
    x = Conv2D(128, (3,3), strides=(1,1), padding='same', name='conv_5', use_bias=False)(x)
    x = BatchNormalization(name='norm_5')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 6
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_6', use_bias=False)(x)
    x = BatchNormalization(name='norm_6')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 7
    x = Conv2D(128, (1,1), strides=(1,1), padding='same', name='conv_7', use_bias=False)(x)
    x = BatchNormalization(name='norm_7')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 8
    x = Conv2D(256, (3,3), strides=(1,1), padding='same', name='conv_8', use_bias=False)(x)
    x = BatchNormalization(name='norm_8')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 9
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_9', use_bias=False)(x)
    x = BatchNormalization(name='norm_9')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 10
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_10', use_bias=False)(x)
    x = BatchNormalization(name='norm_10')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 11
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_11', use_bias=False)(x)
    x = BatchNormalization(name='norm_11')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 12
    x = Conv2D(256, (1,1), strides=(1,1), padding='same', name='conv_12', use_bias=False)(x)
    x = BatchNormalization(name='norm_12')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 13
    x = Conv2D(512, (3,3), strides=(1,1), padding='same', name='conv_13', use_bias=False)(x)
    x = BatchNormalization(name='norm_13')(x)
    x = LeakyReLU(alpha=0.1)(x)

    skip_connection = x

    x = MaxPooling2D(pool_size=(2, 2))(x)

    # Layer 14
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_14', use_bias=False)(x)
    x = BatchNormalization(name='norm_14')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 15
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_15', use_bias=False)(x)
    x = BatchNormalization(name='norm_15')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 16
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_16', use_bias=False)(x)
    x = BatchNormalization(name='norm_16')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 17
    x = Conv2D(512, (1,1), strides=(1,1), padding='same', name='conv_17', use_bias=False)(x)
    x = BatchNormalization(name='norm_17')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 18
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_18', use_bias=False)(x)
    x = BatchNormalization(name='norm_18')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 19
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_19', use_bias=False)(x)
    x = BatchNormalization(name='norm_19')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 20
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_20', use_bias=False)(x)
    x = BatchNormalization(name='norm_20')(x)
    x = LeakyReLU(alpha=0.1)(x)

    # Layer 21
    skip_connection = Conv2D(64, (1,1), strides=(1,1), padding='same', name='conv_21', use_bias=False)(skip_connection)
    skip_connection = BatchNormalization(name='norm_21')(skip_connection)
    skip_connection = LeakyReLU(alpha=0.1)(skip_connection)

    skip_connection = SpaceToDepth(block_size=2)(skip_connection)

    x = concatenate([skip_connection, x])

    # Layer 22
    x = Conv2D(1024, (3,3), strides=(1,1), padding='same', name='conv_22', use_bias=False)(x)
    x = BatchNormalization(name='norm_22')(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = Dropout(0.3)(x) # add dropout

    # Layer 23    e.g. for coco model cls_ins.label_cls_len = 80; for sugarbeet model cls_ins.label_cls_len = 2
    x = Conv2D(BOX * (4 + 1 + cls_ins.label_cls_len), (1, 1), strides=(1, 1), padding='same', name='conv_23')(x)
    output = Reshape((GRID_W, GRID_H, BOX, 4 + 1 + cls_ins.label_cls_len))(x)

    model = keras.models.Model(input_image, output, name=f'yolo_v2_{cls_ins.datasets_name}')

    ####Need to load the pretrained weights to the model, then save the model####
    class WeightReader:
        def __init__(self, weight_file):
            self.offset = 4
            self.all_weights = np.fromfile(weight_file, dtype='float32')

        def read_bytes(self, size):
            self.offset = self.offset + size
            return self.all_weights[self.offset - size:self.offset]

        def reset(self):
            self.offset = 4

    weight_reader = WeightReader(f"./models/{py_base_name}/pretrained weights/yolov2.weights")
    weight_reader.reset()
    nb_conv = 23

    for i in range(1, nb_conv + 1):
        conv_layer = model.get_layer('conv_' + str(i))
        conv_layer.trainable = True

        if i < nb_conv:
            norm_layer = model.get_layer('norm_' + str(i))
            norm_layer.trainable = True

            size = np.prod(norm_layer.get_weights()[0].shape)

            beta = weight_reader.read_bytes(size)
            gamma = weight_reader.read_bytes(size)
            mean = weight_reader.read_bytes(size)
            var = weight_reader.read_bytes(size)

            weights = norm_layer.set_weights([gamma, beta, mean, var])

        if len(conv_layer.get_weights()) > 1:  ## with bias
            bias = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[1].shape))
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel, bias])
        else:  ## without bias
            kernel = weight_reader.read_bytes(np.prod(conv_layer.get_weights()[0].shape))
            kernel = kernel.reshape(list(reversed(conv_layer.get_weights()[0].shape)))
            kernel = kernel.transpose([2, 3, 1, 0])
            conv_layer.set_weights([kernel])

    if (cls_ins.datasets_name == "coco"):
        pass

    ####Reset the last layer trainable with initial random value####
    elif (cls_ins.datasets_name == "sugarbeet" or "raccoon_kangaroo"):
        print(f"Building {cls_ins.datasets_name} updated model now....")
        layer = model.layers[-2]  # last convolutional layer
        layer.trainable = True
        weights = layer.get_weights()
        new_kernel = np.random.normal(size=weights[0].shape) / (GRID_H * GRID_W)
        new_bias = np.random.normal(size=weights[1].shape) / (GRID_H * GRID_W)
        layer.set_weights([new_kernel, new_bias])

    else:
        print("We can not build the model which matches the cls_ins.datasets_name, please check!")

    #### Save the Model ####
    model.save(cls_ins.pretrained_model_path)
    model.summary()
    print(f"{cls_ins.datasets_name} model has {cls_ins.label_cls_len} class labels")
    return model

def model_loading(cls_ins):

    if (cls_ins.datasets_name == "coco"):
        model = tf.keras.models.load_model(cls_ins.pretrained_model_path)

    elif (cls_ins.datasets_name == "sugarbeet" or "raccoon_kangaroo"):
        model = tf.keras.models.load_model(cls_ins.updated_model_path)
        print(f"!!!{cls_ins.datasets_name} updated model loaded!!!")

    else:
        print("We can not build the model which matches the cls_ins.datasets_name, please check!")

    model.summary()
    return model


def load_tra_val_datasets(cls_ins):

    def parse_function(img_obj, true_boxes):
        # x_img_string () <class 'EagerTensor'> <dtype: 'string'>
        x_img_string = tf.io.read_file(img_obj)
        # x_img.shape = (512, 512, 3) <class 'EagerTensor'> <dtype: 'uint8'>
        x_img = tf.image.decode_png(x_img_string, channels=3)  # dtype=tf.uint8
        # # resize(img, (hight, width))
        # x_img.shape = (512, 512, 3) <class 'EagerTensor'> <dtype: 'float32'>
        # pixel value /255, dtype=tf.float32, channels : RGB
        # Images that are represented using floating point values are expected to have values in the range [0,1)
        x_img = tf.image.convert_image_dtype(x_img, tf.float32)
        x_img = tf.image.resize(x_img, (IMAGE_H, IMAGE_W))

        return x_img, true_boxes

    # dataset consists batches<type:tuple>
    # batch : tupple(images, annotations)
    # batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
    # batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    def get_dataset(imgs_path, true_boxes, batch_size):
        # build the datasets
        dataset = tf.data.Dataset.from_tensor_slices((imgs_path, true_boxes))
        print("len(imgs_path)", len(imgs_path))
        # because the sample is very small, we can shuffle all at one time
        dataset = dataset.shuffle(len(imgs_path))
        #  (if count is None or -1) is for the dataset be repeated indefinitely
        dataset = dataset.repeat()
        dataset = dataset.map(parse_function, num_parallel_calls=6)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(10)
        print('Dataset:')
        print('Images count: {}'.format(len(imgs_path)))
        print('Step per epoch: {}'.format(len(imgs_path) // batch_size))
        print('Images per epoch: {}'.format(batch_size * (len(imgs_path) // batch_size)))
        return dataset

    # check the train and val datasets sample
    def display_db_sample(dataset, cls_ins):

        for batch in dataset:
            img = batch[0][0]
            label = batch[1][0]
            img_H_size = K.int_shape(img)[0]

            fig_1 = plt.figure(1, figsize=(10, 10))
            # plt.subplots() is a function that returns a tuple containing a figure and axes object(s).
            ax1 = fig_1.add_subplot(111)
            # f, (ax1) = plt.subplots(1, figsize=(10, 10))
            ax1.imshow(img)
            font = ImageFont.truetype(font='font/arial.ttf',
                                      size=np.floor(3e-2 * img_H_size + 0.5).astype('int32'))
            ax1.set_title(f'{cls_ins.datasets_name} datasets sample Image Size = {img.shape}')

            for i in range(label.shape[0]):
                box = label[i, :]
                box = box.numpy()
                # left:box[0], top:box[1], right:box[2], bottom:box[3]
                x = box[0]
                y = box[1]
                w = box[2] - box[0]
                h = box[3] - box[1]
                # box[4] 0:label for no bounding box, class value starts from 1
                if int(box[4]) == 0:
                    pass
                else:
                    color = np.asarray(cls_ins.color_list[int(box[4]) - 1]) / 255.
                    label_text = cls_ins.class_names[int(box[4]) - 1]

                    xy_origin = np.array([x, y - 3])
                    plt.annotate(label_text, xy=xy_origin, color="black", fontsize=6,
                                 bbox=dict(boxstyle='round,pad=0.2', fc=color, alpha=0.99))

                    rect = patches.Rectangle((x, y), w, h, linewidth=2, edgecolor=color, facecolor='none')
                    ax1.add_patch(rect)
            plt.show()
            break

    # Build image ground truth in YOLO format from image true_boxes and anchors.
    def process_true_boxes(true_boxes, anchors, image_width, image_height):
        # true_boxes : tensor, shape (max_annot, 5), format : x1 y1 x2 y2 c, coords unit : image pixel
        # anchors : list [anchor_1_width, anchor_1_height, anchor_2_width, anchor_2_height...]
            # anchors coords unit : grid cell
        # image_width, image_height : int (pixels)

        # Returns
        # detector_mask : array, shape (GRID_W, GRID_H, anchors_count, 1)
            # 1 if bounding box detected by grid cell, else 0
        # matching_true_boxes : array, shape (GRID_W, GRID_H, anchors_count, 5)
            # Contains adjusted coords of bounding box in YOLO format
        # true_boxes_grid : array, same shape than true_boxes (max_annot, 5),
            # format : x, y, w, h, c, coords unit : grid cell

        # Bounding box in YOLO Format : x, y, w, h, c
        # x, y : center of bounding box, unit : grid cell
        # w, h : width and height of bounding box, unit : grid cell
        # c : label index

        scale = IMAGE_W / GRID_W  # scale = 32

        anchors_count = len(anchors) // 2
        anchors = np.array(anchors)
        anchors = anchors.reshape(len(anchors) // 2, 2)

        detector_mask = np.zeros((GRID_W, GRID_H, anchors_count, 1))
        matching_true_boxes = np.zeros((GRID_W, GRID_H, anchors_count, 5))

        # convert true_boxes numpy array -> tensor
        true_boxes = true_boxes.numpy()

        true_boxes_grid = np.zeros(true_boxes.shape)

        # convert bounding box coords and localize bounding box
        for i, box in enumerate(true_boxes):
            # convert box coords to x, y, w, h and convert to grids coord
            w = (box[2] - box[0]) / scale
            h = (box[3] - box[1]) / scale
            x = ((box[0] + box[2]) / 2) / scale
            y = ((box[1] + box[3]) / 2) / scale
            true_boxes_grid[i, ...] = np.array([x, y, w, h, box[4]])
            if w * h > 0:  # box exists
                # calculate iou between box and each anchors and find best anchors
                best_iou = 0
                best_anchor = 0
                for i in range(anchors_count):
                    # iou (anchor and box are shifted to 0,0)
                    intersect = np.minimum(w, anchors[i, 0]) * np.minimum(h, anchors[i, 1])
                    union = (anchors[i, 0] * anchors[i, 1]) + (w * h) - intersect
                    iou = intersect / union
                    if iou > best_iou:
                        best_iou = iou
                        best_anchor = i
                # localize box in detector_mask and matching true_boxes
                if best_iou > 0:
                    x_coord = np.floor(x).astype('int')
                    y_coord = np.floor(y).astype('int')
                    detector_mask[y_coord, x_coord, best_anchor] = 1
                    yolo_box = np.array([x, y, w, h, box[4]])
                    matching_true_boxes[y_coord, x_coord, best_anchor] = yolo_box
        return matching_true_boxes, detector_mask, true_boxes_grid

    # Ground truth batch generator from a yolo dataset, ready to compare with YOLO prediction in loss function.
    def ground_truth_generator(cls_ins, dataset):
        # Parameters
        # YOLO dataset. Generate batch:
            # batch : tupple(images, annotations)
            # batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
            # batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        # Returns
        # imgs : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
        # detector_mask : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 1)
            # 1 if bounding box detected by grid cell, else 0
        # matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
            # Contains adjusted coords of bounding box in YOLO format
        # class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
            # One hot representation of bounding box label
        # true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
            # true_boxes format : x, y, w, h, c, coords unit : grid cell
        for batch in dataset:
            # imgs
            imgs = batch[0]
            # true boxes
            true_boxes = batch[1]
            # matching_true_boxes and detector_mask
            batch_matching_true_boxes = []
            batch_detector_mask = []
            batch_true_boxes_grid = []

            for i in range(true_boxes.shape[0]):
                one_matching_true_boxes, one_detector_mask, true_boxes_grid = \
                    process_true_boxes(true_boxes[i], ANCHORS, IMAGE_W, IMAGE_H)

                batch_matching_true_boxes.append(one_matching_true_boxes)
                batch_detector_mask.append(one_detector_mask)
                batch_true_boxes_grid.append(true_boxes_grid)

            detector_mask = tf.convert_to_tensor(np.array(batch_detector_mask), dtype='float32')
            matching_true_boxes = tf.convert_to_tensor(np.array(batch_matching_true_boxes), dtype='float32')
            true_boxes_grid = tf.convert_to_tensor(np.array(batch_true_boxes_grid), dtype='float32')

            # class one_hot
            matching_classes = K.cast(matching_true_boxes[..., 4], 'int32')
            class_one_hot = K.one_hot(matching_classes, cls_ins.label_cls_len + 1)[:, :, :, :, 1:]
            class_one_hot = tf.cast(class_one_hot, dtype='float32')

            batch = (imgs, detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid)
            yield batch

    # Test generator pipeline
    def test_gen_pipe(tra_gen):
        # batch
        img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(tra_gen)

        # y
        matching_true_boxes = matching_true_boxes[0, ...]
        detector_mask = detector_mask[0, ...]
        class_one_hot = class_one_hot[0, ...]
        y = K.concatenate((matching_true_boxes[..., 0:4], detector_mask, class_one_hot), axis=-1)

        # display prediction (Yolo Confidence value)
        plt.figure()
        f, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 10))

        # img
        img = img[0, ...]

        ax1.imshow(img)
        ax1.set_title('Image')

        ax2.matshow((K.sum(y[:, :, :, 4], axis=2)))  # YOLO Confidence value
        ax2.set_title('Ground truth')
        ax2.xaxis.set_ticks_position('bottom')

        f.tight_layout()
        plt.show()


    if (cls_ins.datasets_name == "coco"):
        print("coco model has not datasets for trainning, please set the load_datasets = False")

    elif (cls_ins.datasets_name == "sugarbeet" or "raccoon_kangaroo"):
        # tra_imgs_path.shape is (100,) tra_imgs_path[0]="./datasets/yolo_v2/sugarbeet/train/image/X-10-0.png"
        # tra_true_boxes.shape is (100, 40, 5)
        tra_imgs_path, tra_true_boxes = parse_annotation(cls_ins.train_ann_dir, cls_ins.train_img_dir,
                                                         cls_ins.label_cls_names(), IMAGE_W, IMAGE_H)
        # val_true_boxes.shape is (100, 34, 5)
        val_imgs_path, val_true_boxes = parse_annotation(cls_ins.val_ann_dir, cls_ins.val_img_dir,
                                                         cls_ins.label_cls_names(), IMAGE_W, IMAGE_H)
        # tra_gen_len is the trainning datasets size (total trainning inputs numbers)
        tra_gen_len = len(tra_imgs_path)
        val_gen_len = len(val_imgs_path)
        # dataset consists batches<type:tuple>
        # batch : tupple(images, annotations)
        # batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        # batch[0] : <tf.Tensor: shape=(10, 512, 512, 3), dtype=float32
        # batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
        # batch[1] <tf.Tensor: shape=(10, 40, 5), dtype=float64
        tra_dataset = None
        tra_dataset = get_dataset(tra_imgs_path, tra_true_boxes, TRAIN_BATCH_SIZE)
        print(f'^----^--From {cls_ins.train_img_dir} loaded the tra_dataset----^----^')

        val_dataset = None
        val_dataset = get_dataset(val_imgs_path, val_true_boxes, VAL_BATCH_SIZE)
        print(f'^----^--From {cls_ins.val_img_dir} loaded the val_dataset----^----^')

        # using display_db_sample() check the sample when tf.datasets created
        # display_db_sample(tra_dataset, cls_ins)
        aug_tra_dataset = augmentation_generator(tra_dataset, IMAGE_W, IMAGE_H)

        # using display_db_sample() check the sample when tf.datasets created
        display_db_sample(aug_tra_dataset, cls_ins)

        ####Print batch in tf.dataset#### dataset will repeat indefinitly.
        # i = 0
        # for bacth in aug_tra_dataset:
            # i += 1
            # if i < 2:
                # print("bacth in aug_tra_dataset", bacth)
            # else:
                # break

        # batch now consists by below components.
        # imgs : images to predict. tensor (shape : batch_size, IMAGE_H, IMAGE_W, 3)
        # detector_mask : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 1)
            # 1 if bounding box detected by grid cell, else 0
        # matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
            # Contains adjusted coords of bounding box in YOLO format
        # class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
            # One hot representation of bounding box label
        # true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
            # true_boxes format : x, y, w, h, c, coords unit : grid cell
        tra_gen = ground_truth_generator(cls_ins, aug_tra_dataset)
        val_gen = ground_truth_generator(cls_ins, val_dataset)

        # Test generator pipeline
        # test_gen_pipe(tra_gen)

    else:
        print("We can not build the model which matches the cls_ins.datasets_name, please check!")

    return tra_gen, val_gen, tra_gen_len, val_gen_len



def training_data(cls_ins, model, tra_gen, val_gen, tra_gen_len, val_gen_len):

    # tra_gen, val_gen is the datasets generator

    # Calculate IOU between box1 and box2
    def iou(x1, y1, w1, h1, x2, y2, w2, h2):
        # Parameters
        # x, y : box center coords
        # w : box width
        # h : box height

        xmin1 = x1 - 0.5 * w1
        xmax1 = x1 + 0.5 * w1
        ymin1 = y1 - 0.5 * h1
        ymax1 = y1 + 0.5 * h1
        xmin2 = x2 - 0.5 * w2
        xmax2 = x2 + 0.5 * w2
        ymin2 = y2 - 0.5 * h2
        ymax2 = y2 + 0.5 * h2
        interx = np.minimum(xmax1, xmax2) - np.maximum(xmin1, xmin2)
        intery = np.minimum(ymax1, ymax2) - np.maximum(ymin1, ymin2)
        inter = interx * intery
        union = w1 * h1 + w2 * h2 - inter
        iou = inter / (union + 1e-6)
        return iou

    # Calculate YOLO V2 loss from prediction (y_pred) and ground truth tensors (detector_mask,
    # matching_true_boxes, class_one_hot, true_boxes_grid,)
    def yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes_grid, y_pred, info=False):
        # Parameters
        # detector_mask : tensor, shape (batch, size, GRID_W, GRID_H, anchors_count, 1)
            # 1 if bounding box detected by grid cell, else 0
        # matching_true_boxes : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, 5)
            # Contains adjusted coords of bounding box in YOLO format
        # class_one_hot : tensor, shape (batch_size, GRID_W, GRID_H, anchors_count, class_count)
            # One hot representation of bounding box label
        # true_boxes_grid : annotations : tensor (shape : batch_size, max annot, 5)
            # true_boxes_grid format : x, y, w, h, c (coords unit : grid cell)
        # y_pred : prediction from model. tensor (shape : batch_size, GRID_W, GRID_H, anchors count, (5 + labels count)
        # info : boolean. True to get some infox about loss value
        # Returns
        # loss : scalar
        # sub_loss : sub loss list : coords loss, class loss and conf loss : scalar

        # anchors tensor
        anchors = np.array(ANCHORS)
        anchors = anchors.reshape(len(anchors) // 2, 2)

        # grid coords tensor
        coord_x = tf.cast(tf.reshape(tf.tile(tf.range(GRID_W), [GRID_H]), (1, GRID_H, GRID_W, 1, 1)), tf.float32)
        coord_y = tf.transpose(coord_x, (0, 2, 1, 3, 4))
        coords = tf.tile(tf.concat([coord_x, coord_y], -1), [y_pred.shape[0], 1, 1, 5, 1])

        # coordinate loss
        pred_xy = K.sigmoid(y_pred[:, :, :, :, 0:2])  # adjust coords between 0 and 1
        pred_xy = (pred_xy + coords)  # add cell coord for comparaison with ground truth. New coords in grid cell unit
        pred_wh = K.exp(y_pred[:, :, :, :,
                        2:4]) * anchors  # adjust width and height for comparaison with ground truth. New coords in grid cell unit
        # pred_wh = (pred_wh * anchors) # unit : grid cell
        nb_detector_mask = K.sum(tf.cast(detector_mask > 0.0, tf.float32))
        xy_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(matching_true_boxes[..., :2] - pred_xy)) / (
                    nb_detector_mask + 1e-6)  # Non /2
        wh_loss = LAMBDA_COORD * K.sum(detector_mask * K.square(K.sqrt(matching_true_boxes[..., 2:4]) -
                                                                K.sqrt(pred_wh))) / (nb_detector_mask + 1e-6)
        coord_loss = xy_loss + wh_loss

        # class loss
        pred_box_class = y_pred[..., 5:]
        true_box_class = tf.argmax(class_one_hot, -1)
        # class_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        class_loss = K.sparse_categorical_crossentropy(target=true_box_class, output=pred_box_class, from_logits=True)
        class_loss = K.expand_dims(class_loss, -1) * detector_mask
        class_loss = LAMBDA_CLASS * K.sum(class_loss) / (nb_detector_mask + 1e-6)

        # confidence loss
        pred_conf = K.sigmoid(y_pred[..., 4:5])
        # for each detector : iou between prediction and ground truth
        x1 = matching_true_boxes[..., 0]
        y1 = matching_true_boxes[..., 1]
        w1 = matching_true_boxes[..., 2]
        h1 = matching_true_boxes[..., 3]
        x2 = pred_xy[..., 0]
        y2 = pred_xy[..., 1]
        w2 = pred_wh[..., 0]
        h2 = pred_wh[..., 1]
        ious = iou(x1, y1, w1, h1, x2, y2, w2, h2)
        ious = K.expand_dims(ious, -1)

        # for each detector : best ious between prediction and true_boxes (every bounding box of image)
        pred_xy = K.expand_dims(pred_xy, 4)  # shape : m, GRID_W, GRID_H, BOX, 1, 2
        pred_wh = K.expand_dims(pred_wh, 4)
        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half
        true_boxe_shape = K.int_shape(true_boxes_grid)
        true_boxes_grid = K.reshape(true_boxes_grid,
                                    [true_boxe_shape[0], 1, 1, 1, true_boxe_shape[1], true_boxe_shape[2]])
        true_xy = true_boxes_grid[..., 0:2]
        true_wh = true_boxes_grid[..., 2:4]
        true_wh_half = true_wh * 0.5
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half
        intersect_mins = K.maximum(pred_mins, true_mins)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
        intersect_maxes = K.minimum(pred_maxes, true_maxes)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 2
        intersect_wh = K.maximum(intersect_maxes - intersect_mins, 0.)  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, 1, 1
        true_areas = true_wh[..., 0] * true_wh[..., 1]  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = intersect_areas / union_areas  # shape : m, GRID_W, GRID_H, BOX, max_annot, 1
        best_ious = K.max(iou_scores, axis=4)  # Best IOU scores.
        best_ious = K.expand_dims(best_ious)  # shape : m, GRID_W, GRID_H, BOX, 1

        # no object confidence loss
        no_object_detection = K.cast(best_ious < 0.6, K.dtype(best_ious))
        noobj_mask = no_object_detection * (1 - detector_mask)
        nb_noobj_mask = K.sum(tf.cast(noobj_mask > 0.0, tf.float32))

        noobject_loss = LAMBDA_NOOBJECT * K.sum(noobj_mask * K.square(-pred_conf)) / (nb_noobj_mask + 1e-6)
        # object confidence loss
        object_loss = LAMBDA_OBJECT * K.sum(detector_mask * K.square(ious - pred_conf)) / (nb_detector_mask + 1e-6)
        # total confidence loss
        conf_loss = noobject_loss + object_loss

        # total loss
        loss = conf_loss + class_loss + coord_loss
        sub_loss = [conf_loss, class_loss, coord_loss]

        #     # 'triple' mask
        #     true_box_conf_IOU = ious * detector_mask
        #     conf_mask = noobj_mask * LAMBDA_NOOBJECT
        #     conf_mask = conf_mask + detector_mask * LAMBDA_OBJECT
        #     nb_conf_box  = K.sum(tf.to_float(conf_mask  > 0.0))
        #     conf_loss = K.sum(K.square(true_box_conf_IOU - pred_conf) * conf_mask)  / (nb_conf_box  + 1e-6)
        #     # total loss
        #     loss = conf_loss /2. + class_loss + coord_loss /2.
        #     sub_loss = [conf_loss /2., class_loss, coord_loss /2.]

        if info:
            print('conf_loss   : {:.4f}'.format(conf_loss))
            print('class_loss  : {:.4f}'.format(class_loss))
            print('coord_loss  : {:.4f}'.format(coord_loss))
            print('    xy_loss : {:.4f}'.format(xy_loss))
            print('    wh_loss : {:.4f}'.format(wh_loss))
            print('--------------------')
            print('total loss  : {:.4f}'.format(loss))

            # display masks for each anchors
            for i in range(len(anchors)):
                f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
                f.tight_layout()
                f.suptitle('MASKS FOR ANCHOR {} :'.format(anchors[i, ...]))

                ax1.matshow((K.sum(detector_mask[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
                ax1.set_title(
                    'detector_mask, count : {}'.format(K.sum(tf.cast(detector_mask[0, :, :, i] > 0., tf.int32))))
                ax1.xaxis.set_ticks_position('bottom')

                ax2.matshow((K.sum(no_object_detection[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
                ax2.set_title('no_object_detection mask')
                ax2.xaxis.set_ticks_position('bottom')

                ax3.matshow((K.sum(noobj_mask[0, :, :, i], axis=2)), cmap='Greys', vmin=0, vmax=1)
                ax3.set_title('noobj_mask')
                ax3.xaxis.set_ticks_position('bottom')

        return loss, sub_loss

    num_epochs = EPOCHS
    steps_per_epoch_train = int(tra_gen_len/EPOCHS)
    steps_per_epoch_val = int(val_gen_len/EPOCHS)
    print("steps_per_epoch_train", int(tra_gen_len/EPOCHS))
    print("steps_per_epoch_val", int(val_gen_len/EPOCHS))
    best_val_loss = 1e6
    train_loss_history = []
    val_loss_history = []

    # optimizer
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    # checkpoint
    ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=optimizer, net=model)

    manager = tf.train.CheckpointManager(ckpt, cls_ins.model_chkpoint_path, max_to_keep=1)
    ckpt.restore(manager.latest_checkpoint)

    # Set up summary writers for tensorboard
    # log (tensorboard)
    summary_writer = tf.summary.create_file_writer(cls_ins.log_path, flush_millis=20000)
    summary_writer.set_as_default()

    # training
    for epoch in range(num_epochs):
        start = time.time()
        epoch_loss = []
        epoch_val_loss = []
        epoch_val_sub_loss = []
        print('Epoch {} :'.format(epoch))
        # train
        for batch_idx in range(steps_per_epoch_train):

            img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(tra_gen)

            with tf.GradientTape() as tape:
                y_pred = model(img)
                loss, sub_loss = yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes, y_pred)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            epoch_loss.append(loss)
            print('-', end='')
        print(' | ', end='')

        for batch_idx in range(steps_per_epoch_val):

            img, detector_mask, matching_true_boxes, class_one_hot, true_boxes = next(val_gen)

            with tf.GradientTape() as tape:
                y_pred = model(img)
                loss, sub_loss = yolov2_loss(detector_mask, matching_true_boxes, class_one_hot, true_boxes, y_pred)

                ckpt.step.assign_add(1)
                if int(ckpt.step) % 50 == 0:
                    save_path = manager.save()
                    print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))
                    print("loss {:1.2f}".format(loss.numpy()))

            epoch_val_loss.append(loss)
            epoch_val_sub_loss.append(sub_loss)
            print('-', end='')

        end = time.time()
        print("Total time: {:.1f}".format(end - start))

        loss_avg = np.mean(np.array(epoch_loss))
        val_loss_avg = np.mean(np.array(epoch_val_loss))
        sub_loss_avg = np.mean(np.array(epoch_val_sub_loss), axis=0)
        # train_loss_history.append(loss_avg)
        # val_loss_history.append(val_loss_avg)

        # log (tensorboard)
        # after the trainning, enter tensorboard --logdir="./logs/fit/yolo_v2/raccoon_kangaroo/" in the Terminal
        # the Terminal will prompt TensorBoard 2.3.0 at http://localhost:6006/
        # sub_loss = [conf_loss, class_loss, coord_loss]
        tf.summary.scalar('train_loss_avg', loss_avg, epoch)
        tf.summary.scalar('val_loss', val_loss_avg, epoch)
        tf.summary.scalar('val_conf_loss', sub_loss_avg[0], epoch)
        tf.summary.scalar('val_class_loss', sub_loss_avg[1], epoch)
        tf.summary.scalar('val_coord_loss', sub_loss_avg[2], epoch)

        # save
        if val_loss_avg < best_val_loss:
            model.save(cls_ins.updated_model_path)
            print(f"trainning model has been saved to {cls_ins.updated_model_path}")
            best_val_loss = val_loss_avg

        print(' loss = {:.4f}, val_loss = {:.4f} (conf={:.4f}, class={:.4f}, coords={:.4f})'.format(
            loss_avg, val_loss_avg, sub_loss_avg[0], sub_loss_avg[1], sub_loss_avg[2]))


def prediction_display(cls_ins, model, SCORE_THRESHOLD, IOU_THRESHOLD):

    # img_path_list = []
    for img_name in cls_ins.pred_img_input_names:
        img_path = f"{cls_ins.pred_input_path}/{img_name}"
        # img_path_list is a list includes all img path string
        # img_path_list.append(img_path)
        # Display predictions from YOLO model.
        # img_path : string of an image path.
        # model : YOLO model.
        # SCORE_THRESHOLD : threshold used for filtering predicted bounding boxes.
        # IOU_THRESHOLD : threshold used for non max suppression.
        #### OpenCV has colors in BGR (blue/green/red)####
        input_image = cv2.imread(img_path)
        image_data = cv2.resize(input_image, (IMAGE_H, IMAGE_W))
        # ::-1] flipping the color from BGR to RGB
        image_data = image_data[:, :, ::-1]
        image_data = image_data / 255.
        # plt.imshow(image_data)
        # plt.show()
        # image_data = np.array(image_data, dtype='float32')
        image_data = np.expand_dims(image_data, 0)
        # check the weights example
        # print(model.layers[75].get_weights()[0])
        # feats is the model last layer output, feats.shape is (m, 16, 16, 5, 85)
        ####feats last axis 85 = x+y+w+h+possiblity+80 classes####
        feats = model.predict(image_data)
        # check ou the output sample
        # print(feats[:1,7:8,9:10,:3,:5])

        # box_confidence : tensor    Probability estimate for whether each box contains any object.
        box_confidence = K.sigmoid(feats[..., 4:5])
        # box_class_pred : tensor    Probability distribution estimate for each box over class labels.
        box_class_probs = K.softmax(feats[..., 5:])
        # read the box_xy, box_wh original value
        box_xy = K.sigmoid(feats[..., :2])
        box_wh = K.exp(feats[..., 2:4])

        # K.shape(feats)[1:3] = (16, 16), convert dims to a float tensor!!
        conv_dims = K.cast_to_floatx(K.int_shape(feats)[1:3])
        conv_dims = K.reshape(conv_dims, [1, 1, 1, 1, 2])

        # After the tf.tile tf.Tensor=(
        # [ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7
        #   8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7
        #   8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7
        #   8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7
        #   8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7
        #   8  9 10 11 12 13 14 15  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15
        #   0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15], shape=(256,), dtype=int32)
        #### In YOLO the height index is the inner most iteration.####
        coord_x = tf.tile(tf.range(GRID_W), [GRID_H])
        coord_x = tf.reshape(coord_x, (1, GRID_H, GRID_W, 1, 1))
        # coord_x and coord_y shape is  (1, 16, 16, 1, 1)
        coord_x = tf.cast(coord_x, tf.float32)
        coord_y = tf.transpose(coord_x, (0,2,1,3,4))
        # after concat, coords shape is (1, 16, 16, 1, 2)
        conv_index = tf.concat([coord_x, coord_y], -1)
        # after tf.tile, coords shape is (1, 16, 16, 5, 2)
        conv_index = tf.tile(conv_index, [1, 1, 1, 5, 1])

        # Adjust preditions to each spatial grid point and anchor size.
        # Note: YOLO iterates over height index before width index.
        # box_xy and box_wh shape (m, GRID_W, GRID_H, Anchors, 2) = (1, 16, 16, 5, 2)
        box_xy = (box_xy + conv_index) / conv_dims
        box_wh = box_wh * anchors / conv_dims
        # print("box_xy", box_xy.shape, box_xy)

        # Reshape, remove the axis m
        # box_xy(16, 16, 5, 2)
        # box_wh(16, 16, 5, 2)
        # box_confidence(16, 16, 5, 1)
        # box_class_probs(16, 16, 5, 80)
        box_xy = box_xy[0,...]
        box_wh = box_wh[0,...]
        box_confidence = box_confidence[0,...]
        box_class_probs = box_class_probs[0,...]

        # Convert box coords from x,y,w,h to x1,y1,x2,y2 bounding box corners.
        box_xy1 = box_xy - 0.5 * box_wh
        box_xy2 = box_xy + 0.5 * box_wh
        # boxes shape (16, 16, 5, 4), last axis is box y_min, x_min, y_max, x_max
        boxes = K.concatenate((box_xy1, box_xy2), axis=-1)

        # Filter boxes
        # box_scores shape is (16, 16, 5, 80)
        box_scores = box_confidence * box_class_probs
        # box_classes shape (16, 16, 5)  sample like tf.Tensor(
        # [[[ 9 58 58 58 58]
        #   [ 9  9 58  9 58]
        #   [ 9  9  9  9 25]
        #   ...
        #   [ 9  9 58 58 58]
        #   [ 9  9 58 58 58]
        #   [ 9 58 58 58 58]]
        box_classes = K.argmax(box_scores, axis=-1) # best score index
        # box_class_scores shape is the same as box_scores, but shows the value
        box_class_scores = K.max(box_scores, axis=-1) # best score

        prediction_mask = box_class_scores >= SCORE_THRESHOLD
        # e.g. after the mask, boxes shape (6, 4) scores shape (6,) classes shape (6,)
        boxes = tf.boolean_mask(boxes, prediction_mask)
        scores = tf.boolean_mask(box_class_scores, prediction_mask)
        classes = tf.boolean_mask(box_classes, prediction_mask)

        # Scale box to image shape
        boxes = boxes * IMAGE_H

        # Non Max Supression
        # selected_idx output example tf.Tensor([1 5 0 3 2], shape=(5,), dtype=int32), delete the 4th boxes
        # max_output_size = 50  the maximum number of boxes to be selected by non-max suppression.
        selected_idx = tf.image.non_max_suppression(boxes, scores, 50, iou_threshold=IOU_THRESHOLD)

        boxes = K.gather(boxes, selected_idx)
        scores = K.gather(scores, selected_idx)
        classes = K.gather(classes, selected_idx)

        # Scales the boxes to be drawable by image ratio_list = [2.5, 1.40625] 1280/512, 720/512
        ratio_list = [input_image.shape[1]/IMAGE_W, input_image.shape[0]/IMAGE_H]
        boxes = scale_boxes(boxes, ratio_list)

        # Draw image
        # Print predictions info
        print('Found {} boxes for {}'.format(len(boxes), img_path))

        # Draw bounding boxes on the image file
        image_jpg = Image.open(img_path)
        draw_boxes(image_jpg, scores, boxes, classes, cls_ins.label_cls_names(), cls_ins.color_list)

        # Save the predicted bounding box on the image
        image_jpg.save(f"{cls_ins.pred_output_path}output_{img_name}", quality=90)

# model_load only loads pretrained model for coco_y2
coco_y2 = Yolo2model("coco")

# model_load only loads updated model for sugarbeet_y2
sugarbeet_y2 = Y2_Diy("sugarbeet")

# when creating a new class instance, the to-do list is as below
# 1. create {datasets_name}_labels.txt in the model directory
# 2. add one more elif with datasets name to build_model(cls_ins)
# 3. add one more elif with datasets name to model_loading(cls_ins)
# 4. add one more elif with datasets name to load_tra_val_datasets(cls_ins)

raccoon_kangaroo_y2 = Y2_Diy("raccoon_kangaroo")

# When do prediction, only need to load the model, prediction modual will load the data by itself
running_model(coco_y2,
              create_model=False,
              model_load=True,
              load_tra_val_data=False,
              trainning=False,
              prediction=True)


