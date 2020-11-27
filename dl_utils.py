import os
import random
import colorsys
import skimage
import skimage.io
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
import PIL
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import matplotlib.patches as patches
import tensorflow.keras.backend as K
from skimage import io
import cv2
# check the damaged image avoiding error: Premature end of JPEG data
def verify_image(img_file):
    try:
        img = skimage.io.imread(img_file)
    except:
        return False, print(f"########Broken image name is {img_file}########")
    return True

def tensor_to_image(tensor):
    # tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)

# Yolo class color setting
def generate_colors(class_names):
    # class_name is a list consist with class name string
    hsv_tuples = [(x / len(class_names), 1., 1.) for x in range(len(class_names))]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
    colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), colors))
    random.seed(10101)  # Fixed seed for consistent colors across runs.
    random.shuffle(colors)  # Shuffle colors to decorrelate adjacent classes.
    random.seed(None)  # Reset seed to default.
    return colors

# Scales the predicted boxes in order to be drawable on the image
def scale_boxes(boxes, image_shape):

    height = image_shape[0]
    width = image_shape[1]
    image_dims = K.stack([height, width, height, width])
    # add one more dim
    image_dims = K.reshape(image_dims, [1, 4])
    boxes = boxes * image_dims
    return boxes

# Draw bounding boxes on the image file
# image is the original size input image <class 'PIL.JpegImagePlugin.JpegImageFile'>
# out_scores, out_boxes, out_classes are all numpy.array
# class_names is <class 'list'> lenth is 80
# colors is <class 'list'> lenth is 80
def draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors):
    font = ImageFont.truetype(font='font/arial.ttf',
                              size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
    thickness = (image.size[0] + image.size[1]) // 400

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]

        label = '{} {:.2f}'.format(predicted_class, score)

        draw = ImageDraw.Draw(image)
        label_size = draw.textsize(label, font)

        # top, left, bottom, right = box
        left, top, right, bottom = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
        print(label, (left, top), (right, bottom))

        if top - label_size[1] >= 0:
            text_origin = np.array([left, top - label_size[1]])
        else:
            text_origin = np.array([left, top + 1])

        # My kingdom for a good redistributable image drawing library.
        for i in range(thickness):
            draw.rectangle([left + i, top + i, right - i, bottom - i], outline=colors[c])
        draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=colors[c])
        draw.text(text_origin, label, fill=(0, 0, 0), font=font)
        del draw


# Yolo xml labels processing
def parse_annotation(ann_dir, img_dir, labels, IMAGE_W, IMAGE_H):

    # ann_dir : annotations files directory
    # img_dir : images files directory
    # labels : labels list

    # Returns
    # imgs_name : numpy array of images files path (shape : images count, 1)
    # e.g. imgs_name <class 'numpy.ndarray'> (100,)  such as "./datasets/yolo_v2/sugarbeet/train/image/X-10-0.png"
    # true_boxes : numpy array of annotations for each image (shape : image count, max annotation count, 5)
    # e.g true_boxes <class 'numpy.ndarray'> (100, 40, 5)
        # annotation format : xmin, ymin, xmax, ymax, class
        # xmin, ymin, xmax, ymax : image unit (pixel)
    # e.g true_boxes[0] shape is (40, 5), such as below
    # [[  1. 170.  75. 321.   1.]
    #  [112. 214. 232. 325.   1.]
    #  [256. 177. 370. 341.   1.]
    #  [452. 211. 512. 316.   1.]
    #  [466. 385. 488. 403.   2.]
    #  [166. 383. 180. 395.   2.]
    #  [  0.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0.   0.]
    #  [  0.   0.   0.   0.   0.]
    #     .... 40 rows.....

    # class = label index

    max_annot = 0
    imgs_name = []
    annots = []

    # Parse file
    for ann in sorted(os.listdir(ann_dir)):
        annot_count = 0
        boxes = []
        tree = ET.parse(ann_dir + ann)
        for elem in tree.iter():
            if 'filename' in elem.tag:
                imgs_name.append(img_dir + elem.text)
                verify_image(img_dir + elem.text)
            if 'width' in elem.tag:
                w = int(elem.text)
            if 'height' in elem.tag:
                h = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                box = np.zeros((5))
                for attr in list(elem):
                    if 'name' in attr.tag:
                        box[4] = int(labels.index(attr.text) + 1)  # 0:label for no bounding box
                    if 'bndbox' in attr.tag:
                        annot_count += 1
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                box[0] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                box[1] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                box[2] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                box[3] = int(round(float(dim.text)))
                            # Scales the boxes to 512*512 size by image ratio_list e.g.[512/1280, 512/720]
                    ratio_list = [IMAGE_W/w, IMAGE_H/h]
                    box[0:4] = tf.math.round(scale_boxes(box[0:4], ratio_list))
                boxes.append(np.asarray(box))

        # if w != IMAGE_W or h != IMAGE_H:
           # print('Image size error')
            # break

        annots.append(np.asarray(boxes))

        if annot_count > max_annot:
            max_annot = annot_count

        # Rectify annotations boxes : len -> max_annot
    imgs_name = np.array(imgs_name)
    true_boxes = np.zeros((imgs_name.shape[0], max_annot, 5))
    for idx, boxes in enumerate(annots):
        true_boxes[idx, :boxes.shape[0], :5] = boxes

    return imgs_name, true_boxes

def augmentation_generator(yolo_dataset, IMAGE_W, IMAGE_H):
    # Augmented batch generator from a yolo dataset
        # batch : tupple(images, annotations)
        # batch[0] : images : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
        # batch[1] : annotations : tensor (shape : batch_size, max annot, 5)
    # return - augmented batch : tensor (shape : batch_size, IMAGE_W, IMAGE_H, 3)
    for batch in yolo_dataset:
        # conversion tensor->numpy
        img = batch[0].numpy()
        boxes = batch[1].numpy()
        # conversion bbox numpy->ia object
        ia_boxes = []
        for i in range(img.shape[0]):
            ia_bbs = [ia.BoundingBox(x1=bb[0],
                                     y1=bb[1],
                                     x2=bb[2],
                                     y2=bb[3]) for bb in boxes[i]
                      if (bb[0] + bb[1] + bb[2] + bb[3] > 0)]
            ia_boxes.append(ia.BoundingBoxesOnImage(ia_bbs, shape=(IMAGE_W, IMAGE_H)))
        # data augmentation
        seq = iaa.Sequential([
            iaa.Fliplr(0.5),
            iaa.Flipud(0.5),
            iaa.Multiply((0.4, 1.6)),  # change brightness
            # iaa.ContrastNormalization((0.5, 1.5)),
            # iaa.Affine(translate_px={"x": (-100,100), "y": (-100,100)}, scale=(0.7, 1.30))
        ])
        # seq = iaa.Sequential([])
        seq_det = seq.to_deterministic()
        img_aug = seq_det.augment_images(img)
        img_aug = np.clip(img_aug, 0, 1)
        boxes_aug = seq_det.augment_bounding_boxes(ia_boxes)
        # conversion ia object -> bbox numpy
        for i in range(img.shape[0]):
            boxes_aug[i] = boxes_aug[i].remove_out_of_image().clip_out_of_image()
            for j, bb in enumerate(boxes_aug[i].bounding_boxes):
                boxes[i, j, 0] = bb.x1
                boxes[i, j, 1] = bb.y1
                boxes[i, j, 2] = bb.x2
                boxes[i, j, 3] = bb.y2
        # conversion numpy->tensor
        batch = (tf.convert_to_tensor(img_aug), tf.convert_to_tensor(boxes))
        # batch = (img_aug, boxes)
        yield batch

