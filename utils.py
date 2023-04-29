import av
import cv2 as cv
import numpy as np
import tensorflow as tf


VOC_CLASSES = [
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]


class ObjectDetector(tf.keras.Model):

    def __init__(self, model, **kwargs) -> None:
        super().__init__(**kwargs)
        self.model = model

    def compile(self, optimizer, clf_loss, reg_loss, **kwargs):
        super().compile(**kwargs)
        self.clf_loss = clf_loss
        self.reg_loss = reg_loss
        self.optimizer = optimizer

    def train_step(self, batch, **kwargs):
        x, y = batch
        with tf.GradientTape() as tape:
            y_hat = self.model(x, training=True)
            batch_clf_loss, batch_reg_loss = self.get_losses(y, y_hat)
            total_loss = 0.5*batch_clf_loss + batch_reg_loss
            grad = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grad, self.model.trainable_variables)
        )
        return {
            'total_loss': total_loss,
            'batch_clf_loss': batch_clf_loss,
            'batch_reg_loss': batch_reg_loss
        }

    def test_step(self, batch, **kwargs):
        x, y = batch
        y_hat = self.model(x, training=False)
        batch_clf_loss, batch_reg_loss = self.get_losses(y, y_hat)
        total_loss = 0.5*batch_clf_loss + batch_reg_loss
        return {
            'total_loss': total_loss,
            'batch_clf_loss': batch_clf_loss,
            'batch_reg_loss': batch_reg_loss
        }

    def get_losses(self, y, y_hat):
        return self.clf_loss(y[0], y_hat[0]), self.reg_loss(y[1], y_hat[1])

    def call(self, x, **kwargs):
        return self.model(x, **kwargs)


def partition(
    dataset: tf.data.Dataset, train: float = 0.8, test: float = 0.1, val: float = 0.1,
    shuffle: bool = True, shuffle_size: int = 1000, seed: int = 101
) -> tuple:
    assert(train + test + val == 1)

    if shuffle:
        dataset = dataset.shuffle(shuffle_size, seed)

    dataset_size = len(dataset)
    train_size = int(train * dataset_size)
    val_size = int(val * dataset_size)

    dataset_train = dataset.take(train_size)
    dataset_test = dataset.skip(train_size).skip(val_size)
    dataset_val = dataset.skip(train_size).take(val_size)

    return dataset_train, dataset_test, dataset_val


def preprocess_data(data: dict, image_size: int) -> tuple:
    image = data['image']
    bbox = data['objects']['bbox'][0]
    label = data['objects']['label'][0]
    # resize the image
    image = tf.image.resize(image, (image_size, image_size))
    # normalize the pixel values
    image = image / 255.0
    # one-hot labels
    label = tf.one_hot(label, depth=20)
    return image, (label, bbox)


def pipeline(data: tf.data.Dataset, batch_size: int = 32, image_size: int = 256) -> tf.data.Dataset:
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = data.map(lambda entry: preprocess_data(
        entry, image_size), num_parallel_calls=AUTOTUNE)
    data = data.shuffle(len(data))
    data = data.padded_batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    return data


def build_model(unfreezing: int = 5, image_size: int = 256):
    input_layer = tf.keras.layers.Input(shape=(image_size, image_size, 3))
    # base model
    vgg = tf.keras.applications.VGG16(include_top=False)
    # freeze the convolutional base
    for layer in vgg.layers[:-unfreezing]:
        layer.trainable = False
    vgg = vgg(input_layer)
    # classification: object category model
    f = tf.keras.layers.GlobalMaxPooling2D()(vgg)
    clf_in = tf.keras.layers.Dense(2048, activation='relu')(f)
    clf_out = tf.keras.layers.Dense(20, activation='softmax')(clf_in)
    # regression: bounding box model
    g = tf.keras.layers.GlobalMaxPooling2D()(vgg)
    reg_in = tf.keras.layers.Dense(2048, activation='relu')(g)
    reg_out = tf.keras.layers.Dense(4, activation='sigmoid')(reg_in)
    return tf.keras.Model(inputs=input_layer, outputs=(clf_out, reg_out))


def localization_loss(y, y_hat):
    delta_coord = tf.reduce_sum(tf.square(y[:, :2] - y_hat[:, :2]))
    delta_size = tf.reduce_sum(
        tf.square((y[:, 3] - y[:, 1]) - (y_hat[:, 3] - y_hat[:, 1])) +
        tf.square((y[:, 2] - y[:, 0]) - (y_hat[:, 2] - y_hat[:, 0]))
    )
    return delta_coord + delta_size


def frame_callback(frame, model: tf.keras.Model, image_size: int, threshold: float):
    if isinstance(frame, av.VideoFrame):
        frame = frame.to_ndarray(format="bgr24")

    h, w, _ = frame.shape
    # convert color space
    rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

    # process data
    resized = tf.image.resize(rgb, (image_size, image_size))
    normalized = np.expand_dims(resized / 255, 0)

    # detect
    labels, bboxs = model.predict(normalized)

    label_index = np.argmax(labels[0])
    label_confident = labels[0][label_index]
    label = VOC_CLASSES[label_index]

    bbox = bboxs[0] * ([h, w] * 2)

    # show results
    if label_confident > threshold:

        # control the bounding box
        cv.rectangle(
            frame,
            bbox[:2][::-1].astype(int),
            bbox[2:][::-1].astype(int),
            color=(255, 0, 0),
            thickness=2
        )

        # control the label
        cv.rectangle(
            frame,
            np.add(bbox[:2][::-1].astype(int), [0, -20]),
            np.add(bbox[:2][::-1].astype(int), [len(label)*13 + 5, 0]),
            color=(255, 0, 0),
            thickness=-1
        )

        # controls the text rendered
        cv.putText(
            frame,
            f"{label}: {label_confident:.2f}",
            np.add(bbox[:2][::-1].astype(int), [0, -5]),
            fontFace=cv.FONT_HERSHEY_SIMPLEX,
            fontScale=0.75,
            color=(255, 255, 255),
            thickness=2,
            lineType=cv.LINE_AA
        )
    return av.VideoFrame.from_ndarray(frame, format="bgr24")


def detect(source: any, model: tf.keras.Model, image_size: int = 256, threshold: float = 0.5, with_output: bool = False):
    captured = cv.VideoCapture(source)

    if with_output:
        success, frame = captured.read()
        h, w, _ = frame.shape
        fourcc = cv.VideoWriter_fourcc(*'mpv4')
        out = cv.VideoWriter("detected_video.mp4", fourcc, 20.0, (w, h))

    while captured.isOpened():

        success, frame = captured.read()

        if not success:
            print("WARNING: unable to read from video source")
            break

        frame_callback(frame, model, image_size, threshold)

        if with_output:
            out.write(frame)
        cv.imshow("object detection", frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    captured.release()
    cv.destroyAllWindows()
    