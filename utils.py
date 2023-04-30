import av
import cv2 as cv
import numpy as np
import tensorflow as tf


# a list of 20 object classes from the PASCAL VOC dataset
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


def partition(
    dataset: tf.data.Dataset, train: float = 0.8, test: float = 0.1, val: float = 0.1,
    shuffle: bool = True, shuffle_size: int = 1000, seed: int = 101
) -> tuple:
    """
    Partitions the given dataset into train, test, and validation sets.

    Args:
        dataset (tf.data.Dataset): The dataset to partition.
        train (float): The proportion of the dataset to use for training. Default is 0.8.
        test (float): The proportion of the dataset to use for testing. Default is 0.1.
        val (float): The proportion of the dataset to use for validation. Default is 0.1.
        shuffle (bool): Whether to shuffle the dataset before partitioning. Default is True.
        shuffle_size (int): The buffer size to use for shuffling. Default is 1000.
        seed (int): The random seed to use for shuffling. Default is 101.

    Returns:
        tuple: A tuple of three datasets, representing the train, test, and validation sets.
    """
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
    """
    Preprocesses the given data.

    Args:
        data (dict): A dictionary containing the data to preprocess.
        image_size (int): The size of the image to resize to.

    Returns:
        tuple: A tuple of an image tensor and a tuple of a one-hot encoded label tensor and a bounding box tensor.
    """
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
    """
    Builds a data pipeline for the given dataset.

    Args:
        data (tf.data.Dataset): The dataset to build the pipeline for.
        batch_size (int): The batch size to use. Default is 32.
        image_size (int): The size of the image to resize to. Default is 256.

    Returns:
        tf.data.Dataset: The transformed dataset.
    """
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    data = data.map(lambda entry: preprocess_data(
        entry, image_size), num_parallel_calls=AUTOTUNE)
    data = data.shuffle(len(data))
    data = data.padded_batch(batch_size)
    data = data.prefetch(AUTOTUNE)
    return data


def localization_loss(y: tf.Tensor, y_hat: tf.Tensor) -> float:
    """
    Computes the localization loss between the true bounding box and the predicted bounding box.

    Args:
        y (tf.Tensor): The true bounding box tensor.
        y_hat (tf.Tensor): The predicted bounding box tensor.

    Returns:
        float: The localization loss tensor.
    """
    delta_coord = tf.reduce_sum(tf.square(y[:, :2] - y_hat[:, :2]))
    delta_size = tf.reduce_sum(
        tf.square((y[:, 3] - y[:, 1]) - (y_hat[:, 3] - y_hat[:, 1])) +
        tf.square((y[:, 2] - y[:, 0]) - (y_hat[:, 2] - y_hat[:, 0]))
    )
    return delta_coord + delta_size


def frame_callback(frame: any, model: tf.keras.Model, image_size: int, threshold: float) -> av.VideoFrame:
    """
    A callback function to process each frame of a video.

    Args:
        frame (any): The current video frame.
        model (tf.keras.Model): The object detection model to use.
        image_size (int): The size of the image to resize to.
        threshold (float): The confidence threshold to use for object detection.

    Returns:
        av.VideoFrame: The processed video frame.
    """
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


def detect(source: any, model: tf.keras.Model, image_size: int = 256, threshold: float = 0.5, with_output: bool = False) -> None:
    """
    Detects objects in the specified video source using the given object detection model.

    Args:
        source (any): The video source to use.
        model (tf.keras.Model): The object detection model to use.
        image_size (int): The size of the image to resize to. Default is 256.
        threshold (float): The confidence threshold to use for object detection. Default is 0.5.
        with_output (bool): Whether to save the output video to a file. Default is False.
    """
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
