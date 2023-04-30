import tensorflow as tf


def build_model(unfreezing: int = 5, image_size: int = 256):
    """
    Builds a TensorFlow Keras model for object detection.

    Args:
        unfreezing (int): Number of layers to unfreeze for fine-tuning.
        image_size (int): Size of input images.

    Returns:
        A compiled TensorFlow Keras model for object detection.
    """
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


class ObjectDetector(tf.keras.Model):
    """
    A custom TensorFlow Keras model for object detection.
    """

    def __init__(self, model: tf.keras.Model, **kwargs) -> None:
        """
        Initializes the ObjectDetector.

        Args:
            model (tf.keras.Model): A pre-trained TensorFlow Keras model.
        """
        super().__init__(**kwargs)
        self.model = model

    def compile(self, optimizer: tf.keras.optimizers.Optimizer, clf_loss: callable, reg_loss: callable, **kwargs) -> None:
        """
        Compiles the ObjectDetector.

        Args:
            optimizer (tf.keras.optimizers.Optimizer): An optimizer.
            clf_loss (callable): A classification loss function.
            reg_loss (callable): A regression loss function.
        """
        super().compile(**kwargs)
        self.clf_loss = clf_loss
        self.reg_loss = reg_loss
        self.optimizer = optimizer

    def train_step(self, batch: tuple, **kwargs) -> dict:
        """
        Defines a single training step.

        Args:
            batch (tuple): A tuple of (input, ground truth) data.

        Returns:
            dict: A dictionary containing the total loss, classification loss, and regression loss.
        """
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

    def test_step(self, batch: tuple, **kwargs) -> dict:
        """
        Defines a single evaluation step.

        Args:
            batch (tuple): A tuple of (input, ground truth) data.

        Returns:
            dict: A dictionary containing the total loss, classification loss, and regression loss.
        """
        x, y = batch
        y_hat = self.model(x, training=False)
        batch_clf_loss, batch_reg_loss = self.get_losses(y, y_hat)
        total_loss = 0.5*batch_clf_loss + batch_reg_loss
        return {
            'total_loss': total_loss,
            'batch_clf_loss': batch_clf_loss,
            'batch_reg_loss': batch_reg_loss
        }

    def get_losses(self, y: tf.Tensor, y_hat: tf.Tensor) -> tuple:
        """
        Computes the classification and regression loss.

        Args:
            y (tf.Tensor): The true labels for classification and regression.
            y_hat (tf.Tensor): The predicted labels for classification and regression.

        Returns:
            tuple: A tuple of the classification and regression loss.
        """
        return self.clf_loss(y[0], y_hat[0]), self.reg_loss(y[1], y_hat[1])

    def call(self, data: tf.data.Dataset, **kwargs) -> tf.keras.Model:
        """
        Calls the ObjectDetector.

        Args:
            data: Input data.

        Returns:
            tf.keras.Model: The output of the model.
        """
        return self.model(data, **kwargs)
