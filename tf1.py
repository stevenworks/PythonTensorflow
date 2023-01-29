# Import the Tensorflow machine learning library under the alias "tf" and display the version number
import tensorflow as tf 
print(f"Tensorflow Version: {tf.__version__}")

# Grab the dataset and get training and testing splits (scaled to the interval [0, 1.0])
fashion_mnist = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
x_train, x_test = x_train/255.0, x_test/255.0 # Uses 255.0 as float to prevent unintentional truncation 

# Build model
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.Flatten(input_shape = (28, 28)))
model.add(tf.keras.layers.Dense(128, activation="relu"))
model.add(tf.keras.layers.Dropout(0.2))
model.add(tf.keras.layers.Dense(10))
model.compile(optimizer = "adam",
              loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics = ["accuracy"]
)

# Train model
model.fit(x_train, y_train, epochs = 5)
model.evaluate(x_test, y_test, verbose = 2)

# Build probability model
probabilityModel = tf.keras.models.Sequential()
probabilityModel.add(model)
probabilityModel.add(tf.keras.layers.Softmax())

probabilityModel(x_test[:5])
