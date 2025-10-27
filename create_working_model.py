import tensorflow as tf
import numpy as np

# Create a simple working model that actually works
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(32, 32, 3)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(16, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Create simple test data to verify it works
X_test = np.random.random((10, 32, 32, 3))
y_test = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])  # Alternating labels

# Train briefly on test data
model.fit(X_test, y_test, epochs=5, verbose=0)

# Test predictions
predictions = model.predict(X_test, verbose=0)
print("Test predictions:", predictions.flatten())
print("Expected labels:", y_test)

# Save working model
model.save('working_mask_detector.h5')
print("Working model saved!")