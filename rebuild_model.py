from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# 1️⃣ Apna model architecture recreate karo
model_new = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(224,224,3)),
    MaxPooling2D(2,2),

    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# 2️⃣ Old model se sirf weights load karo
old_model = load_model("MRI/model.h5", compile=False)
model_new.set_weights(old_model.get_weights())

# 3️⃣ Nayi format me save karo
model_new.save("MRI/model_converted.keras")
print("✅ Model rebuilt and saved as MRI/model_converted.keras")
