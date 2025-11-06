from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.models import load_model

# Rebuild your model architecture manually
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Load only weights (not full model)
try:
    old_model = load_model("MRI/model.h5", compile=False)
    model.set_weights(old_model.get_weights())
    print("✅ Weights successfully loaded into new model.")
except Exception as e:
    print("❌ Failed to load weights:", e)

# Save the fixed model
model.save("MRI/fixed_model.h5")
print("✅ New fixed model saved as MRI/fixed_model.h5")
