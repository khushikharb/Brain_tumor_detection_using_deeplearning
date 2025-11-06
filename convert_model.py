from tensorflow.keras.models import load_model

# 1️⃣ Purani model file load karo (without compile)
model = load_model("MRI/model.h5", compile=False)

# 2️⃣ Nayi format (.keras) me save karo
model.save("MRI/model_converted.keras")

print("✅ Model converted successfully and saved as 'MRI/model_converted.keras'")
