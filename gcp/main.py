from google.cloud import storage
import tensorflow as tf
from PIL import Image
import numpy as np

BUCKET_NAME = "tomato_bucket1"

class_names = [
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight', 
 'Tomato___Late_blight', 
 'Tomato___Leaf_Mold',
 'Tomato___Mosaic_virus',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites_Two_spotted_spider_mite', 
 'Tomato___Target_Spot',
 'Tomato___Yellow_Leaf_Curl_virus', 
 'Tomato___healthy']

model = None

def download_blob(bucket_name, source_blob, dest_file):
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(source_blob)
    blob.download_to_filename(dest_file)

def predictgcp(request):
    global model
    if model is None:
        print("here1\n")
        download_blob(BUCKET_NAME, "models/crop.h5", "/tmp/crop.h5",)
        model = tf.keras.models.load_model("/tmp/crop.h5")
        print("here2\n")

    image = request.files["file"]

    image = np.array(Image.open(image).convert("RGB").resize((256,256)))
    image = image/255
    img_array = tf.expand_dims(image,0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100*(np.max(predictions[0])),2)

    return {"class":predicted_class, "confidence":confidence}
