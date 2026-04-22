from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
from matplotlib.pyplot import imread, imshow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions, preprocess_input
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten
from tensorflow.keras.regularizers import l1
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop
from efficientnet.tfkeras import EfficientNetB0
from tensorflow.keras import models
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout, Flatten, GlobalAveragePooling2D
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import l1
from tensorflow.keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
#Acne and Rosacea Photos


#Actinic Keratosis Basal Cell Carcinoma and other Malignant Lesions

#Atopic Dermatitis Photos

#Bullous Disease Photos
# Define your class names manually (in the same order as training)


# Define the class names
class_names = ["ambulence","fire_truck","other"]

# Initialize and set the LabelEncoder with the class names
label_encoder = LabelEncoder()
label_encoder.classes_ = np.array(class_names)

# Number of classes based on the class names
num_classes = len(label_encoder.classes_)

# Load EfficientNetB0 without the top classification layers
efficientnet_model = EfficientNetB0(input_shape=(100,100, 3), include_top=False, weights='efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5')

# Define the new model
inputs = efficientnet_model.input

# Get the output of the last convolutional layer for Grad-CAM
conv_output = efficientnet_model.layers[-1].output

# Add Global Average Pooling
x = GlobalAveragePooling2D()(conv_output)

# Add dense layers with regularization, batch normalization, and dropout
x = Dense(128, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(64, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

x = Dense(32, kernel_regularizer=l1(0.0001), activation='relu')(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)

# Add the final classification layer
outputs = Dense(units=num_classes, activation='softmax')(x)

# Combine into the model
model = models.Model(inputs=inputs, outputs=outputs)

# Compile the model
custom_optimizer = RMSprop(learning_rate=0.0001)
model.compile(optimizer=custom_optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Load the saved weights
from tensorflow.keras.models import load_model
model = load_model('efficientnet.h5')

print("Model loaded successfully!")



# Preprocess the image
def preprocess_single_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error loading image: {image_path}")
    
    # Resize the image to target size
    image = cv2.resize(image, (100,100))
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply CLAHE
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_enhanced = clahe.apply(gray)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(clahe_enhanced, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, threshold1=50, threshold2=150)
    
    # Create an RGB edge map (edges in red)
    edges_colored = np.zeros_like(image)
    edges_colored[:, :, 2] = edges
    
    # Overlay the edges onto the original image
    processed_image = cv2.addWeighted(image, 0.8, edges_colored, 0.5, 0)

    cv2.imwrite("static/output_image.png",processed_image)
    
    # Normalize the image (scaling pixel values between 0 and 1)
    processed_image = processed_image / 255.0
    
    return np.expand_dims(processed_image, axis=0)





def pred_emergency_vehicle(img_path):
# Path to the image
            image_path =img_path
            preprocessed_image = preprocess_single_image(image_path)

            # Make prediction
            predictions = model.predict(preprocessed_image)

            # Decode the predicted label
            predicted_label = label_encoder.inverse_transform([np.argmax(predictions)])
            confidence = np.max(predictions)

            print(f"Predicted Label: {predicted_label[0]}, Confidence: {confidence * 100:.2f}%")

            return predicted_label[0],(confidence)

#pred_emergency_vehicle("other_0_1.jpg")
#pred_sugar_cane("benign-familial-chronic-pemphigus-10.jpg")