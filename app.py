import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
import streamlit as st
from PIL import Image
import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from utils.helpers import validate_image_dir, load_sample_images


# Configuration
DATA_DIR = "data"
SOIL_IMAGE_DIR = os.path.join(DATA_DIR, "soil_images")
SOIL_CSV_PATH = os.path.join(DATA_DIR, "soil_params.csv")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# Soil properties mapping (expand with your actual data)
SOIL_PROPS = {
    'Black_Soil': {'N': 90, 'P': 42, 'K': 43, 'ph': 6.5},
    'Cinder_Soil': {'N': 50, 'P': 30, 'K': 35, 'ph': 7.2},
    'Laterite_Soil': {'N': 60, 'P': 55, 'K': 44, 'ph': 7.8},
    'Peat_Soil': {'N': 85, 'P': 38, 'K': 39, 'ph': 5.2},
    'Yellow_Soil': {'N': 70, 'P': 45, 'K': 40, 'ph': 6.8}
}

# Define the exact feature order to be used consistently
FEATURE_ORDER = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

def get_weather_data(district):
    # Replace with your actual OpenWeatherMap API key
    API_KEY = "a20a8808bfaa41934b9c893f48a6bd73"
    base_url = "http://api.openweathermap.org/data/2.5/weather"

    params = {
        'q': f"{district},IN",
        'appid': API_KEY,
        'units': 'metric'
    }

    try:
        response = requests.get(base_url, params=params)
        data = response.json()

        if response.status_code == 200:
            temperature = data['main']['temp']
            rainfall = data.get('rain', {}).get('1h', 0)  # Rainfall in the last hour
            return {
                'temperature': temperature,
                'rainfall': rainfall
            }
        else:
            st.error(f"Error fetching weather data: {data.get('message', 'Unknown error')}")
            return None
    except Exception as e:
        st.error(f"Exception occurred while fetching weather data: {e}")
        return None


class SoilAI:
    def __init__(self):
        self.cnn_model = None
        self.crop_model = None
        self.class_names = sorted(SOIL_PROPS.keys())
        
    def create_image_dataset(self):
        """Create TF Dataset from your soil image folders"""
        return tf.keras.utils.image_dataset_from_directory(
            SOIL_IMAGE_DIR,
            labels='inferred',
            label_mode='int',
            class_names=self.class_names,
            image_size=(256, 256),
            batch_size=32,
            shuffle=True,
            seed=42
        )
    
    def train_soil_cnn(self):
        """Train CNN model for soil classification"""
        train_ds = self.create_image_dataset()
        
        # Model architecture optimized for soil images
        model = tf.keras.Sequential([
            tf.keras.layers.Rescaling(1./255),
            tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(256, 256, 3)),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(64, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Conv2D(128, 3, activation='relu'),
            tf.keras.layers.MaxPooling2D(),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names))
        ])
        
        model.compile(
            optimizer='adam',
            loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        
        model.fit(train_ds, epochs=15)
        model.save(os.path.join(MODEL_DIR, "soil_cnn.h5"))
        return model
    
    def train_crop_model(self, df):
        """Train crop recommendation model with locked feature order"""
        # Validate all required features exist
        missing = [f for f in FEATURE_ORDER if f not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in training data: {missing}")
        
        X = df[FEATURE_ORDER]  # Use the locked feature order
        y = df['label']
        
        model = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
        model.fit(X, y)
        
        # Save feature order in the model object
        model.feature_names_ = FEATURE_ORDER
        
        joblib.dump(model, os.path.join(MODEL_DIR, "crop_rf.pkl"))
        return model
    
    def predict_soil_type(self, img_path):
        """Predict soil type from image"""
        img = tf.keras.utils.load_img(img_path, target_size=(256, 256))
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)
        
        predictions = self.cnn_model.predict(img_array)
        predicted_class = self.class_names[np.argmax(predictions[0])]
        confidence = tf.nn.softmax(predictions[0]).numpy().max()
        
        return predicted_class, confidence
    
    def recommend_crops(self, soil_type, env_params):
        """Generate crop recommendations with strict feature validation"""
        # Start with default soil properties
        params = SOIL_PROPS[soil_type].copy()
        params.update(env_params)
        
        # Validate all features exist
        missing = [f for f in FEATURE_ORDER if f not in params]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        
        # Create DataFrame with exact feature order
        X = pd.DataFrame([params])[FEATURE_ORDER]
        
        # Debug print (remove after verification)
        # st.write("Features being used:", X.columns.tolist())
        
        # Get predictions
        proba = self.crop_model.predict_proba(X)[0]
        return {crop: prob for crop, prob in zip(self.crop_model.classes_, proba)}

def set_background(image_path):
    with open(image_path, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        [data-testid="stSidebar"] > div:first-child {{
            background-color: rgba(255, 255, 255, 0); /* Make sidebar transparent */
        }}
        </style>
        """,
        unsafe_allow_html=True
    )


def main():
    st.set_page_config(page_title="Agri-Tech", layout="wide")

    # Set image background
    set_background("data/backimg.jpg")
    st.markdown(
        """
        <style>
        .stApp {
            color: #5C4033;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.title("🌱 Suggesting crops based on location and soil")

    # Initialize system
    ai = SoilAI()

    # Data validation
    if not validate_image_dir(SOIL_IMAGE_DIR):
        st.error("Soil image dataset structure invalid! Check console for details.")
        return

    # Model loading/training
    cnn_path = os.path.join(MODEL_DIR, "soil_cnn.h5")
    rf_path = os.path.join(MODEL_DIR, "crop_rf.pkl")

    if not (os.path.exists(cnn_path) and os.path.exists(rf_path)):
        with st.spinner("Training models for the first time..."):
            ai.cnn_model = ai.train_soil_cnn()
            df = pd.read_csv(SOIL_CSV_PATH)
            ai.crop_model = ai.train_crop_model(df)
    else:
        ai.cnn_model = tf.keras.models.load_model(cnn_path)
        ai.crop_model = joblib.load(rf_path)

    # UI Layout
    # Custom CSS for green sidebar
    # st.markdown(
    #     """
    #     <style>
    #     [data-testid="stSidebar"] {
    #         background-color: #90ee90; /* light green shade */
    #     }
    #     </style>
    #     """,
    #     unsafe_allow_html=True
    # )

    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to", ["Crop Recommender", "Dataset Info"])

    if page == "Crop Recommender":
        st.header("Smart Crop Recommendation")

        input_method = st.radio("Input Method:", ["Image Analysis", "Manual Input", "Image + Location", "Location"])

        if input_method == "Image Analysis":
            uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "jpeg", "png"])
            if uploaded_file:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(uploaded_file, caption="Uploaded Soil Sample", width=300)
                    temp_path = os.path.join(DATA_DIR, "temp_soil.jpg")
                    with open(temp_path, "wb") as f:
                        uploaded_file.seek(0)
                        f.write(uploaded_file.read())
                    soil_type, confidence = ai.predict_soil_type(temp_path)
                    st.success(f"*Identified Soil:* {soil_type.replace('_', ' ')}  \n*Confidence:* {confidence:.1%}")

                with col2:
                    st.subheader("Environmental Conditions")
                    temp = st.number_input("Temperature (°C)", 10.0, 40.0, 25.0)
                    rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)
                    humidity = st.number_input("Humidity (%)", 30.0, 100.0, 75.0)
                    if st.button("Get Recommendations"):
                        with st.spinner("Analyzing..."):
                            recommendations = ai.recommend_crops(
                                soil_type,
                                {'temperature': temp, 'rainfall': rainfall, 'humidity': humidity}
                            )
                            st.subheader("Top Crop Recommendations")
                            top_crops = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
                            for crop, prob in top_crops:
                                st.markdown(f"""
                                <div style="margin:10px 0;padding:10px;background:#90EE90;color:ffffff;border-radius:5px">
                                    <h4>{crop.capitalize()} <span style="float:right">{prob:.1%}</span></h4>
                                    <progress value="{prob}" max="1" style="width:100%"></progress>
                                </div>
                                """, unsafe_allow_html=True)

        elif input_method == "Manual Input":
            with st.form("manual_input"):
                st.subheader("Soil Parameters")
                col1, col2 = st.columns(2)

                with col1:
                    soil_type = st.selectbox("Soil Type", options=list(SOIL_PROPS.keys()), format_func=lambda x: x.replace('_', ' ')) 
                    n = st.number_input("Nitrogen (N)", 0, 100, SOIL_PROPS[soil_type]['N'])
                    p = st.number_input("Phosphorus (P)", 0, 100, SOIL_PROPS[soil_type]['P'])

                with col2:
                    k = st.number_input("Potassium (K)", 0, 100, SOIL_PROPS[soil_type]['K'])
                    ph = st.number_input("pH Level", 4.0, 9.0, SOIL_PROPS[soil_type]['ph'], format="%.1f")

                st.subheader("Environmental Conditions")
                temp = st.number_input("Temperature (°C)", 10.0, 40.0, 25.0)
                rainfall = st.number_input("Rainfall (mm)", 0.0, 500.0, 200.0)
                humidity = st.number_input("Humidity (%)", 30.0, 100.0, 75.0)

                if st.form_submit_button("Recommend Crops"):
                    recommendations = ai.recommend_crops(
                        soil_type,
                        {'N': n, 'P': p, 'K': k, 'ph': ph,
                         'temperature': temp, 'rainfall': rainfall, 'humidity': humidity}
                    )
                    st.subheader("Recommendation Analysis")
                    fig, ax = plt.subplots(figsize=(10, 6))
                    top_crops = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:10]
                    ax.barh([c[0].capitalize() for c in top_crops], [c[1] for c in top_crops], color=plt.cm.viridis(np.linspace(0, 1, len(top_crops))))
                    ax.set_xlabel('Suitability Probability')
                    ax.set_title('Top Recommended Crops')
                    st.pyplot(fig)

        elif input_method == "Image + Location":
            st.subheader("Upload Soil Image and Select Location")

            # Upload Soil Image
            uploaded_file = st.file_uploader("Upload Soil Image", type=["jpg", "jpeg", "png"])
            
            # Select District
            districts = [
                "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar",
                "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada",
                "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar",
                "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru",
                "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"
            ]
            district = st.selectbox("Select District", districts)

            if uploaded_file and district:
                # Display Uploaded Image
                st.image(uploaded_file, caption="Uploaded Soil Sample", use_container_width=300)

                # Save Uploaded Image Temporarily
                temp_image_path = os.path.join(DATA_DIR, "temp_soil.jpg")
                with open(temp_image_path, "wb") as f:
                    f.write(uploaded_file.read())

                # Predict Soil Type
                soil_type, confidence = ai.predict_soil_type(temp_image_path)
                st.success(f"*Identified Soil:* {soil_type.replace('_', ' ')}  \n*Confidence:* {confidence:.1%}")

                # Fetch Weather Data
                with st.spinner("Fetching weather data..."):
                    weather_data = get_weather_data(district)
                    if weather_data:
                        temperature = weather_data['temperature']
                        rainfall = weather_data['rainfall']
                        st.info(f"*Temperature:* {temperature} °C  \n*Rainfall:* {rainfall} mm")
                    else:
                        st.error("Failed to retrieve weather data.")
                        return

                # Recommend Crops
                if st.button("Get Recommendations"):
                    with st.spinner("Analyzing..."):
                        try:
                            recommendations = ai.recommend_crops(
                                soil_type,
                                {
                                    'temperature': temperature,
                                    'rainfall': rainfall,
                                    'humidity': 70  # Default humidity value; adjust as needed
                                }
                            )

                            # Display Top Crop Recommendations
                            st.subheader("Top Crop Recommendations")
                            top_crops = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
                            for crop, prob in top_crops:
                                st.markdown(f"""
                                <div style="margin:10px 0;padding:10px;background:#90EE90;border-radius:5px">
                                    <h4>{crop.capitalize()} <span style="float:right">{prob:.1%}</span></h4>
                                    <progress value="{prob}" max="1" style="width:100%"></progress>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error during recommendation: {e}")
        elif input_method == "Location":
            st.subheader("Predict Crop from Location Only")

            districts = [
                "Bagalkot", "Ballari", "Belagavi", "Bengaluru Rural", "Bengaluru Urban", "Bidar",
                "Chamarajanagar", "Chikkaballapur", "Chikkamagaluru", "Chitradurga", "Dakshina Kannada",
                "Davanagere", "Dharwad", "Gadag", "Hassan", "Haveri", "Kalaburagi", "Kodagu", "Kolar",
                "Koppal", "Mandya", "Mysuru", "Raichur", "Ramanagara", "Shivamogga", "Tumakuru",
                "Udupi", "Uttara Kannada", "Vijayapura", "Yadgir"
            ]
            district = st.selectbox("Select District", districts)

            if st.button("Predict Based on Location"):
                with st.spinner("Fetching weather data..."):
                    weather_data = get_weather_data(district)
                    if weather_data:
                        temperature = weather_data['temperature']
                        rainfall = weather_data['rainfall']
                        st.info(f"Temperature: {temperature} °C  \n*Rainfall:* {rainfall} mm")

                        # Let user select soil type manually or assume default
                        soil_type = st.selectbox("Select Soil Type", list(SOIL_PROPS.keys()), format_func=lambda x: x.replace('_', ' '))

                        try:
                            recommendations = ai.recommend_crops(
                                soil_type,
                                {
                                    'temperature': temperature,
                                    'rainfall': rainfall,
                                    'humidity': 70  # Assume default or fetch from elsewhere
                                }
                            )
                            st.subheader("Top Crop Recommendations")
                            top_crops = sorted(recommendations.items(), key=lambda x: x[1], reverse=True)[:5]
                            for crop, prob in top_crops:
                                st.markdown(f"""
                                <div style="margin:10px 0;padding:10px;background:#90EE90;border-radius:5px">
                                    <h4>{crop.capitalize()} <span style="float:right">{prob:.1%}</span></h4>
                                    <progress value="{prob}" max="1" style="width:100%"></progress>
                                </div>
                                """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error(f"Error during recommendation: {e}")
                    else:
                        st.error("Could not fetch weather data.")                
if __name__ == "__main__":
    main()