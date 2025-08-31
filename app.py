import streamlit as st
import pandas as pd
from joblib import load

# Load the trained model
model = load('best_model.joblib')
print(type(model))
import sklearn
print(sklearn.__version__)

# Title and description
st.title("Mobile Price Range Classification")
st.write("""
This app predicts the price range of a mobile phone based on its features.
Enter the features below, and the model will predict if the price range is Low, Medium, High, or Very High.
""")

# Input fields for features
st.header("Input Mobile Features")

# Organize inputs in columns for better UI
col1, col2, col3 = st.columns(3)

with col1:
    battery_power = st.slider("Battery Power (mAh)", min_value=500, max_value=2000, value=1000, step=1)
    clock_speed = st.slider("Clock Speed (GHz)", min_value=0.5, max_value=3.0, value=1.5, step=0.1)
    fc = st.slider("Front Camera (MP)", min_value=0, max_value=20, value=5, step=1)
    int_memory = st.slider("Internal Memory (GB)", min_value=2, max_value=64, value=32, step=1)
    m_dep = st.slider("Mobile Depth (cm)", min_value=0.1, max_value=1.0, value=0.5, step=0.1)
    mobile_wt = st.slider("Mobile Weight (g)", min_value=80, max_value=200, value=140, step=1)

with col2:
    n_cores = st.slider("Number of Cores", min_value=1, max_value=8, value=4, step=1)
    px_height = st.slider("Pixel Height", min_value=0, max_value=2000, value=1000, step=1)
    px_width = st.slider("Pixel Width", min_value=500, max_value=2000, value=1000, step=1)
    ram = st.slider("RAM (MB)", min_value=256, max_value=4000, value=2000, step=1)
    sc_h = st.slider("Screen Height (cm)", min_value=5, max_value=19, value=12, step=1)
    sc_w = st.slider("Screen Width (cm)", min_value=5, max_value=18, value=6, step=1)

with col3:
    talk_time = st.slider("Talk Time (hours)", min_value=2, max_value=20, value=10, step=1)
    blue = 1 if st.checkbox("Bluetooth", value=False) else 0
    dual_sim = 1 if st.checkbox("Dual SIM", value=False) else 0
    four_g = 1 if st.checkbox("4G", value=False) else 0
    three_g = 1 if st.checkbox("3G", value=False) else 0
    touch_screen = 1 if st.checkbox("Touch Screen", value=False) else 0
    wifi = 1 if st.checkbox("WiFi", value=False) else 0

# Button to predict
if st.button("Predict Price Range"):
    # Feature engineering
    screen_size = sc_h * sc_w
    screen_area = px_height * px_width

    # Create input dictionary
    input_data = {
        'battery_power': battery_power,
        'blue': blue,
        'clock_speed': clock_speed,
        'dual_sim': dual_sim,
        'fc': fc,
        'four_g': four_g,
        'int_memory': int_memory,
        'm_dep': m_dep,
        'mobile_wt': mobile_wt,
        'n_cores': n_cores,
        'ram': ram,
        'talk_time': talk_time,
        'three_g': three_g,
        'touch_screen': touch_screen,
        'wifi': wifi,
        'screen_size': screen_size,
        'screen_area': screen_area
    }

    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    # Map prediction to label
    price_ranges = {0: 'Low', 1: 'Medium', 2: 'High', 3: 'Very High'}
    predicted_range = price_ranges.get(prediction, 'Unknown')

    # Display result
    st.success(f"The predicted price range is: **{predicted_range}**")