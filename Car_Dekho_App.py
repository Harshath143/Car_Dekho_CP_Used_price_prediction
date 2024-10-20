import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
loaded_model = joblib.load(r'C:\Users\harsh\OneDrive\Desktop\New folder\Car_price_mini model\Dataset\rf_trainedmodel.pkl')
loaded_encoders = joblib.load(r'C:\Users\harsh\OneDrive\Desktop\New folder\Car_price_mini model\Dataset\label_encoders.pkl')
loaded_scaler = joblib.load(r'C:\Users\harsh\OneDrive\Desktop\New folder\Car_price_mini model\Dataset\min.pkl')

categorical_columns = [
    "Body_type", "Transmission", "Original_equipment_manufacturer",
    "Model", "Variant_name", "Insurance_validity", "Fuel_type",
    "Colour", "Location"
]

@st.cache_resource
def predict_price(mileage, engine_displacement, year_of_manufacture,
                  transmission, fuel_type, owner_no, model_year,
                  location, kilometer_driven, body_type):
    input_data = pd.DataFrame({
        "Mileage": [mileage],
        "Engine_displacement": [engine_displacement],
        "Year_of_manufacture": [year_of_manufacture],
        "Transmission": [transmission],
        "Fuel_type": [fuel_type],
        "Owner_No.": [owner_no],
        "Model_year": [model_year],
        "Location": [location],
        "Kilometer_Driven": [kilometer_driven],
        "Body_type": [body_type],
    })

    for col in categorical_columns:
        if col not in {
            "Original_equipment_manufacturer", "Model", "Variant_name",
            "Insurance_validity", "Colour"
        }:
            le = loaded_encoders[col]
            input_data[col] = le.transform(input_data[col].astype(str))

    predicted_price = loaded_model.predict(input_data)
    predicted_price_norm = loaded_scaler.inverse_transform([[predicted_price[0]]])[0][0]
    return predicted_price_norm

# Streamlit UI Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="https://th.bing.com/th?id=OIP.mLh3isk2acupA_droYNDxgHaGL&w=273&h=228&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2",
    layout="wide"
)

# Custom CSS to apply the new background image
background_url = "https://wallpaperaccess.com/full/2529113.jpg"
st.markdown(
    f"""
    <style>
        .stApp {{
            background: url('{background_url}') no-repeat center center fixed;
            background-size: cover;
        }}
        .stApp > div:first-child {{
            background: transparent;
        }}
        .sidebar .sidebar-content {{
            background-color: rgba(138, 43, 226, 0.9);
            border-radius: 10px;
            padding: 20px;
            color: white;
        }}
        .price-display {{
            font-size: 36px;
            font-weight: bold;
            color: #FFD700;
            text-align: center;
            padding: 20px;
            border: 2px solid #FFD700;
            border-radius: 10px;
            background-color: rgba(0, 0, 0, 0.5);
            margin-top: 20px;
        }}
        .header {{
            display: flex;
            align-items: center;
            justify-content: flex-start;
            margin: 0 50px;
        }}
        .logo {{
            margin-right: 20px;
        }}
    </style>
    """,
    unsafe_allow_html=True
)

# Header with unique light violet and sky blue styling
st.markdown(
    """
    <style>
        .header {
            display: flex;
            align-items: center; /* Center vertically */
            justify-content: center; /* Center horizontally */
            background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%); /* Light violet gradient */
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1); /* Subtle shadow */
            animation: fadeIn 1.5s ease-in-out;
        }
        .logo {
            margin-right: 20px;
            width: 120px;
            height: 120px;
            border-radius: 15px; /* Square shape with rounded corners */
            border: 2px solid #00BFFF; /* Sky blue border */
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2); /* Slight shadow */
            transition: transform 0.3s ease-in-out;
        }
        .logo:hover {
            transform: scale(1.1); /* Zoom effect on hover */
        }
        h1 {
            color: #00BFFF; /* Sky blue color */
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            margin: 0;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2); /* Glow effect */
            animation: slideDown 1.2s ease-in-out;
        }
        @keyframes fadeIn {
            0% { opacity: 0; }
            100% { opacity: 1; }
        }
        @keyframes slideDown {
            0% { transform: translateY(-20px); opacity: 0; }
            100% { transform: translateY(0); opacity: 1; }
        }
    </style>
    <div class='header'>
        <img class='logo' src='https://st1.latestly.com/wp-content/uploads/2022/11/DCA047E4-E31D-4DF2-96C3-ECB1B6BE3D41.jpeg' alt='Car Logo' />
        <h1>Car Price Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Custom CSS to style sliders and inputs
st.markdown(
    """
    <style>
    /* Style the slider background to be transparent */
    .stSlider {
        background-color: rgba(255, 255, 255, 0.3); /* Adjust the opacity as needed */
        border-radius: 10px; /* Rounded corners */
    }
    
    /* Style the number input background to be transparent */
    .stNumberInput {
        background-color: rgba(255, 255, 255, 0.3); /* Adjust the opacity as needed */
        border: 2px solid #0072B1; /* Border color */
        border-radius: 5px; /* Rounded corners */
    }
    
    /* Style the selectbox background to be transparent */
    .stSelectbox, .stSelectSlider {
        background-color: rgba(255, 255, 255, 0.3); /* Adjust the opacity as needed */
        border: 2px solid #0072B1; /* Border color */
        border-radius: 5px; /* Rounded corners */
    }
    
    /* Style the header in the sidebar */
    .stSidebar h1, .stSidebar h2, .stSidebar h3 {
        color: #0072B1; /* Header color */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
st.sidebar.header("Input Features")
mileage = st.sidebar.number_input("Mileage (kmpl)", min_value=0.0, format="%.1f", step=0.1)
engine_displacement = st.sidebar.number_input("Engine Displacement (cc)", min_value=0)
year_of_manufacture = st.sidebar.number_input("Year of Manufacture", min_value=1900, max_value=2024)
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel"])
owner_no = st.sidebar.number_input("Owner No.", min_value=0)
model_year = st.sidebar.number_input("Model Year", min_value=1900, max_value=2024)
location = st.sidebar.selectbox("Location", ["Chennai", "Bangalore", "Delhi", "Kolkata", "Jaipur", "Hyderabad"])
body_type = st.sidebar.selectbox("Body Type", [
    "Hatchback", "SUV", "Sedan", "MUV", "Minivans", "Coupe",
    "Pickup Trucks", "Convertibles", "Hybrids", "Wagon",
    "Crossover", "Station Wagon"
])
kilometer_driven = st.sidebar.number_input("Kilometer Driven", min_value=0)


# Prediction button and result display with advanced styling and animation
if st.sidebar.button("Estimate Used Car Price"):
    with st.spinner('🔍 Analyzing inputs and estimating price...'):
        try:
            # Make the prediction
            predicted_price = predict_price(
                mileage, engine_displacement, year_of_manufacture,
                transmission, fuel_type, owner_no, model_year,
                location, kilometer_driven, body_type
            )

            # Conditional styling based on the predicted price
            if predicted_price < 300000:
                color = "#32CD32"  # Lime Green
                feedback = "🚗 **This is a budget-friendly car!**"
            elif predicted_price <= 1000000:
                color = "#FFA500"  # Orange
                feedback = "🤑 **This seems like a good deal!**"
            else:
                color = "#FF4500"  # Red
                feedback = "💎 **This is a premium car!**"

            # Display the estimated price with vibrant formatting
            st.markdown(
                f"""
                <div class='price-display' style='border-color: {color}; color: {color};'>
                    **Estimated Price: ₹{predicted_price:,.2f}**
                </div>
                <p style='text-align: center; font-size: 18px; color: {color};'>
                    {feedback}
                </p>
                """,
                unsafe_allow_html=True
            )

            # Stylish Summary of Inputs
            st.markdown(
                f"""
                <div style='padding: 15px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%); 
                            color: white; border-radius: 12px; margin-top: 20px;'>
                    <h3 style='text-align: center; margin-bottom: 15px;'>✨ Summary of Your Input ✨</h3>
                    <ul style='list-style: none; padding: 0; font-size: 16px;'>
                        <li>📅 <strong>Model Year:</strong> {model_year}</li>
                        <li>⚙️ <strong>Transmission:</strong> {transmission}</li>
                        <li>⛽ <strong>Fuel Type:</strong> {fuel_type}</li>
                        <li>📍 <strong>Location:</strong> {location}</li>
                        <li>🚗 <strong>Body Type:</strong> {body_type}</li>
                        <li>🔢 <strong>Kilometer Driven:</strong> {kilometer_driven:,} km</li>
                        <li>🔧 <strong>Engine Displacement:</strong> {engine_displacement:,} cc</li>
                        <li>📉 <strong>Mileage:</strong> {mileage:.1f} kmpl</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True
            )

        except Exception as e:
            st.error(f"⚠️ An error occurred: {e}")

