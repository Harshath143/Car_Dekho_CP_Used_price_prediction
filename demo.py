import streamlit as st
import pandas as pd
import joblib

# Load the model and encoders
loaded_model = joblib.load(r'C:\Users\harsh\OneDrive\Desktop\New folder\Car_price_mini model\Dataset\rf_trainedmodel.pkl')
loaded_encoders = joblib.load(r'C:\Users\harsh\OneDrive\Desktop\New folder\Car_price_mini model\Dataset\label_encoders.pkl')
loaded_scaler = joblib.load(r'C:\Users\harsh\OneDrive\Desktop\New folder\Car_price_mini model\Dataset\min.pkl')

# Define only relevant categorical columns
categorical_columns = [
    "Body_type", "Transmission", "Original_equipment_manufacturer",
    "Model", "Variant_name", "Fuel_type",  "Location"
]

@st.cache_resource
def predict_price(oem, model, variant_name, mileage, engine_displacement, 
                  year_of_manufacture, transmission, fuel_type, owner_no, 
                  model_year, location, kilometer_driven, body_type):
    # Create DataFrame for input data
    input_data = pd.DataFrame({
        "Original_equipment_manufacturer": [oem],
        "Model": [model],
        "Variant_name": [variant_name],
        "Mileage": [mileage],
        "Engine_displacement": [engine_displacement],
        "Year_of_manufacture": [year_of_manufacture],
        "Transmission": [transmission],
        "Fuel_type": [fuel_type],
        "Owner_No.": [owner_no],
        "Model_year": [model_year],
        "Location": [location],
        "Kilometer_Driven": [kilometer_driven],
        "Body_type": [body_type]
    })

    # Apply encoders only to the relevant categorical columns
    for col in categorical_columns:
        le = loaded_encoders[col]
        input_data[col] = le.transform(input_data[col].astype(str))

    # Predict the price and apply inverse scaling
    predicted_price = loaded_model.predict(input_data)
    predicted_price_norm = loaded_scaler.inverse_transform([[predicted_price[0]]])[0][0]
    
    return predicted_price_norm

# Streamlit UI Configuration
st.set_page_config(
    page_title="Car Price Prediction",
    page_icon="https://th.bing.com/th?id=OIP.mLh3isk2acupA_droYNDxgHaGL&w=273&h=228&c=8&rs=1&qlt=90&o=6&dpr=1.3&pid=3.1&rm=2",
    layout="wide"
)

# Custom CSS for background and layout styling
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
            justify-content: center;
            background: linear-gradient(135deg, #E6E6FA 0%, #D8BFD8 100%);
            padding: 30px;
            border-radius: 15px;
            margin-bottom: 20px;
            box-shadow: 0 6px 12px rgba(0, 0, 0, 0.1);
            animation: fadeIn 1.5s ease-in-out;
        }}
        .logo {{
            margin-right: 20px;
            width: 120px;
            height: 120px;
            border-radius: 15px;
            border: 2px solid #00BFFF;
            box-shadow: 0 6px 10px rgba(0, 0, 0, 0.2);
            transition: transform 0.3s ease-in-out;
        }}
        .logo:hover {{
            transform: scale(1.1);
        }}
        h1 {{
            color: #00BFFF;
            font-family: 'Poppins', sans-serif;
            font-size: 2.8rem;
            margin: 0;
            text-shadow: 2px 2px 5px rgba(0, 0, 0, 0.2);
            animation: slideDown 1.2s ease-in-out;
        }}
        @keyframes fadeIn {{
            0% {{ opacity: 0; }}
            100% {{ opacity: 1; }}
        }}
        @keyframes slideDown {{
            0% {{ transform: translateY(-20px); opacity: 0; }}
            100% {{ transform: translateY(0); opacity: 1; }}
        }}
    </style>
    <div class='header'>
        <img class='logo' src='https://st1.latestly.com/wp-content/uploads/2022/11/DCA047E4-E31D-4DF2-96C3-ECB1B6BE3D41.jpeg' alt='Car Logo' />
        <h1>Car Price Prediction</h1>
    </div>
    """,
    unsafe_allow_html=True
)

# Sidebar inputs
data = pd.read_csv("finalcombined_cars.csv")
car_companies = data['Original_equipment_manufacturer'].unique()
car_models = data['Model'].unique()
model_variants = data['Variant_name'].unique()

# Sidebar inputs with enhanced styling
st.sidebar.markdown(
    """
    <style>
        .sidebar-title {
            font-size: 24px;
            color: #FFD700;
            text-align: center;
            margin-bottom: 20px;
            font-weight: bold;
            text-shadow: 1px 1px 2px black;
        }
        .sidebar-input {
            margin-bottom: 15px;
            padding: 10px;
            background-color: rgba(255, 255, 255, 0.1);
            border-radius: 8px;
        }
        .sidebar select, .sidebar input {
            font-size: 16px;
            color: #333;
            border-radius: 5px;
            border: 1px solid #8A2BE2;
            padding: 8px;
        }
    </style>
    <div class='sidebar-title'>🚘 Input Features</div>
    """, 
    unsafe_allow_html=True
)

with st.sidebar:
    # Car Company input
    st.markdown("<div class='sidebar-input'>🏢 Car Company </div>", unsafe_allow_html=True)
    oem = st.selectbox("Select Car Company", car_companies)

    # Car Model input
    st.markdown("<div class='sidebar-input'>🚙 Car Model </div>", unsafe_allow_html=True)
    model = st.selectbox("Select Car Model", car_models)

    # Car Variant input
    st.markdown("<div class='sidebar-input'>🔖 Car Variant </div>", unsafe_allow_html=True)
    variant_name = st.selectbox("Select Car Variant", model_variants)

    # Mileage input
    st.markdown("<div class='sidebar-input'>📉 Mileage (kmpl) </div>", unsafe_allow_html=True)
    mileage = st.number_input("Enter Mileage", min_value=0.0, step=0.1, format="%.1f")

    # Engine Displacement input
    st.markdown("<div class='sidebar-input'>🔧 Engine Displacement (cc) </div>", unsafe_allow_html=True)
    engine_displacement = st.number_input("Enter Engine Displacement", min_value=0)

    # Year of Manufacture input
    st.markdown("<div class='sidebar-input'>📅 Year of Manufacture </div>", unsafe_allow_html=True)
    year_of_manufacture = st.number_input("Enter Year of Manufacture", min_value=1900, max_value=2024)

    # Transmission input
    st.markdown("<div class='sidebar-input'>⚙️  Transmission </div>", unsafe_allow_html=True)
    transmission = st.selectbox("Select Transmission", ["Manual", "Automatic"])

    # Fuel Type input
    st.markdown("<div class='sidebar-input'>⛽ Fuel Type </div>", unsafe_allow_html=True)
    fuel_type = st.selectbox("Select Fuel Type", ["Petrol", "Diesel"])

    # Owner No. input
    st.markdown("<div class='sidebar-input'>👥 Owner No </div>", unsafe_allow_html=True)
    owner_no = st.number_input("Enter Owner No.", min_value=0)

    # Model Year input
    st.markdown("<div class='sidebar-input'>📆 Model Year </div>", unsafe_allow_html=True)
    model_year = st.number_input("Enter Model Year", min_value=1900, max_value=2024)

    # Location input
    st.markdown("<div class='sidebar-input'>📍 Location </div>", unsafe_allow_html=True)
    location = st.selectbox("Select Location", ["Chennai", "Bangalore", "Delhi", "Kolkata", "Jaipur", "Hyderabad"])

    # Body Type input
    st.markdown("<div class='sidebar-input'>🚗  Body Type </div>", unsafe_allow_html=True)
    body_type = st.selectbox(
        "Select Body Type", [
            "Hatchback", "SUV", "Sedan", "MUV", "Minivans", "Coupe", 
            "Pickup Trucks", "Convertibles", "Hybrids", "Wagon", 
            "Crossover", "Station Wagon"
        ]
    )

    # Kilometer Driven input
    st.markdown("<div class='sidebar-input'>🔢 **Kilometer Driven**</div>", unsafe_allow_html=True)
    kilometer_driven = st.number_input("Enter Kilometer Driven", min_value=0)



# Prediction button and result display with advanced styling and animation
if st.sidebar.button("Estimate Used Car Price"):
    with st.spinner('🔍 Analyzing inputs and estimating price...'):
        try:
            # Make the prediction
            predicted_price = predict_price(
                oem, model, variant_name, mileage, engine_displacement, year_of_manufacture,
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

            # Stylish Summary of Inputs with OEM, Model, and Variant Name
            st.markdown(
                f"""
                <div style='padding: 15px; background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
                            color: white; border-radius: 12px; margin-top: 20px;'>
                    <h3 style='text-align: center; margin-bottom: 15px;'>✨ Summary of Your Input ✨</h3>
                    <ul style='list-style: none; padding: 0; font-size: 16px;'>
                        <li>🏢 <strong>Company:</strong> {oem}</li>
                        <li>🚙 <strong>Model:</strong> {model}</li>
                        <li>🔖 <strong>Variant:</strong> {variant_name}</li>
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

