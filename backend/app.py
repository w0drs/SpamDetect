import streamlit as st
import requests

st.title("🔍 Checking message for spam")

with st.form("spam_check_form"):
    message = st.text_area("Message", placeholder="Enter your message here...")
    submit = st.form_submit_button("Check for spam")

if submit and message.strip():
    try:
        data = {"text": message}

        with st.spinner("🔍 Analyzing message..."):
            response = requests.post("http://127.0.0.1:8000/predict", json=data)

        if response.status_code == 200:
            result = response.json()
            prediction = result["prediction"]

            if prediction == "spam":
                st.error("🚨 This message is SPAM!")
            else:
                st.success("✅ This message is NOT spam")

            st.info(f"Prediction: {prediction}")
        else:
            st.error(f"❌ API Error: {response.status_code}")
    except Exception as e:
        st.error(f"❌ Connection error: {e}")

elif submit and not message.strip():
    st.warning("⚠️ Please enter a message to check")