import streamlit as st
from multiapp import MultiApp
from apps import home, data, model, prediction # import your app modules here

app = MultiApp()

st.markdown("""
# Prediction with LSTM



""")

# Add all your application here
app.add_app("Trend", model.app)

app.add_app("Prediction", prediction.app)
# app.add_app("Trend", model.app)
# app.add_app("Home", home.app)
# app.add_app("Data", data.app)
# app.add_app("Model", model.app)
# app.add_app("prediction", prediction.app)
# The main app
app.run()
