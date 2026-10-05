import streamlit as st
import pandas as pd
import numpy as np
import joblib

art = joblib.load("model.joblib")
model, cols, defaults = art["model"], art["columns"], art["defaults"]

st.title("House Price Predictor")
st.caption("Based on 1970s Boston data. Price shown in dollars of that time. Reliable only up to about $50k, because the data is capped there.")

rm = st.number_input("Average rooms per house (RM)", 3.0, 9.0, 6.2)
lstat = st.number_input("% lower-status population (LSTAT)", 1.0, 40.0, 12.0)
ptratio = st.number_input("Students per teacher (PTRATIO)", 12.0, 22.0, 18.0)
crim = st.number_input("Crime rate (CRIM)", 0.0, 90.0, 0.3)

if st.button("Predict"):
    row = dict(defaults)
    row.update({"RM": rm, "LSTAT": lstat, "PTRATIO": ptratio, "CRIM": crim})
    X = pd.DataFrame([row])[cols]
    # X["LSTAT"] = np.log1p(X["LSTAT"])   # uncomment ONLY if you trained with the log on LSTAT
    price = model.predict(X)[0]
    st.success(f"Predicted price: ${price * 1000:,.0f}")
    if price >= 45:
        st.warning("The data is capped at 50, so this may be underestimated.")