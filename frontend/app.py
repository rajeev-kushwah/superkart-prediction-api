
# Streamlit Web App for SuperKart Sales Forecasting
import streamlit as st
import requests
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Backend API URL
# ---------------------------------------------------------
BACKEND_URL = "http://backend:7860"

# ---------------------------------------------------------
# App Title
# ---------------------------------------------------------
st.title("🛒 SuperKart Sales Forecasting App")

st.markdown(
    """
    🔍 Enter product and store attributes to forecast
    **monthly product sales revenue**.

    _All sales are reported in ($) USD._
    """
)


# ---------------------------------------------------------
# SINGLE PRODUCT PREDICTION
# ---------------------------------------------------------

st.subheader("Single Product Prediction")

Product_Weight = st.number_input(
    "Product Weight (oz)",
    min_value=0.0,
    value=12.66,
    help="Weight of the product"
)

Product_Sugar_Content = st.selectbox(
    "Product Sugar Content",
    ["Low Sugar", "Regular", "No Sugar"]
)

Product_Allocated_Area = st.number_input(
    "Product Allocated Area (linear in.)",
    min_value=0.0,
    value=100.0
)

Product_MRP = st.number_input(
    "Maximum Retail Price (USD)",
    min_value=0.0,
    value=150.0
)

Store_Size = st.selectbox(
    "Store Size",
    ["Small", "Medium", "High"]
)

Store_Location_City_Type = st.selectbox(
    "Store Location City Type",
    ["Tier 1", "Tier 2", "Tier 3"]
)

Store_Type = st.selectbox(
    "Store Type",
    [
        "Supermarket Type1",
        "Supermarket Type2",
        "Departmental Store",
        "Food Mart"
    ]
)

Store_Age_Years = st.slider(
    "Store Age (years)",
    min_value=0,
    max_value=30,
    value=10
)

Product_Type_Category = st.selectbox(
    "Product Type Category",
    ["Perishables", "Non Perishables"]
)


# ---------------------------------------------------------
# Prepare JSON payload
# ---------------------------------------------------------

product_data = {
    "Product_Weight": Product_Weight,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Store_Age_Years": Store_Age_Years,
    "Product_Type_Category": Product_Type_Category
}


# ---------------------------------------------------------
# Predict Single Product
# ---------------------------------------------------------

if st.button("Predict", type="primary"):

    try:

        response = requests.post(
            f"{BACKEND_URL}/v1/predict",
            json=product_data,
            timeout=30
        )

        if response.status_code == 200:

            result = response.json()

            predicted_sales = result["Predicted_Sales"]

            st.success(
                f"📈 Predicted Monthly Sales: "
                f"**${predicted_sales:,.2f} USD**"
            )

        else:

            st.error(
                f"❌ API Error ({response.status_code})"
            )

            try:
                st.json(response.json())
            except Exception:
                st.write(response.text)

    except requests.exceptions.ConnectionError:

        st.error(
            "⚠️ Unable to connect to the backend API."
        )

    except requests.exceptions.Timeout:

        st.error(
            "⚠️ Backend request timed out."
        )

    except Exception as e:

        st.error(
            f"⚠️ Unexpected error: {e}"
        )


# ---------------------------------------------------------
# BATCH PREDICTION
# ---------------------------------------------------------

st.divider()

st.subheader("Batch Prediction")

st.markdown(
    """
    Upload a CSV containing multiple products to generate
    sales predictions for all rows.
    """
)

uploaded_file = st.file_uploader(
    "Upload a CSV file",
    type=["csv"]
)


# ---------------------------------------------------------
# Predict Batch
# ---------------------------------------------------------

if uploaded_file is not None:

    if st.button(
        "Predict for Batch",
        type="primary"
    ):

        try:

            files = {
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "text/csv"
                )
            }

            response = requests.post(
                f"{BACKEND_URL}/v1/predictbatch",
                files=files,
                timeout=60
            )


            # -------------------------------------------------
            # Successful API response
            # -------------------------------------------------

            if response.status_code == 200:

                results = response.json()

                st.success(
                    "✅ Predictions completed successfully!"
                )

                try:

                    if isinstance(results, list):

                        df = pd.DataFrame(results)

                    elif isinstance(results, dict):

                        # Dictionary containing only scalar values
                        if all(
                            not isinstance(v, (list, dict))
                            for v in results.values()
                        ):

                            df = pd.DataFrame([results])

                        else:

                            df = pd.DataFrame(results)

                    else:

                        df = pd.DataFrame(
                            {"Result": [results]}
                        )


                    # Display results
                    st.dataframe(
                        df,
                        use_container_width=True
                    )


                    # Optional download
                    csv = df.to_csv(
                        index=False
                    ).encode("utf-8")

                    st.download_button(
                        label="⬇️ Download Predictions",
                        data=csv,
                        file_name="superkart_predictions.csv",
                        mime="text/csv"
                    )


                except Exception as e:

                    st.error(
                        f"Unable to display results "
                        f"as a table: {e}"
                    )

                    st.json(results)


            # -------------------------------------------------
            # Backend returned error
            # -------------------------------------------------

            else:

                st.error(
                    f"❌ API Error ({response.status_code})"
                )

                try:
                    st.json(response.json())

                except Exception:
                    st.write(response.text)


        except requests.exceptions.ConnectionError:

            st.error(
                "⚠️ Unable to connect to the backend API."
            )


        except requests.exceptions.Timeout:

            st.error(
                "⚠️ Batch prediction request timed out."
            )


        except Exception as e:

            st.error(
                f"⚠️ Unexpected error: {e}"
            )
