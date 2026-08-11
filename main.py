import json
import os
import re
import time
import gspread
from google.oauth2.service_account import Credentials
import pandas as pd
from PIL import Image
import streamlit as st
import google.generativeai as genai

st.set_page_config(page_title="DAR - Gemini AI", layout="wide")
st.title("📝 DAR Form Scanner - Gemini AI")

# Setup Gemini API with missing key guard
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY") or os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY is missing! Please configure it in your Streamlit Cloud Secrets settings.")
    st.stop()

genai.configure(api_key=GEMINI_API_KEY)

# Google Sheets setup
SHEET_ID = "1vCLLfhNHj-SV5r5O9Ntq0kJp9K6htRJomelyIN4CnkI"


def get_gsheet_client():
    """Connect to Google Sheets using service account credentials."""
    scopes = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]
    service_account_info = dict(st.secrets["gcp_service_account"])
    creds = Credentials.from_service_account_info(service_account_info, scopes=scopes)
    return gspread.authorize(creds)


def safe_generate_content(model_name, img, prompt):
    """Generate content using the specified Gemini model."""
    model = genai.GenerativeModel(model_name)
    return model.generate_content([prompt, img])


def extract_dar_gemini(image):
    """Extract survey form data using Gemini with fallback and 429 quota handling."""
    prompt = """
    Extract data from this survey form into a JSON list.
    For page 1 "STORE NAME" section upto "REASON FOR NOT PURCHASE" section:
    1. Get "STORE NAME" at the second column on the left if filled out , extract the handwritten text.
    2. For each column, return the hand written/checked answer. If none, return "".
    3. For "WHAT CURRENT BRAND DID YOU USED" upto "REASON FOR NOT PURCHASED" extract the word, letter or number.
    Return keys like: "STORE NAME", "LIVITY 850G", "LIVITY 400G", "GENDER", "18-30", "31-49", "50 & ABOVE", "PURCHASE OF ESSENTIAL ITEMS", "CUSTOMER IS CANVASSING ONLY", "PURCHASE OF MEDICINE", "OTHERS", "WHAT CURRENT BRAND DID YOU USED", "SWITCH", "UPGRADE", "TOP UP", "LIVITY USER", "YES", "NO", "LIVITY 850G", "LIVITY 400G", "YES", "NO", "REASON FOR NOT PURCHASE", etc.
    For handwritten parts in section 1, use "STORE NAME", "WHAT CURRENT BRAND DID YOU USED", "REASON FOR NOT PURCHASE" for text.
    Only return valid JSON array with 1 object, no other text.
    Example: [{"STORE NAME": "MDC AYALA MALLS", "LIVITY 850G": "8", "AGELIVITY 400G": "0", "GENDER": "F", "18-30": "/,1", "31-49": "/,1", "50 & ABOVE": "/,1", "WHAT CURRENT BRAND DID YOU USED?": "BEAR BRAND ADULT", "REASON FOR NOT PURCHASE": "PRICE EXPENSIVE"}]"""
    
    # 1. First preference: exact, known standard model strings
    preferred_models = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-1.5-pro"]
    
    # 2. Dynamic backup: query all active models supported by your API key
    try:
        available_models = [
            m.name for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
    except Exception:
        available_models = []

    # Merge while preserving preferred order and deduplicating
    models_to_try = preferred_models + [m for m in available_models if m not in preferred_models]

    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            response = safe_generate_content(model_name, image, prompt)
            break  # Success! Exit loop
        except Exception as e:
            last_error = e
            err_msg = str(e)
            if "429" in err_msg or "quota" in err_msg.lower():
                st.toast(f"⏳ Rate limit hit on `{model_name}`. Switching model...", icon="⚠️")
                time.sleep(3)
            continue

    if not response:
        raise Exception(f"All model attempts failed. Last error: {last_error}")

    json_text = response.text.strip()
    
    # Clean markdown code block wraps
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()

    return json.loads(json_text)


# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload DAR Photo", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ready to scan", use_container_width=True)

    if st.button("🔍 Run AI Scan", type="primary"):
        with st.spinner('Gemini AI is reading... ~3-5 seconds'):
            try:
                table_data = extract_dar_gemini(image)
                if table_data:
                    st.success("✅ Extracted dar data!")
                    st.session_state.df = pd.DataFrame(table_data)
                    st.rerun()
                else:
                    st.warning("Walang na-detect na data. Try mo mas malinaw na picture.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Show editor + sync controls if data exists
if st.session_state.df is not None:
    st.subheader("📋 Verify Data - Edit mo kung may mali")
    
    edited_df = st.data_editor(
        st.session_state.df,
        num_rows="dynamic",
        use_container_width=True,
        key="editor"
    )
    st.session_state.df = edited_df

    col1, col2 = st.columns(2)
    with col1:
        csv = st.session_state.df.to_csv(index=False).encode('utf-8')
        st.download_button(
            "📥 Download CSV",
            csv,
            "dar_data.csv",
            "text/csv",
            use_container_width=True
        )

    with col2:
        if st.button("🚀 Sync All to Google Sheets", use_container_width=True):
            try:
                with st.spinner('Syncing to Google Sheets...'):
                    client = get_gsheet_client()
                    sheet = client.open_by_key(SHEET_ID).sheet1
                    
                    # Convert DataFrame to clean strings/lists for gspread
                    df_to_sync = st.session_state.df.fillna("")
                    headers = df_to_sync.columns.tolist()
                    rows = df_to_sync.astype(str).values.tolist()

                    existing_records = sheet.get_all_values()
                    
                    if len(existing_records) == 0:
                        # Append both header and rows if spreadsheet is empty
                        sheet.append_rows([headers] + rows, value_input_option='USER_ENTERED')
                    else:
                        # Append only new data rows if headers already exist
                        sheet.append_rows(rows, value_input_option='USER_ENTERED')

                    st.success(f"✅ {len(rows)} row(s) synced sa Google Sheets!")
                    st.balloons()
            except Exception as e:
                st.error(f"Sync failed: {str(e)}")
                st.code(f"Error details: {repr(e)}")
                st.info("Check: 1. Naka-share ba sheet sa service account? 2. Tama ba secrets?")
else:
    st.info("👆 Upload a dar photo to start")
    st.warning("⚠️ REVIEW and EDIT kung may MALI")
