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

    # Fix private key formatting if newlines were escaped in Streamlit secrets
    if "private_key" in service_account_info:
        service_account_info["private_key"] = service_account_info["private_key"].replace("\\n", "\n")

    creds = Credentials.from_service_account_info(
        service_account_info,
        scopes=scopes
    )
    return gspread.authorize(creds)


def safe_generate_content(model_name, img, prompt):
    """Generate content forcing pure JSON output."""
    model = genai.GenerativeModel(model_name)
    generation_config = genai.GenerationConfig(
        response_mime_type="application/json"
    )
    return model.generate_content([prompt, img], generation_config=generation_config)


def extract_dar_gemini(image):
    """Extract survey form data using Gemini with complete row & column mappings."""
    
    prompt = """
    You are an expert OCR system. Extract ALL filled table rows (from line 1 down to the last filled row) from this Daily Activity Report (DAR) image into a JSON list of objects.

    For EVERY row with written data, map every column explicitly using these exact unique JSON keys:
    
    - "NO": Line number (e.g. "1", "2", "3")
    - "STORE_NAME": Handwritten store name (e.g., "MDC ABAD SANTOS")
    - "INV_LIVITY_850G": Beginning Inventory Livity 850g (e.g. "1")
    - "INV_LIVITY_400G": Beginning Inventory Livity 400g (e.g. "1")
    - "GENDER": Customer Gender ("F" or "M")
    - "AGE_18_30": Mark under 18-30 bracket ("1" if marked, else "")
    - "AGE_31_49": Mark under 31-49 bracket ("1" if marked, else "")
    - "AGE_50_ABOVE": Mark under 50 & ABOVE bracket ("1" if marked, else "")
    - "REASON_PURCHASE_ESSENTIAL": Mark under Reason for Visiting -> Purchase Essential Items ("1" if marked, else "")
    - "REASON_CANVASSING": Mark under Reason for Visiting -> Customer Canvassing Only ("1" if marked, else "")
    - "REASON_PURCHASE_MEDICINE": Mark under Reason for Visiting -> Purchase Medicine ("1" if marked, else "")
    - "REASON_OTHERS": Mark/text under Reason for Visiting -> Others ("1" or text if marked, else "")
    - "CURRENT_BRAND_USED": Text under "WHAT CURRENT BRAND DID YOU USED?" (e.g. "MEDICINE", "ESSENTIAL", "LIVITY", "ENSURE", "GLUCERNA")
    - "CATEGORY_SWITCH": Mark under Category -> Switch ("1" if marked, else "")
    - "CATEGORY_UPGRADE": Mark under Category -> Upgrade ("1" if marked, else "")
    - "CATEGORY_TOP_UP": Mark under Category -> Top-Up ("1" if marked, else "")
    - "CATEGORY_LIVITY_USER": Mark under Category -> Livity User ("1" if marked, else "")
    - "PURCHASE_YES": Mark under Purchase -> YES ("1" if marked, else "")
    - "PURCHASE_NO": Mark under Purchase -> NO ("1" if marked, else "")
    - "PURCHASE_SKU_850G": Mark under What SKU Did Customer Purchase? -> Livity 850g ("1" if marked, else "")
    - "PURCHASE_SKU_400G": Mark under What SKU Did Customer Purchase? -> Livity 400g ("1" if marked, else "")
    - "SAMPLES_RECEIVED_YES": Mark under Did Customer Received Samples? -> YES ("1" if marked, else "")
    - "SAMPLES_RECEIVED_NO": Mark under Did Customer Received Samples? -> NO ("1" if marked, else "")
    - "REASON_FOR_NOT_PURCHASE": Text under "REASON FOR NOT PURCHASE" (e.g., "NO BUDGET", "MAY DIABETES", "MAY MAINTENANCE", "MAY STOCK PA DAW", etc.)

    RULES:
    1. Scan EVERY row that contains entries (Rows 1 to 10 or more).
    2. Do NOT stop after row 1.
    3. If a cell is empty or unmarked, set its value to "".
    4. Return ONLY a valid JSON array of objects. Do not include markdown formatting or extra text.
    """

    preferred_models = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash"]
    
    try:
        available_models = [
            m.name.replace("models/", "") for m in genai.list_models()
            if "generateContent" in m.supported_generation_methods
        ]
    except Exception:
        available_models = []

    models_to_try = preferred_models + [m for m in available_models if m not in preferred_models]

    response = None
    last_error = None

    for model_name in models_to_try:
        try:
            response = safe_generate_content(model_name, image, prompt)
            break
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
    
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", json_text, re.DOTALL)
    if match:
        json_text = match.group(1).strip()

    try:
        return json.loads(json_text)
    except json.JSONDecodeError:
        json_match = re.search(r"\[\s*\{.*\}\s*\]|\{.*\}", json_text, re.DOTALL)
        if json_match:
            try:
                parsed = json.loads(json_match.group(0))
                return parsed if isinstance(parsed, list) else [parsed]
            except json.JSONDecodeError:
                pass
        raise Exception(f"AI response was not valid JSON:\n{json_text}")


# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None

uploaded_file = st.file_uploader("Upload DAR Photo", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Ready to scan", use_container_width=True)

    if st.button("🔍 Run AI Scan", type="primary"):
        with st.spinner('Gemini AI is reading all rows and columns... ~3-5 seconds'):
            try:
                table_data = extract_dar_gemini(image)
                if table_data:
                    st.success("✅ Extracted DAR data!")
                    st.session_state.df = pd.DataFrame(table_data)
                    st.rerun()
                else:
                    st.warning("No data detected. Try a clearer picture.")
            except Exception as e:
                st.error(f"Error: {str(e)}")

# Show editor + sync controls if data exists
if st.session_state.df is not None:
    st.subheader("📋 Verify Data - Edit as needed")
    
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
                    
                    df_to_sync = st.session_state.df.fillna("")
                    headers = df_to_sync.columns.tolist()
                    rows = df_to_sync.astype(str).values.tolist()

                    # Find actual occupied rows (ignoring trailing empty formatted rows)
                    existing_records = sheet.get_all_values()
                    
                    # Determine true last non-empty row
                    last_occupied_row = 0
                    for idx, row in enumerate(existing_records, start=1):
                        if any(cell.strip() for cell in row):
                            last_occupied_row = idx

                    if last_occupied_row == 0:
                        # Blank sheet: write headers + data starting at A1
                        payload = [headers] + rows
                        start_row = 1
                    else:
                        # Existing data: write only rows directly under the last populated row
                        payload = rows
                        start_row = last_occupied_row + 1

                    range_to_update = f"A{start_row}"
                    sheet.update(range_name=range_to_update, values=payload, value_input_option='USER_ENTERED')

                    st.success(f"✅ {len(rows)} row(s) synced starting at row {start_row}!")
                    st.balloons()
            except Exception as e:
                st.error(f"Sync failed: {str(e)}")
                st.code(f"Error details: {repr(e)}")
                st.info("Check: 1. Is the sheet shared with the service account? 2. Are credentials correct?")
else:
    st.info("👆 Upload a DAR photo to start")
    st.warning("⚠️ REVIEW and EDIT if there are errors")
