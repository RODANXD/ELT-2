import streamlit as st
import os
import json
import pandas as pd
from utils.file_loader import (
    load_excel,
    load_destination_schema,
    load_destination_tables
)
from utils.gpt_mapper import map_schema_with_gpt
from utils.transformer import transform_data
from utils.progress_state import init_progress_state
from utils.validator import validate_transformed_data
from config import APP_TITLE
from logger import setup_logger, get_log_stream
from zipfile import ZipFile

# ─────────────────────────────────────────────────────────────────────────────
# Setup logger and page config
logger = setup_logger("app")
st.set_page_config(page_title=APP_TITLE, layout="wide")
st.title(APP_TITLE)
st.markdown(
    '<h6 style="color:#4CAF50;">AI Based Automated schema mapping, transformation, and validation</h6>',
    unsafe_allow_html=True
)

# Inject CSS for smaller font and scrollable logs container
st.markdown(
    """
    <style>
    .logs-container {
        max-height: 300px;
        overflow-y: auto;
        font-family: monospace;
        font-size: 12px;
        background-color: #f0f0f0;
        border: 1px solid #ddd;
        padding: 10px;
        white-space: pre-wrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Initialize progress state at startup
init_progress_state()

# ─────────────────────────────────────────────────────────────────────────────
# Load and verify destination schema + reference data at app startup
dest_ready = False
try:
    # Check if data directory and required files exist
    data_dir = "data"
    required_files = ["DestinationSchema.xlsx", "DestinationTables.xlsx"]
    
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found. Please create it and add required files.")
    
    for file in required_files:
        file_path = os.path.join(data_dir, file)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Required file '{file}' not found in data directory.")
    
    logger.info("Loading destination schema and reference data at startup.")
    dest_schema = load_destination_schema("data/DestinationSchema.xlsx")
    logger.info(f"Loaded destination schema: {len(dest_schema)} tables found.")

    dest_tables = load_destination_tables("data/DestinationTables.xlsx")
    print(f"Destination tables loaded: {len(dest_tables)} tables found.")
    logger.info(f"Loaded destination tables: {len(dest_tables)} rows found.")


    st.success("✅ Destination schema and reference data loaded successfully.")
    dest_ready = True
except FileNotFoundError as e:
    logger.exception(f"File system error: {e}")
    st.error(f"❌ {str(e)}")
    dest_ready = False
except Exception as e:
    logger.exception("Failed to load destination schema or reference data at startup.")
    st.error(f"❌ Error loading destination schema/reference data: {e}")
    dest_ready = False


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar input UI
st.sidebar.title("Input Parameters")


activity_cats_df = dest_tables['DE1_ActivityCategory']
activity_subcats_df = dest_tables['DE1_ActivitySubcategory']
company = st.sidebar.text_input("Company Name", "")
country = st.sidebar.text_input("Country", "")

activity_cat = st.sidebar.selectbox(
    "Activity Category",
    activity_cats_df["ActivityCategory"].tolist()
)

filtered_subcats = activity_subcats_df[
    activity_subcats_df["ActivityCategoryID"] ==
    activity_cats_df.loc[
        activity_cats_df["ActivityCategory"] == activity_cat,
        "ActivityCategoryID"
    ].iloc[0]
]
activity_subcat = st.sidebar.selectbox(
    "Activity Subcategory",
    filtered_subcats["ActivitySubcategoryName"].tolist()
)

# Add Reporting Year input
reporting_year = st.sidebar.number_input("Reporting Year", min_value=2000, max_value=2100, value=2025)

calc_method = st.sidebar.selectbox(
    "Calculation Method",
    ["Expenses-based", "Consumption-based"]
)

# Only show file uploader and Run Mapping if destination data is ready
uploaded_file = None
def _show_file_uploader():
    return st.file_uploader("Upload SourceData Excel", type=["xlsx"])

# Only show file uploader when destination tables are loaded and after we have necessary org-unit input
if not dest_ready:
    st.warning("Cannot upload source file until destination schema and reference data load successfully.")

company_mode = st.radio(
    "Does the company already exist?",
    ["A single company / unit", "Multiple companies or organizational units"],
    key="company_mode"
)

# default unit_mode to ensure it's always defined for transform_data call
unit_mode = "Single unit"

company_mode = st.radio(

if company_mode == "A single company / unit":
    unit_mode = st.radio(
        "Is this data for a single unit or multiple units?",
        ["Single unit", "Multiple units"],
        key="unit_mode"
    )
    if unit_mode == "Single unit":
        # Ask the user to provide the single organizational unit name
        org_unit_name = st.text_input("Organizational Unit Name (this will be created/linked in D_OrganizationalUnit)")
    else:
        # multiple units under single company
        multi_unit_upload = st.radio(
            "Are you uploading each organizational unit's data as a separate file?",
            ["Yes - uploading separately", "No - all units in one file"],
            key="multi_unit_upload"
        )
        if multi_unit_upload == "Yes - uploading separately":
            # For separate uploads we ask for the unit name for this upload
            org_unit_name = st.text_input("Organizational Unit Name for this upload")
        else:
            org_unit_name = None
            st.info("Please ensure the file you upload contains a column identifying the organizational unit (e.g., 'organizational_unit' or 'unit_name'). The AI will use that column to create/link units.")
else:
    # Multiple companies / org units scenario - AI will attempt to auto-detect
    org_unit_name = None
    st.info("The AI will auto-detect companies & organizational units from the file when possible.")

# Show file uploader now that org-unit inputs are displayed (only if destination data loaded)
if dest_ready:
    uploaded_file = _show_file_uploader()

# ─────────────────────────────────────────────────────────────────────────────
# Layout: Main area with 1 column (content)
left_col = st.container()

with left_col:
    if st.button("Run ETL"):
        logger.info("User triggered schema mapping process.")

        if not dest_ready:
            logger.warning("Attempted to run mapping without destination data ready.")
            st.error("Destination data not loaded. Cannot proceed.")
        elif not uploaded_file:
            logger.warning("File upload is missing.")
            st.error("Please upload a source Excel file.")
        elif not all([company, country, calc_method, activity_cat, activity_subcat]):
            logger.warning("One or more input parameters are missing.")
            st.error("Please fill in all input parameters.")
        else:
            try:
                with st.spinner("🔄 Loading source file..."):
                    logger.info("Loading uploaded Excel file.")
                    source_table_name, source_df = load_excel(uploaded_file)
                    logger.info(f"Loaded source file.")

                    os.makedirs("outputs", exist_ok=True)
                    logger.info("Source file loaded successfully. Proceeding with mapping.")

                with st.spinner("🤖 Mapping schema with AI..."):
                    logger.info("Calling GPT to perform schema mapping.")
                    mapping = map_schema_with_gpt(
                    source_df.columns.tolist(), 
                    dest_schema,
                    source_table_name,
                    calc_method, 
                    activity_cat,
                    activity_subcat,
                    source_df  
                )
                    #with open("outputs/mappings.json", "r", encoding="utf-8") as f:
                    #   mapping = json.load(f)
                    logger.info("GPT-based schema mapping completed.")

                with st.spinner("🔄 Transforming data..."):
                    logger.info("Starting data transformation.")
                    transformed_data = transform_data(
                        source_df,
                        mapping,
                        company,
                        country,
                        calc_method,
                        activity_cat,
                        activity_subcat,
                        dest_schema,
                        dest_tables,
                        reporting_year,
                        org_unit_name,
                        company_mode,   # <-- add this
                        unit_mode  
                    )
                    
                    logger.info("Data transformation complete.")

                with st.spinner("💾 Validating and saving outputs..."):
                    logger.info("Starting data validation and output file creation.")
                    validation_passed = validate_transformed_data(
                        transformed_data,
                        dest_schema,
                        "outputs"
                    )
                    
                    if not validation_passed:
                        logger.warning("Validation issues found. Check the validation report in the output folder.")
                    
                    logger.info("Validation complete. Outputs saved.")


                st.success("✅ Process complete!")
                logger.info("Process completed successfully.")
            except Exception as e:
                logger.exception("An error occurred during processing.")
                st.error(f"❌ An error occurred: {e}")
