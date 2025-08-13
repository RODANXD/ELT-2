import streamlit as st

def init_progress_state():
    if 'progress' not in st.session_state:
        st.session_state.progress = {
            'current_table': '',
            'total_records': 0,
            'processed_records': 0,
            'status': ''
        }
    return st.session_state.progress

def update_progress(table_name=None, total=None, processed=None, status=None):
    if 'progress' not in st.session_state:
        init_progress_state()
        
    if table_name is not None:
        st.session_state.progress['current_table'] = table_name
    if total is not None:
        st.session_state.progress['total_records'] = total
    if processed is not None:
        st.session_state.progress['processed_records'] = processed
    if status is not None:
        st.session_state.progress['status'] = status
