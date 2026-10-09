import os
import tempfile
import streamlit as st

from workflow import workflow, load_pdf_chunks

st.set_page_config(
    page_title="Gemini AI Document Assistant",
    page_icon="📄",
    layout="wide",
)

st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] .main .block-container {
        padding-bottom: 6rem;
    }
    [data-testid="stForm"] {
        border: 0 !important;
        padding: 0 !important;
    }
    [data-testid="stForm"] [data-testid="stHorizontalBlock"] {
        position: relative;
        gap: 0 !important;
    }
    [data-testid="stForm"] [data-testid="stColumn"]:first-child {
        width: 100% !important;
        flex: 1 1 100% !important;
    }
    [data-testid="stForm"] [data-testid="stColumn"]:last-child {
        position: absolute !important;
        top: 0;
        left: auto !important;
        right: 0.25rem;
        width: 3rem !important;
        min-width: 3rem !important;
        max-width: 3rem !important;
        flex: 0 0 3rem !important;
        height: 100%;
        z-index: 2;
    }
    [data-testid="stForm"] [data-testid="stTextInput"] input {
        height: 50px !important;
        padding-right: 3.75rem !important;
    }
    [data-testid="stForm"] [data-testid="stTextInputRootElement"] {
        min-height: 54px !important;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] {
        height: 100%;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button {
        width: 2.75rem;
        height: 2.75rem;
        min-height: 2.75rem;
        padding: 0;
        border: 0;
        border-radius: 50%;
        background: transparent;
        color: #43574e;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button p {
        margin: 0;
        font-size: 1.5rem !important;
        font-weight: 600;
        line-height: 1;
    }
    [data-testid="stForm"] [data-testid="stFormSubmitButton"] button:hover {
        background: rgba(67, 87, 78, 0.08);
        color: #20342d;
    }
    .app-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        padding: 0.7rem 1rem 0.5rem;
        background: var(--background-color);
        z-index: 99;
    }
    .footer-divider {
        width: min(calc(100% - 2.5rem), 980px);
        height: 1px;
        margin: 0 auto;
        background: rgba(49, 51, 63, 0.2);
    }
    .footer-caption {
        color: var(--text-color);
        font-size: 0.8rem;
        opacity: 0.65;
        text-align: center;
    }
    @media (max-width: 640px) {
        .footer-caption {
            font-size: 0.72rem;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📄 AI Document Assistant")
st.write("Upload a PDF and ask questions about its content.")

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"],
)

if uploaded_file:
    if (
        "uploaded_filename" not in st.session_state
        or st.session_state.uploaded_filename
        != uploaded_file.name
    ):
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf",
        ) as temp_file:
            temp_file.write(uploaded_file.getvalue())
            temp_path = temp_file.name

        try:
            with st.spinner("Reading your PDF..."):
                st.session_state.chunks = load_pdf_chunks(
                    temp_path
                )
                st.session_state.uploaded_filename = (
                    uploaded_file.name
                )
                st.session_state.chat_history = []
            st.success("PDF loaded successfully!")

        except Exception as error:
            st.error(f"Could not read PDF: {error}")

        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)


if "chunks" in st.session_state:
    st.subheader("Ask a question")

    with st.form("document_question_form", clear_on_submit=True):
        input_column, arrow_column = st.columns([12, 1])
        with input_column:
            question = st.text_input(
                "Ask something",
                placeholder="Ask something about your PDF...",
                label_visibility="collapsed",
            )
        with arrow_column:
            submitted = st.form_submit_button(
                "↑",
                help="Send question",
                use_container_width=True,
            )

    for message in st.session_state.get(
        "chat_history", []
    ):
        with st.chat_message("user"):
            st.write(message["question"])

        with st.chat_message("assistant"):
            st.write(message["answer"])

    if submitted and question.strip():
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("Gemini is thinking..."):
                try:
                    result = workflow.invoke({
                        "question": question,
                        "context": "",
                        "answer": "",
                        "chat_history": st.session_state.chat_history,
                        "chunks": st.session_state.chunks,
                    })
                    answer = result["answer"]
                    st.write(answer)
                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                    })
                except Exception as error:
                    st.error(f"Something went wrong: {error}")
    elif submitted:
        st.warning("Enter a question first.")
else:
    st.info("Upload a PDF to start chatting with your document.")

st.markdown(
    '<footer class="app-footer">'
    '<div class="footer-divider"></div>'
    '<div class="footer-caption">Built with Python • Streamlit • LangGraph • Gemini</div>'
    '</footer>',
    unsafe_allow_html=True,
)