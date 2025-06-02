import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd

st.markdown("""
    <style>
    h1 {
        text-align: center !important;
        font-size: 4rem !important;
    }
    /* Whole app right-to-left and right-aligned */
    .block-container, .stTextArea, .stTextInput, .stButton, .stTable, .stDataFrameContainer, .stMarkdown, .stAlert, .css-ffhzg2 {
        direction: RTL !important;
        text-align: right !important;
        font-family: "Amiri", "Cairo", "Arial", sans-serif !important;
    }
    /* Style text areas and inputs */
    textarea, input, select {
        direction: RTL !important;
        text-align: right !important;
    }
    /* Table headers and cells */
    th, td {
        direction: RTL !important;
        text-align: right !important;
        font-family: "Amiri", "Cairo", "Arial", sans-serif !important;
    }
    /* Align Streamlit titles and headers */
    .st-emotion-cache-1v0mbdj, .st-emotion-cache-1r6slb0, .st-emotion-cache-10trblm, .st-emotion-cache-16idsys {
        text-align: right !important;
        direction: RTL !important;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_resource
def load_model_and_tokenizer():
    model_path = "TisTis/arabert-dialect-checkpoint"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    return tokenizer, model, device

tokenizer, model, device = load_model_and_tokenizer()

label2id = {'اللغة العربية الفصحى': 0, 'لبناني': 1, 'مغربي': 2, 'خليجي': 3, 'مصري': 4, 'تونسي': 5}
id2label = {v: k for k, v in label2id.items()}

st.title("مصنف اللهجات العربية")
st.markdown("أدخل نصًا عربيًا لتحديد اللهجة باستخدام نموذج AraBERT المدرب")

st.subheader("تجربة نص واحد")
user_text = st.text_area("اكتب نصك العربي هنا:", value="إنت فين يا عم؟", height=120)

if st.button("تعرّف على اللهجة"):
    if user_text.strip():
        encoding = tokenizer(
            [user_text],
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            output = model(**encoding)
            pred_id = output.logits.argmax(-1).item()
            pred_label = id2label[pred_id]
        st.success(f"**اللهجة المتوقعة:** {pred_label}")
    else:
        st.warning("يرجى إدخال نص.")

st.subheader("تجربة مجموعة نصوص (كل سطر نص)")
batch_input = st.text_area("أدخل عدة نصوص (كل نص في سطر)", value="", height=150, key="batch_text_area")
if st.button("تعرّف على اللهجات دفعة واحدة"):
    lines = [l.strip() for l in batch_input.split('\n') if l.strip()]
    if lines:
        encoding = tokenizer(
            lines,
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors='pt'
        )
        encoding = {k: v.to(device) for k, v in encoding.items()}
        with torch.no_grad():
            output = model(**encoding)
            pred_ids = output.logits.argmax(-1).tolist()
            pred_labels = [id2label[pred_id] for pred_id in pred_ids]
        result_df = pd.DataFrame({'النص': lines, 'اللهجة المتوقعة': pred_labels})
        st.table(result_df)
    else:
        st.warning("يرجى إدخال نصوص.")

st.markdown("<hr><center>تم التنفيذ باستخدام AraBERT وStreamlit</center>", unsafe_allow_html=True)
