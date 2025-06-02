# Arabic Dialects Classification with AraBERT & Streamlit

Classify six major Arabic dialects—including Modern Standard Arabic—using state-of-the-art deep learning and a fun interactive web app!

## 🚀 Overview

This project demonstrates how machine learning and NLP can be used to distinguish between six Arabic varieties:

- **Egyptian**
- **Lebanese**
- **Gulf**
- **Moroccan**
- **Tunisian**
- **Formal Arabic (Modern Standard Arabic)**

We use the [MADAR dataset](https://github.com/ARBML/MADAR) as our foundation, starting with a classic logistic regression model and then fine-tuning [AraBERT v2](https://huggingface.co/aubmindlab/bert-base-arabertv02) for better accuracy. The app is built with [Streamlit](https://streamlit.io/) and features a fully right-to-left user interface for a native experience.

Try it out live:  
👉 [Arabic Dialects Classification Streamlit App](https://arabic-dialects-classification-with-arabert-app-a7bej9g5vlrpkr.streamlit.app/)

## 📈 Results

- **Logistic Regression:** ~82% accuracy across all six dialects
- **AraBERT v2 (fine-tuned):** 89% accuracy

## 🛠️ How It Works

1. **Data:** The app uses the MADAR corpus to train and evaluate the models.
2. **Baseline:** Logistic Regression for multiclass dialect classification.
3. **Deep Learning:** Fine-tuned AraBERT v2 using Hugging Face Transformers for improved results.
4. **Web App:** Deployed via Streamlit with a right-to-left interface. Paste in any Arabic text and get an instant dialect prediction!

## 👨‍💻 Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/TisTis0004/Arabic-Dialects-Classification-with-AraBERT-Streamlit.git
cd Arabic-Dialects-Classification-with-AraBERT-Streamlit
```

### 2. Install dependencies

Create a new virtual environment (optional but recommended), then:

```bash
pip install -r requirements.txt
```

### 3. Run the app

```bash
streamlit run app.py
```

> **Note:** The fine-tuned model is loaded automatically from Hugging Face Hub ([TisTis/arabert-dialect-checkpoint](https://huggingface.co/TisTis/arabert-dialect-checkpoint)), so you don’t need to download anything extra.

## 🌐 Live Demo

Check out the live app:  
[https://arabic-dialects-classification-with-arabert-app-a7bej9g5vlrpkr.streamlit.app/](https://arabic-dialects-classification-with-arabert-app-a7bej9g5vlrpkr.streamlit.app/)

## 🤝 Team

- [Fares Hatahet](https://github.com/TisTis0004)
- Basel Qarout

## 📄 License

This project is open-source and free to use under the [MIT License](LICENSE).

## 📬 Contact

Feel free to open an issue or reach out if you have suggestions or questions!
