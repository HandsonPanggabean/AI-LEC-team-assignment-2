// install python 3.12 kalau belum ada ( https://www.python.org/downloads )

python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
