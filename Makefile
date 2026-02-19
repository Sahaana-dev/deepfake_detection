.PHONY: install test app

install:
	python -m pip install -r requirements.txt

test:
	pytest -q

app:
	streamlit run app.py
