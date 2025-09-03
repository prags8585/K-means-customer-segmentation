install:
	python -m venv .venv && . .venv/bin/activate && pip install -r requirements.txt

prep:
	python src/data_prep.py --input data/raw/sample_transactions.csv --out data/rfm_features.csv

train:
	python src/train_kmeans.py --input data/rfm_features.csv --k 4 --out data/clustered_customers.csv

app:
	streamlit run src/app_streamlit.py
