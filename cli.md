python main.py downloader_ill -y --sub h --classes "Child" --type_csv train --limit 120

source .venv_mp/bin/activate
pip install -r requirements.txt
uvicorn src.app:app --reload
