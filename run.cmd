echo Installing required packages ...
pip install pip setuptools wheel virtualenv virtualenvwrapper
echo Creating and activating a fresh virtual environment ...
python -m venv venv
call venv\Scripts\activate.bat
echo Testing virtual environment ...
python test_environment.py
echo Installing project requirements from requirements.txt
pip install -r requirements.txt
python src\clean\make_dataset.py "data\raw" "data\processed"
python src\analyse\run_analysis.py "data\processed" "models"
@REM python src\clean\make_dataset.py "data\raw" "data\processed"
deactivate