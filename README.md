# lung-cancer-prediction
This is a repo for Lung Cancer prediction implementation. 

## Setting Up the Environment for Windows

1. **Create a Virtual Environment**
   ```sh
   python3 -m venv .venv
2. **Activate the Virtual Environment**
   ```sh
   .venv/bin/activate
3. **Install Required Packages**
   ```sh
   pip install -r requirements.txt
4. **Running the Backend Service**
   ```sh
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

## Setting Up the Environment for Linux Environment

1. **Create a Virtual Environment**
   ```sh
   python3.10 -m venv .venv
2. **Activate the Virtual Environment**
   ```sh
   source .venv/bin/activate
3. **Install Required Packages**
   ```sh
   pip3 install -r requirements.txt
4. **Running the Backend Service**
   ```sh
   uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Use Code Above or Code below

# If You want to try Docker, using the steps below. 

## Build and Run Docker Container

1. **Build docker image**
   ```sh
   sudo docker build -t lung-cancer-app .
2. **Run Container**
   ```sh
   sudo docker run -d -p 8000:8000 -p 8501:8501 lung-cancer-app
