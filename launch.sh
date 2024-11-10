#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}Starting DataSherlocks Project Setup...${NC}"

# Create and activate virtual environment
echo -e "${GREEN}Creating Python virtual environment...${NC}"
python3.11 -m venv venv
source venv/bin/activate

# Install dependencies
echo -e "${GREEN}Installing required packages...${NC}"
pip install -r requirements.txt

# Create necessary directories with full permissions
echo -e "${GREEN}Creating required directories...${NC}"
mkdir -p ./data
mkdir -p ./logs
mkdir -p ./cache
mkdir -p ./models
mkdir -p ./embeddings
mkdir -p ./index

# Set permissions
echo -e "${GREEN}Setting directory permissions...${NC}"
chmod -R 777 ./data
chmod -R 777 ./logs
chmod -R 777 ./cache
chmod -R 777 ./models
chmod -R 777 ./embeddings
chmod -R 777 ./index

# Check if Ollama is installed
if ! command -v ollama &> /dev/null; then
    echo -e "${RED}Ollama is not installed. Installing...${NC}"
    curl -fsSL https://ollama.com/install.sh | sh
    echo -e "${GREEN}Installing LLaMA model...${NC}"
    ollama pull llama3.2:3b-instruct-fp16
    ollama list
fi

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        return 0
    else
        return 1
    fi
}

# Launch the application
echo -e "${GREEN}Launching the application...${NC}"

# Check if streamlit port is available (default 8501)
if check_port 8501; then
    echo -e "${RED}Port 8501 is already in use. Please free the port first.${NC}"
    exit 1
fi

# Launch Streamlit application
echo -e "${BLUE}Starting Streamlit server...${NC}"
streamlit run app_chart.py

# Commented out alternative FastAPI launch
# echo -e "${BLUE}Starting FastAPI server...${NC}"
# uvicorn app:app --host 0.0.0.0 --port 5000 --reload

# Alternative Gunicorn launch (commented out)
# echo -e "${BLUE}Starting Gunicorn server...${NC}"
# pkill gunicorn
# gunicorn -D -w 6 -t 8 -k uvicorn.workers.UvicornWorker app:app --bind 0.0.0.0:5000 --timeout 0 --reload