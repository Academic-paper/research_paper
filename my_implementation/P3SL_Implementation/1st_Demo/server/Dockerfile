# 1. Start from the official, pre-built PyTorch image (Bypasses the massive pip install)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

WORKDIR /app

# 2. Copy requirements
COPY requirements.txt .

# 3. Install remaining lightweight packages
RUN pip install --no-cache-dir -r requirements.txt

# 4. Copy your python scripts
COPY . .

# 5. Run the script (Adjust this based on whether it is the server or client)
CMD ["python", "server.py"]