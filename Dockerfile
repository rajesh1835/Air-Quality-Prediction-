# 1. Use an official Python runtime as a parent image (3.11-slim is a good balance)
FROM python:3.11-slim

# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy the requirements file into the container
COPY requirements.txt .

# 4. Install the required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copy the rest of the application files into the container
COPY . .

# 6. Specify the port on which the Streamlit application runs
EXPOSE 8501

# 7. Define the command to run the Streamlit application when the container starts
# The format is ENTRYPOINT ["executable", "param1", "param2"]
ENTRYPOINT ["streamlit", "run", "src/aqimodel.py", "--server.port=8501", "--server.address=0.0.0.0"]
