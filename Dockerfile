# Use Python 3.10.12 on Bookworm
FROM python:3.10.12-bookworm

# Set working directory
WORKDIR /root/code

# Copy requirements first to leverage caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Copy your source code
COPY ./code /root/code

# Expose ports
EXPOSE 8888 8050

# Set password environment variable for Jupyter
ENV JUPYTER_PASSWORD_HASH=argon2:$argon2id$v=19$m=10240,t=10,p=8$btpQFdm0YERLMLZl1iyFnQ$DlptzSUOs7ut+59YPk84ydjk2EbCOLewBRR12hTOrp4

# Start both Jupyter Notebook and Dash
CMD jupyter notebook --ip=0.0.0.0 --port=8888 --allow-root --NotebookApp.password=$JUPYTER_PASSWORD_HASH --no-browser & \
    python3 /root/code/app.py
