# 使用輕量版 Python 基底映像
FROM python:3.11-slim

# 安裝系統套件 (LightGBM 需要 libgomp1，另外安裝 gcc/g++)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 gcc g++ \
    && rm -rf /var/lib/apt/lists/*

# 設定工作目錄
WORKDIR /app

# 複製 requirements.txt 並安裝 Python 套件
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案的所有程式碼 (app.py, Train.py, models/, templates/...)
COPY . .

# 設定 Cloud Run 預設的 PORT
ENV PORT=8080

# 用 gunicorn 啟動 Flask app
CMD ["gunicorn", "-b", ":8080", "app:app"]
