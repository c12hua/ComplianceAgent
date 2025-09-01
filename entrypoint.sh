#!/bin/bash
set -e

# 等待数据库就绪
echo "⏳ Waiting for PostgreSQL..."
until pg_isready -h db -p 5432 -U user; do
  sleep 5
done

# 迁移数据库
echo "📦 Running migrations..."
python manage.py makemigrations
python manage.py migrate

# 启动服务
echo "🚀 Starting Django..."
exec "$@"
