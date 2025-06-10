#!/bin/bash

# 设置工作目录
cd /home/thezfz/MVP

# 清理之前可能存在的压缩包
rm -f AIGraphX_Project.tar.gz

# 创建临时目录用于拷贝需要打包的文件
mkdir -p temp_package
rm -rf temp_package/*

# 拷贝后端文件，排除不需要的目录和文件
echo "复制后端文件..."
rsync -av --progress AIGraphX/ temp_package/AIGraphX/ \
  --exclude=".git/" \
  --exclude=".vscode/" \
  --exclude=".mypy_cache/" \
  --exclude=".pytest_cache/" \
  --exclude=".ruff_cache/" \
  --exclude="__pycache__/" \
  --exclude="*.pyc" \
  --exclude="data/" \
  --exclude="models_cache/" \
  --exclude="logs/" \
  --exclude=".env" \
  --exclude="aigraphx.egg-info/"

# 拷贝前端文件，排除不需要的目录和文件
echo "复制前端文件..."
rsync -av --progress AIGraphX_Frontend/ temp_package/AIGraphX_Frontend/ \
  --exclude=".git/" \
  --exclude=".vscode/" \
  --exclude="node_modules/" \
  --exclude="dist/" \
  --exclude=".env*"

# 打包临时目录中的文件
echo "正在打包..."
tar -czf AIGraphX_Project.tar.gz -C temp_package .

# 验证压缩包是否正常
echo "验证压缩包..."
mkdir -p verify_package
tar -tzf AIGraphX_Project.tar.gz > /dev/null 2>&1
if [ $? -eq 0 ]; then
  echo "压缩包创建成功！大小：$(du -h AIGraphX_Project.tar.gz | cut -f1)"
  ls -lh AIGraphX_Project.tar.gz
else
  echo "警告：压缩包验证失败，可能已损坏"
  exit 1
fi

# 清理临时文件
echo "清理临时文件..."
rm -rf temp_package
rm -rf verify_package

echo "完成！" 