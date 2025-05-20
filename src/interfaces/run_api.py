import os
import sys
import uvicorn

# 添加專案根目錄到 Python 路徑
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

if __name__ == "__main__":
    uvicorn.run(
        "interfaces.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=["interfaces"]
    ) 