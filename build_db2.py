import pandas as pd
import duckdb
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "CMAPSSData"
DB_PATH = BASE_DIR / "cmapss.db"

def update_db_with_preprocessed():
    # 1. DuckDB 연결
    con = duckdb.connect(str(DB_PATH))
    
    for i in range(1, 5):
        ds = f"FD00{i}"
        ds_lower = ds.lower()
        
        prep_candidates = [
            DATA_DIR / f"test_{ds}_preprocessed.csv",
            DATA_DIR / f"preprocessed_test_{ds}.csv",
        ]
        prep_path = next(
            (
                path for path in prep_candidates
                if path.exists() and path.stat().st_size > 1024
            ),
            None,
        )
        
        if prep_path is not None:
            # 2. 전처리 CSV 로드
            prep_df = pd.read_csv(prep_path)
            
            # 3. DuckDB에 테이블 생성/교체
            # 테이블명 예시: test_fd001_prep
            table_name = f"test_{ds_lower}_prep"
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM prep_df")
            
            print(f"✅ {table_name} 테이블 저장 완료 (소스: {prep_path.name})")
        else:
            print(f"⚠️ 유효한 전처리 파일을 찾을 수 없습니다: {[str(p) for p in prep_candidates]}")

    # 데이터 확인을 위한 로그
    print("-" * 30)
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"현재 DB 내 테이블 리스트: {tables}")
    
    con.close()

if __name__ == "__main__":
    # 이 함수를 실행하면 기존 db 파일에 테이블이 추가됩니다.
    update_db_with_preprocessed()
