import pandas as pd
import duckdb
import os

def update_db_with_preprocessed():
    # 1. DuckDB 연결
    con = duckdb.connect('cmapss.db')
    
    # 기본 경로 설정
    base_path = r"CMAPSSData"
    
    for i in range(1, 5):
        ds = f"FD00{i}"
        ds_lower = ds.lower()
        
        # --- [기존] 전처리된 CSV 파일 경로 설정 ---
        # 파일명 형식: test_FD001_preprocessed.csv
        prep_file_name = f"test_{ds}_preprocessed.csv"
        prep_path = os.path.join(base_path, prep_file_name)
        
        if os.path.exists(prep_path):
            # 2. 전처리 CSV 로드
            prep_df = pd.read_csv(prep_path)
            
            # 3. DuckDB에 테이블 생성/교체
            # 테이블명 예시: test_fd001_prep
            table_name = f"test_{ds_lower}_prep"
            con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM prep_df")
            
            print(f"✅ {table_name} 테이블 저장 완료 (소스: {prep_file_name})")
        else:
            print(f"⚠️ 파일을 찾을 수 없습니다: {prep_path}")

    # 데이터 확인을 위한 로그
    print("-" * 30)
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"현재 DB 내 테이블 리스트: {tables}")
    
    con.close()

if __name__ == "__main__":
    # 이 함수를 실행하면 기존 db 파일에 테이블이 추가됩니다.
    update_db_with_preprocessed()