import pandas as pd
import duckdb
import os

def create_db():
    # 1. DuckDB 연결
    con = duckdb.connect('cmapss.db')
    
    cols = ['unit_nr', 'time_cycles', 'setting_1', 'setting_2', 'setting_3'] + [f's_{i}' for i in range(1, 22)]
    
    # 상위 폴더 경로 설정 (안전하게 os.path 사용)
    base_path = r"CMAPSSData" 
    
    for i in range(1, 5):
        ds = f"FD00{i}"
        
        # [수정] sep=r'\s+' 로 변경 (r을 붙여야 에러가 안 납니다)
        test_path = os.path.join(base_path, f"test_{ds}.txt")
        test_df = pd.read_csv(test_path, sep=r'\s+', header=None, names=cols)
        
        rul_path = os.path.join(base_path, f"RUL_{ds}.txt")
        rul_df = pd.read_csv(rul_path, sep=r'\s+', header=None, names=['RUL'])
        
        # ... (이하 동일한 로직) ...
        rul_df['unit_nr'] = rul_df.index + 1
        max_cycle = test_df.groupby('unit_nr')['time_cycles'].max().reset_index()
        max_cycle.columns = ['unit_nr', 'max_at_test']
        
        test_df = test_df.merge(max_cycle, on='unit_nr').merge(rul_df, on='unit_nr')
        test_df['true_rul'] = (test_df['max_at_test'] + test_df['RUL']) - test_df['time_cycles']
        
        con.execute(f"CREATE OR REPLACE TABLE test_{ds.lower()} AS SELECT * FROM test_df")
        print(f"✅ test_{ds.lower()} 테이블 생성 완료")

    con.close()

if __name__ == "__main__":
    create_db()