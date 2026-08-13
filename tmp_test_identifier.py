import pandas as pd
from app import detect_identifier_columns

cases = {}

# CASE 1:
cases['case1'] = pd.DataFrame({'id':[1,2,3],'age':[25,30,35],'salary':[50000,60000,70000]})
# CASE 2:
cases['case2'] = pd.DataFrame({'employee_id':[1001,1002,1003],'age':[25,30,35],'salary':[50000,60000,70000]})
# CASE 3:
cases['case3'] = pd.DataFrame({'s.no':[1,2,3],'experience':[2,4,6],'performance':[80,85,90]})
# CASE 4:
cases['case4'] = pd.DataFrame({'age':[25,30,35],'salary':[50000,60000,70000],'performance_score':[80,85,90]})
# CASE 5: unique numeric but non-identifier name
cases['case5'] = pd.DataFrame({'salary':[1001,2002,3003],'revenue':[10,20,30],'value':[5,6,7]})

for name, df in cases.items():
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    excluded = detect_identifier_columns(df, numeric_cols)
    print(name, 'numeric_cols=', numeric_cols, 'excluded=', excluded)
