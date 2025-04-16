import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

test = pd.read_csv('open3/test.csv')
train = pd.read_csv('open3/train.csv')
train['지속기간'] = 2025 - train['설립연도']
test['지속기간'] = 2025 - test['설립연도']
train_df = train.drop(columns=['ID', '설립연도'])
test_df = test.drop(columns=['ID', '설립연도'])

# 분야 결측값 처리
invest = train_df['투자단계'].values
encoder1 = LabelEncoder()
encoder1.fit(invest)
train_df['투자단계'] = encoder1.transform(invest)

country = train_df['국가'].values
encoder2 = LabelEncoder()
encoder2.fit(country)
train_df['국가'] = encoder2.transform(country)

insu = train_df['인수여부'].values
encoder4 = LabelEncoder()
encoder4.fit(insu)
train_df['인수여부'] = encoder4.transform(insu)

sang = train_df['상장여부'].values
encoder5 = LabelEncoder()
encoder5.fit(sang)
train_df['상장여부'] = encoder5.transform(sang)

df_notnull = train_df[train_df['분야'].notnull()]
df_null = train_df[train_df['분야'].isnull()]

cat = df_notnull['분야'].values
encoder6 = LabelEncoder()
encoder6.fit(cat)
df_notnull['분야'] = encoder6.transform(cat)

x = df_notnull[['국가', '투자단계', '총 투자금(억원)', '연매출(억원)']]
y = df_notnull['분야']

model0 = RandomForestClassifier()
model0.fit(x, y)

train_df.loc[train_df['분야'].notnull(), '분야'] = df_notnull['분야']
train_df.loc[train_df['분야'].isnull(), '분야'] = model0.predict(df_null[x.columns])

# 직원 수 결측값 처리
train_df['직원 수'] = train_df.groupby(['국가', '투자단계'])['직원 수'].transform(lambda x: x.fillna(x.mean()))

# 고객수 결측값 처리
df_notnull = train_df[train_df['고객수(백만명)'].notnull()]
df_null = train_df[train_df['고객수(백만명)'].isnull()]

cols = ['총 투자금(억원)', '연매출(억원)', 'SNS 팔로워 수(백만명)', '지속기간']
model1 = RandomForestRegressor()
model1.fit(df_notnull[cols], df_notnull['고객수(백만명)'])

train_df.loc[train_df['고객수(백만명)'].isnull(), '고객수(백만명)'] = model1.predict(df_null[cols])

# 기업가치 결측값처리
df_notnull = train_df[train_df['기업가치(백억원)'].notnull()]
df_null = train_df[train_df['기업가치(백억원)'].isnull()]

cat = df_notnull['기업가치(백억원)'].values
encoder3 = LabelEncoder()
encoder3.fit(cat)
df_notnull['기업가치(백억원)'] = encoder3.transform(cat)

cols = ['총 투자금(억원)', '연매출(억원)', 'SNS 팔로워 수(백만명)', '지속기간']
model2 = RandomForestClassifier()
model2.fit(df_notnull[cols], df_notnull['기업가치(백억원)'])

train_df.loc[train_df['기업가치(백억원)'].notnull(), '기업가치(백억원)'] = df_notnull['기업가치(백억원)']
train_df.loc[train_df['기업가치(백억원)'].isnull(), '기업가치(백억원)'] = model2.predict(df_null[cols])

a = test_df.copy()

a['투자단계'] = encoder1.transform(a['투자단계'])
a['국가'] = encoder2.transform(a['국가'])
a['인수여부'] = encoder4.transform(a['인수여부'])
a['상장여부'] = encoder5.transform(a['상장여부'])

a_notnull = a.loc[a['분야'].notnull()]
a_null = a.loc[a['분야'].isnull()]

a_notnull['분야'] = encoder6.transform(a_notnull['분야'].values)

a.loc[a['분야'].notnull(), '분야'] = a_notnull['분야']
a.loc[a['분야'].isnull(), '분야'] = model0.predict(a_null[x.columns])

a['직원 수'] = a.groupby(['국가', '투자단계'])['직원 수'].transform(lambda x: x.fillna(x.mean()))

a_notnull = a[a['고객수(백만명)'].notnull()]
a_null = a[a['고객수(백만명)'].isnull()]
a.loc[a['고객수(백만명)'].isnull(), '고객수(백만명)'] = model1.predict(a_null[cols])

a_notnull = a[a['기업가치(백억원)'].notnull()]
a_null = a[a['기업가치(백억원)'].isnull()]

a_notnull['기업가치(백억원)'] = encoder3.transform(a_notnull['기업가치(백억원)'].values)
a.loc[a['기업가치(백억원)'].notnull(), '기업가치(백억원)'] = a_notnull['기업가치(백억원)']
a.loc[a['기업가치(백억원)'].isnull(), '기업가치(백억원)'] = model2.predict(a_null[cols])