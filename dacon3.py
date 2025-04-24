import numpy as np
import pandas as pd
import optuna

from xgboost import XGBRegressor
from catboost import CatBoostRegressor, Pool

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import cross_val_score, train_test_split

from lightgbm import LGBMRegressor


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

a['기업가치(백억원)'] = a['기업가치(백억원)'] + 1
a['투자단계'] = a['투자단계'] + 1

train_df['기업가치(백억원)'] = train_df['기업가치(백억원)'] + 1
train_df['투자단계'] = train_df['투자단계'] + 1
train_df['ROI'] = train_df['연매출(억원)'] / train_df['총 투자금(억원)']
train_df['ARPU'] = train_df['연매출(억원)'] / train_df['고객수(백만명)']
train_df['직원당매출'] = train_df['연매출(억원)'] / train_df['직원 수']
# train_df['평균성장률'] = train_df['기업가치(백억원)'] / train_df['지속기간']
train_df['sns'] = train_df['SNS 팔로워 수(백만명)'] / train_df['고객수(백만명)']
train_df['고객성장률'] = train_df['고객수(백만명)'] / train_df['지속기간']

a['ROI'] = a['연매출(억원)'] / a['총 투자금(억원)']
a['ARPU'] = a['연매출(억원)'] / a['고객수(백만명)']
a['직원당매출'] = a['연매출(억원)'] / a['직원 수']
# a['평균성장률'] = a['기업가치(백억원)'] / a['지속기간']
a['sns'] = a['SNS 팔로워 수(백만명)'] / a['고객수(백만명)']
a['고객성장률'] = a['고객수(백만명)'] / a['지속기간']

train_df['직원 수'] = np.log1p(train_df['직원 수'])
train_df['고객수(백만명)'] = np.log1p(train_df['고객수(백만명)'])
train_df['총 투자금(억원)'] = np.log1p(train_df['총 투자금(억원)'])
train_df['연매출(억원)'] = np.log1p(train_df['연매출(억원)'])

scaler = MinMaxScaler()
scaled_cols = scaler.fit_transform(train_df[['직원 수', '고객수(백만명)', '총 투자금(억원)', '연매출(억원)']])
train_df[['직원 수', '고객수(백만명)', '총 투자금(억원)', '연매출(억원)']] = scaled_cols

a['직원 수'] = np.log1p(a['직원 수'])
a['고객수(백만명)'] = np.log1p(a['고객수(백만명)'])
a['총 투자금(억원)'] = np.log1p(a['총 투자금(억원)'])
a['연매출(억원)'] = np.log1p(a['연매출(억원)'])

scaled_cols = scaler.fit_transform(a[['직원 수', '고객수(백만명)', '총 투자금(억원)', '연매출(억원)']])
a[['직원 수', '고객수(백만명)', '총 투자금(억원)', '연매출(억원)']] = scaled_cols

y = train_df['성공확률']
x = train_df.drop(columns=['성공확률'])
dum_x = pd.get_dummies(x)
test_x = pd.get_dummies(a)
x_train, x_test, y_train, y_test = train_test_split(dum_x, y, test_size=0.08)





#optuna for forest
def object(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 30),
        'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
        'max_features': trial.suggest_float('max_features', 0.1, 1.0),
        'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
        'random_state': 42,
        'n_jobs': -1
    }

    model = RandomForestRegressor(**params)
    score = cross_val_score(model, x_train, y_train, cv=5, scoring='neg_root_mean_squared_error')
    return -score.mean()

study = optuna.create_study(direction='minimize')
study.optimize(object, n_trials=50)

print("Best params:", study.best_params)

RandomForestRegressor(n_estimators=945, max_depth=21, min_samples_split=4, min_samples_leaf=1, max_features=0.28411362712866717, bootstrap=True, random_state=42)best_rr =
best_rr.fit(x_train, y_train)
pred = best_rr.predict(x_test)
print(mean_absolute_error(y_test, pred))
pred_test = best_rr.predict(test_x)

#optuna for XGB
def object(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 50, 1000),
        'max_depth': trial.suggest_int('max_depth', 3, 15),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 0.5, 2.0)
    }

    model = XGBRegressor(**params, random_state=42)
    score = cross_val_score(model, x_train, y_train, scoring='neg_mean_squared_error', cv=3).mean()
    return -score


study = optuna.create_study(direction='minimize')
study.optimize(object, n_trials=50)

print("Best params:", study.best_params)



model_xgb = XGBRegressor(n_estimators=104, max_depth=12, learning_rate=0.026888816720944776, subsample=0.6822156907095462, colsample_bytree= 0.9952793703123048, reg_alpha=0.9287594490356247, reg_lambda=1.9210118926583737, random_state=42, eval_metric='mae')
model_xgb.fit(x_train, y_train)
pred_xgb = model_xgb.predict(x_test)
print(mean_absolute_error(y_test, pred_xgb), r2_score(y_test, pred_xgb))
pred_test = model_xgb.predict(test_x)

train_pool = Pool(x_train, y_train)
valid_pool = Pool(x_test, y_test)

def objective(trial):
    params = {
        "iterations": 1000,
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "depth": trial.suggest_int("depth", 4, 10),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        "random_strength": trial.suggest_float("random_strength", 0.0, 1.0),
        "bagging_temperature": trial.suggest_float("bagging_temperature", 0.0, 1.0),
        "border_count": trial.suggest_int("border_count", 32, 255),
        "loss_function": "MAE",
        "eval_metric": "MAE",
        "early_stopping_rounds": 50,
        "verbose": 0,
        "random_seed": 42
    }

    model = CatBoostRegressor(**params)
    model.fit(train_pool, eval_set=valid_pool, use_best_model=True)
    preds = model.predict(x_test)
    mae = mean_absolute_error(y_test, preds)
    return mae

# Optuna 최적화 실행
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

print("Best MAE:", study.best_value)
print("Best Params:", study.best_params)

cat_features = x.select_dtypes(include=["object"]).columns.tolist()
model_cat = CatBoostRegressor(
    iterations=3000,
    learning_rate=0.1884120270980683,
    depth=6,
    l2_leaf_reg=4.007163440040909,
    loss_function='MAE',
    early_stopping_rounds=100,
    verbose=100,
    random_strength=0.6260476533631646,
    bagging_temperature=0.23389681577968308,
    border_count=129
)

model_cat.fit(x_train, y_train, eval_set=(x_test,y_test))
pred_cat = model_cat.predict(x_test)
mae = mean_absolute_error(y_test, pred_cat)
print("MAE:", mae)
pred_test = model_cat.predict(test_x)

voting_model = VotingRegressor([
    ('model_cat', model_cat),
    ('model_xgb', model_xgb),
    ('best_rr', best_rr)
])

voting_model.fit(x_train, y_train)
pred_voting = voting_model.predict(x_test)
rmse_voting = mean_squared_error(y_test, pred_voting)
print("Voting Regressor RMSE:", rmse_voting)
print("Voting Regressor MAE:", mean_absolute_error(y_test, pred_voting))
pred_test = voting_model.predict(test_x)

submission01 = pd.DataFrame()
submission01['ID'] = test["ID"]
submission01['성공확률'] = pred_test
submission01.to_csv('submission18.csv', index=False)