from preprocess import load_and_merge_data, add_features
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import joblib

df = load_and_merge_data()
df = add_features(df)

X = df.drop(columns=['Weekly_Sales', 'Date', 'Type'])
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lgbm = LGBMRegressor(random_state=42)

param_grid = {
    'num_leaves': [20, 31, 50, 100],
    'learning_rate': [0.005, 0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 500],
    'max_depth': [-1, 5, 10, 15],
    'min_child_samples': [10, 20, 30],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

random_search = RandomizedSearchCV(
    lgbm,
    param_distributions=param_grid,
    n_iter=30,
    cv=3,
    verbose=1,
    n_jobs=-1,
    scoring='neg_root_mean_squared_error'
)

random_search.fit(X_train, y_train)

best_model = random_search.best_estimator_

y_pred = best_model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print('Best Parameters: ', random_search.best_params_)
print(f'LightGBM RMSE: {rmse:.2f}')
print(f'R2 Score: {r2:.4f}')

feat_imp = pd.Series(best_model.feature_importances_, index=X.columns).sort_values(ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title('LightGBM Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance_lgbm.png')
plt.show()

joblib.dump(best_model, 'lgbm_model.pkl')
print('Model saved!')
print(best_model.feature_importances_.values)