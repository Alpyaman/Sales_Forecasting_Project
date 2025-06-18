import joblib
from preprocess import add_features, load_and_merge_data
from sklearn.model_selection import train_test_split

df = load_and_merge_data()
df = add_features(df)

X = df.drop(columns=['Weekly_Sales', 'Date', 'Type'])
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=42)

model = joblib.load('lgbm_model.pkl')

y_pred = model.predict(X_test)

predictions_df = X_test.copy()
predictions_df['Actual_Sales'] = y_test.values
predictions_df['Predicted_Sales'] = y_pred

predictions_df.to_csv('test_set_predictions.csv', index=False)
print('Predictions Saved.')
