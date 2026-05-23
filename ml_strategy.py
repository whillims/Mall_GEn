import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, accuracy_score
import joblib

class MLStrategy:
    def __init__(self, model_type: str = 'random_forest'):
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False

    def _create_model(self):
        if self.model_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              feature_columns: list) -> dict:
        self.feature_columns = feature_columns
        X = X_train[feature_columns].values
        y = y_train.values

        X_scaled = self.scaler.fit_transform(X)

        self.model = self._create_model()
        self.model.fit(X_scaled, y)
        self.is_trained = True

        train_pred = self.model.predict(X_scaled)
        accuracy = accuracy_score(y, train_pred)

        feature_importance = dict(zip(feature_columns,
                                    self.model.feature_importances_))

        print(f"模型训练完成，训练集准确率: {accuracy:.4f}")
        print("\n特征重要性 Top 10:")
        for feat, imp in sorted(feature_importance.items(),
                               key=lambda x: x[1], reverse=True)[:10]:
            print(f"  {feat}: {imp:.4f}")

        return {
            'accuracy': accuracy,
            'feature_importance': feature_importance
        }

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        X_values = X[self.feature_columns].values
        X_scaled = self.scaler.transform(X_values)
        predictions = self.model.predict(X_scaled)
        return predictions

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        if not self.is_trained:
            raise ValueError("模型尚未训练")

        X_values = X[self.feature_columns].values
        X_scaled = self.scaler.transform(X_values)
        probabilities = self.model.predict_proba(X_scaled)
        return probabilities

    def save_model(self, filepath: str):
        if not self.is_trained:
            raise ValueError("模型尚未训练，无法保存")
        joblib.dump({
            'model': self.model,
            'scaler': self.scaler,
            'feature_columns': self.feature_columns,
            'model_type': self.model_type
        }, filepath)
        print(f"模型已保存到: {filepath}")

    def load_model(self, filepath: str):
        data = joblib.load(filepath)
        self.model = data['model']
        self.scaler = data['scaler']
        self.feature_columns = data['feature_columns']
        self.model_type = data['model_type']
        self.is_trained = True
        print(f"模型已从 {filepath} 加载")
