import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV

class ModelTrainer:
    def __init__(self, output_dir):
        """
        ModelTrainer sınıfı, model eğitim ve değerlendirme işlemleri için kullanılır.
        Çıktı görsellerinin kaydedileceği klasörün adı da parametre olarak alınır.
        """
        self.model = None  # Model örneği
        self.output_dir = output_dir  # Çıktıların kaydedileceği dizin

        # results dizini var mı, kontrol et yoksa oluştur
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def preprocess_data(self, data):
        """
        Veriyi işler, özellikler ve hedef değişkeni ayırır.
        'filename' sütununu veri setinden kaldırır.
        """
        # 'filename' sütununu veri setinden kaldır
        if 'file_name' in data.columns:
            data = data.drop(columns=['file_name'])

        # Tüm sayısal sütunları al
        numeric_columns = data.select_dtypes(include=[float, int]).columns.tolist()

        # Hedef değişken son sütun olarak varsayalım
        target_column = numeric_columns[-1]  # Son sayısal sütunu hedef olarak al
        feature_columns = numeric_columns[:-1]  # Hedef değişken dışındaki tüm sayısal sütunları al

        # Hedef değişkeni ayır
        y = data[target_column]

        # Özellikleri ayır
        X = data[feature_columns].astype(float)

        return X, y

    def train_test_split(self, X, y, test_size=0.2, random_state=42):
        """
        Veriyi eğitim ve test setlerine böler.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    def tune_model(self, X_train, y_train):
        """
        GridSearchCV ile modelin hiperparametrelerini optimize eder.
        """
        param_grid = {
            'n_estimators': [100, 200],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
        }

        grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)

        self.model = grid_search.best_estimator_
        print(f"En iyi model parametreleri: {grid_search.best_params_}")

    def train_model(self, X_train, y_train):
        """
        RandomForest modelini eğitir.
        """
        if not self.model:
            self.model = RandomForestClassifier(n_estimators=10, random_state=42)

        self.model.fit(X_train, y_train)
        print("Model eğitimi tamamlandı.")

    def evaluate_model(self, X_test, y_test):
        """
        Eğitilen modeli test eder ve performansı değerlendirir.
        """
        y_pred = self.model.predict(X_test)

        # Modelin performansı
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Model Doğruluğu (Accuracy): {accuracy:.2f}")

        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')

        # Confusion Matrix'i results klasörüne kaydet
        cm_output_path = os.path.join(self.output_dir, 'confusion_matrix.png')
        plt.savefig(cm_output_path, bbox_inches='tight')  # Kaydetme işlemi
        print(f"Confusion Matrix kaydedildi: {cm_output_path}")
        plt.close()

        # Classification Report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))



        return accuracy

    def feature_importance(self, X_train):
        """
        Modelin hangi özelliklere daha çok önem verdiğini analiz eder.
        """
        importances = self.model.feature_importances_
        feature_names = X_train.columns
        feature_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values(
            by='Importance', ascending=False)

        print("\nÖzelliklerin Önemi (Feature Importances):")
        print(feature_importances)

        # Özelliklerin önemini görselleştirme
        plt.figure(figsize=(20, 12))  # Uygun bir boyut
        feature_importances.plot(kind='bar')
        plt.title('Feature Importances')
        plt.xticks(rotation=45, ha='right')  # X-tiklerini döndür

        # Feature Importances'ı kaydet
        fi_output_path = os.path.join(self.output_dir, 'feature_importances.png')
        plt.savefig(fi_output_path, bbox_inches='tight')  # Kaydetme işlemi
        print(f"Feature Importances görseli kaydedildi: {fi_output_path}")
        plt.close()

    def run_training(self, data):
        """
        Model eğitim ve değerlendirme adımlarını çalıştırır.
        """
        # Veriyi ön işle
        X, y = self.preprocess_data(data)

        # Eğitim ve test setlerine böl
        X_train, X_test, y_train, y_test = self.train_test_split(X, y)

        # Eğitim verisi boş mu kontrol et
        if X_train.empty or y_train.empty:
            raise ValueError("Eğitim verisi boş. Lütfen verinizi kontrol edin.")

        # Modeli eğit
        self.train_model(X_train, y_train)

        # Özellik önemini analiz et
        self.feature_importance(X_train)

        # Modeli değerlendir
        return self.evaluate_model(X_test, y_test)

    def cross_validate(self, X, y, cv=5):
        """
        Modeli K-Fold çapraz doğrulama ile değerlendirir.
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=kfold)
        print(f"\n{cv}-Fold Cross Validation Accuracy Scores: {scores}")
        print(f"Ortalama Doğruluk: {scores.mean():.2f}")
