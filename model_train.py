import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import  numpy as np

class ModelTrainer:  # ModelTrainer sınıfı, makine öğrenimi modellerinin eğitim, değerlendirme ve tahmin süreçlerini yönetir.

    def __init__(self, output_dir, model_choice):
        """
        ModelTrainer sınıfı, model eğitim ve değerlendirme işlemleri için kullanılır.
        Çıktı görsellerinin kaydedileceği klasörün adı da parametre olarak alınır.
        """
        self.model = None  # Model örneği
        self.output_dir = output_dir  # Çıktıların kaydedileceği dizin
        self.model_choice = model_choice
        self.results = []  # Tahmin sonuçlarını saklamak için liste
        # results dizini var mı, kontrol et yoksa oluştur
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    @staticmethod # Veriyi işler, özellikler ve hedef değişkeni ayırır.
    def preprocess_data(data):
        """
        Veriyi işler, özellikler ve hedef değişkeni ayırır.
        'filename' sütununu veri setinden kaldırır.
        """
        # 'filename' sütununu veri setinden kaldır
        if 'file_name' in data.columns:
            data = data.drop(columns=['file_name'])
            data = data.drop(columns=['min_freq'])
            data = data.drop(columns=['max_freq'])
            data = data.drop(columns=['mean_magnitude_prev'])

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

    @staticmethod # Veriyi eğitim ve test setlerine böler.
    def split_data(X, y, test_size=0.2, random_state=42):
        """
        Veriyi eğitim ve test setlerine böler.
        """
        return train_test_split(X, y, test_size=test_size, random_state=random_state)

    # GridSearchCV ile modelin hiperparametrelerini optimize eder.
    # En iyi parametreleri belirler ve modeli günceller.
    def tune_model(self, X_train, y_train):
        """
        GridSearchCV ile modelin hiperparametrelerini optimize eder.
        """
        if self.model_choice not in ["random_forest", "svm"]:
            raise ValueError(
                f"Geçersiz model seçimi: {self.model_choice}. 'random_forest' veya 'svm' olarak ayarlanmalıdır.")
        grid_search = None  # grid_search'ü burada None ile başlatıyoruz
        if self.model_choice == "random_forest":
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
            }
            grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
        elif self.model_choice == "svm":
            param_grid = {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf'],
                'gamma': ['scale', 'auto']
            }
            grid_search = GridSearchCV(SVC(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)

        grid_search.fit(X_train, y_train)
        self.model = grid_search.best_estimator_
        print(f"En iyi model parametreleri: {grid_search.best_params_}")

    # Seçilen modelle eğitim yapar.
    def train_model(self, X_train, y_train):
        """
        Modeli eğitir. RandomForest veya SVM modeline göre eğitim yapar.
        """
        if self.model_choice == "random_forest":
            if not self.model:
                self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            print("RandomForest modeli ile eğitim başlıyor.")
        elif self.model_choice == "svm":
            if not self.model:
                self.model = SVC(kernel='linear', random_state=42)
            print("SVM modeli ile eğitim başlıyor.")

        self.model.fit(X_train, y_train)
        print("Model eğitimi tamamlandı.")

    # Doğruluk, Confusion Matrix ve Classification Report sunar.
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

    # Modelin özellikler üzerindeki önemini analiz eder.
    def feature_importance(self, X_train, y_train):
        """
        Modelin hangi özelliklere daha çok önem verdiğini analiz eder.
        RandomForest'ta feature_importances_ kullanılır, SVM'de linear kernel için coef_ kullanılır.
        Diğer SVM kernel'ları için permütasyon yöntemi kullanılır.
        """
        if isinstance(self.model, RandomForestClassifier):
            importances = self.model.feature_importances_
            feature_names = X_train.columns
            feature_importances = pd.DataFrame(importances, index=feature_names, columns=['Importance']).sort_values(
                by='Importance', ascending=False)

            print("\nÖzelliklerin Önemi (Feature Importances - RandomForest):")
            print(feature_importances)

            # Özelliklerin önemini görselleştirme
            plt.figure(figsize=(20, 12))  # Uygun bir boyut
            feature_importances.plot(kind='bar')
            plt.title('Feature Importances (RandomForest)')
            plt.xticks(rotation=45, ha='right')  # X-tiklerini döndür

            # Feature Importances'ı kaydet
            fi_output_path = os.path.join(self.output_dir, 'feature_importances_rf.png')
            plt.savefig(fi_output_path, bbox_inches='tight')  # Kaydetme işlemi
            print(f"Feature Importances görseli kaydedildi: {fi_output_path}")
            plt.close()
        elif isinstance(self.model, SVC):
            if self.model.kernel == 'linear':
                # SVM'de linear kernel için feature importance, model.coef_ kullanılarak elde edilir
                importances = self.model.coef_[0]  # Linear SVM için coef_
                feature_names = X_train.columns
                feature_importances = pd.DataFrame(importances, index=feature_names,
                                                   columns=['Importance']).sort_values(
                    by='Importance', ascending=False)

                print("\nÖzelliklerin Önemi (Feature Importances - SVM):")
                print(feature_importances)

                # Özelliklerin önemini görselleştirme
                plt.figure(figsize=(20, 12))  # Uygun bir boyut
                feature_importances.plot(kind='bar')
                plt.title('Feature Importances (SVM - Linear Kernel)')
                plt.xticks(rotation=45, ha='right')  # X-tiklerini döndür

                # Feature Importances'ı kaydet
                fi_output_path = os.path.join(self.output_dir, 'feature_importances_svm.png')
                plt.savefig(fi_output_path, bbox_inches='tight')  # Kaydetme işlemi
                print(f"Feature Importances görseli kaydedildi: {fi_output_path}")
                plt.close()
        else:
            print("Özellik önemi analizi, yalnızca RandomForest veya SVM için desteklenmektedir.")

    # Veriyi ön işler, böler, modeli eğitir ve değerlendirir.
    def run_training(self, data):
        """
        Model eğitim ve değerlendirme adımlarını çalıştırır.
        """
        # Veriyi ön işle
        X, y = self.preprocess_data(data)

        # Eğitim ve test setlerine böl
        X_train, X_test, y_train, y_test = self.split_data(X, y)

        # Eğitim verisi boş mu kontrol et
        if X_train.empty or y_train.empty:
            raise ValueError("Eğitim verisi boş. Lütfen verinizi kontrol edin.")

        # Modelin hiperparametrelerini optimize et
        self.tune_model(X_train, y_train)
        # Modeli eğit
        self.train_model(X_train, y_train)

        # Özellik önemini analiz et
        self.feature_importance(X_train,y_train)

        # Modeli değerlendir
        return self.evaluate_model(X_test, y_test)

    # Modeli K-Fold çapraz doğrulama ile değerlendirir.
    def cross_validate(self, X, y, cv=5):
        """
        Modeli K-Fold çapraz doğrulama ile değerlendirir.
        """
        kfold = KFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=kfold)
        print(f"\n{cv}-Fold Cross Validation Accuracy Scores: {scores}")
        print(f"Ortalama Doğruluk: {scores.mean():.2f}")

    # Yeni verilerle tahmin yapar ve sonuçları CSV dosyasına kaydeder.
    def predict(self, features_list):
        """
        Yeni verilerle tahmin yapar ve sonuçları CSV dosyasına kaydeder.
        :param features_list: Tahmin yapılacak özelliklerin listesi (list formatında).
        :return: Tahmin sonuçları
        """
        for features in features_list:
            # Özellikleri numpy dizisine çevir ve model için uygun şekle getir
            features_array = np.array(features).reshape(1, -1)

            # Tahmin yapma
            predictions = self.model.predict(features_array)

            # Sonuçları sakla
            self.results.append(predictions[0])  # predictions[0] çünkü tek bir tahmin alıyoruz

        # Tahmin sonuçlarını CSV dosyasına yaz
        results_df = pd.DataFrame(self.results, columns=["predictions"])
        results_csv_path = "results/predictions.csv"
        results_df.to_csv(results_csv_path, index=False)
        print(f"Tahmin sonuçları {results_csv_path} dosyasına kaydedildi.")

        return self.results



