from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib, json, os, numpy as np, logging
from typing import List, Dict, Any
from utils.feature_extractor import FeatureExtractor

class LocalHeadingModel:
    def __init__(self, model_dir: str = "model"):
        self.model_dir = model_dir
        self.classifier = None
        self.scaler = None
        self.label_encoder = None
        self.feature_names = None
        os.makedirs(model_dir, exist_ok=True)
        logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
    
    def train_model(self, training_data_file: str) -> Dict[str, Any]:
        with open(training_data_file, 'r', encoding='utf-8') as f:
            training_data = json.load(f)
        if not training_data:
            self.logger.error("No training data found.")
            return {}
        
        feature_extractor = FeatureExtractor()
        features = feature_extractor.extract_features(training_data)
        labels = [s['label'] for s in training_data]

        if len(features) == 0:
            self.logger.error("No features extracted.")
            return {}
        
        self.feature_names = feature_extractor.get_feature_names()
        X = features
        y_str = np.array(labels)

        # 🔁 Encode string labels to integers
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y_str)

        unique_labels = np.unique(y)
        if len(unique_labels) <= 1:
            self.logger.warning("Only one class in training data. Model cannot be meaningfully trained.")
            return {'accuracy': 1.0}

        label_counts = {label: np.sum(y == label) for label in unique_labels}
        if any(count < 2 for count in label_counts.values()):
            self.logger.warning(f"Some classes have only 1 member: {label_counts}. Training without stratification.")
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)

        self.classifier = XGBClassifier(
            use_label_encoder=False,
            eval_metric='mlogloss',
            max_depth=6,
            n_estimators=100,
            learning_rate=0.1,
            random_state=42
        )
        self.classifier.fit(X_train_scaled, y_train)

        X_test_scaled = self.scaler.transform(X_test)
        y_pred = self.classifier.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        self.logger.info(f"Model trained with accuracy: {accuracy:.3f}")

        self._save_model()
        return {'accuracy': accuracy}
    
    def predict(self, blocks: List[Dict[str, Any]]) -> List[str]:
        if not self.classifier or not self.scaler or not self.label_encoder:
            self.logger.error("Model not loaded.")
            return ['NONE'] * len(blocks)

        features = FeatureExtractor().extract_features(blocks)
        if len(features) == 0:
            return ['NONE'] * len(blocks)

        features_scaled = self.scaler.transform(features)
        preds = self.classifier.predict(features_scaled)
        
        # 🔁 Decode integer predictions to string labels
        return self.label_encoder.inverse_transform(preds).tolist()
    
    def _save_model(self):
        joblib.dump(self.classifier, os.path.join(self.model_dir, 'heading_classifier.joblib'))
        joblib.dump(self.scaler, os.path.join(self.model_dir, 'feature_scaler.joblib'))
        joblib.dump(self.label_encoder, os.path.join(self.model_dir, 'label_encoder.joblib'))
        with open(os.path.join(self.model_dir, 'feature_names.json'), 'w') as f:
            json.dump(self.feature_names, f)
        self.logger.info("Model saved successfully.")
    
    def load_model(self) -> bool:
        try:
            self.classifier = joblib.load(os.path.join(self.model_dir, 'heading_classifier.joblib'))
            self.scaler = joblib.load(os.path.join(self.model_dir, 'feature_scaler.joblib'))
            self.label_encoder = joblib.load(os.path.join(self.model_dir, 'label_encoder.joblib'))
            with open(os.path.join(self.model_dir, 'feature_names.json'), 'r') as f:
                self.feature_names = json.load(f)
            self.logger.info("Model loaded successfully.")
            return True
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            return False
