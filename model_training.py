import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE

dataset_path = "data/features_30_sec.csv"
df = pd.read_csv(dataset_path)

y = df["label"]  
X = df.drop(columns=["filename", "length", "label"])  

y = LabelEncoder().fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train, y_train)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

mlp_model = MLPClassifier(
            hidden_layer_sizes=(150,100, 50),
            max_iter = 800, 
            random_state = 42,
            learning_rate_init = 0.0005,
            alpha = 0.01
)

print("Training Neuronal Network...")
mlp_model.fit(X_train, y_train)
y_pred_mlp = mlp_model.predict(X_test)
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
print(f"MLPClassifier Accuracy: {mlp_accuracy:.2f}")
print("Classification Report:\n", classification_report(y_test, y_pred_mlp))

joblib.dump(mlp_model, "music_genre_model.pkl")
joblib.dump(scaler, "scaler.pkl")

def predecir_genero(features):
    features = scaler.transform([features])
    prediction = mlp_model.predict(features)
    return prediction[0]

sample_features = X_test[0]
predicted_genre = predecir_genero(sample_features)
print(f"Predicted genre: {predicted_genre}")