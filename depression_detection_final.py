# 1. Imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import joblib 
import librosa
from IPython.display import Audio
import librosa.display
# import random

def preprocess_data(features, labels):
    # Scale features
    global scaler
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)
    
    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(scaled_features, labels, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    
    # Reshape for CNN (add channel dimension)
    # X_train = add_noise(X_train)
    X_train = X_train.reshape(-1, 13, 1)
    X_val = X_val.reshape(-1, 13, 1)
    X_test = X_test.reshape(-1, 13, 1)
    
    return X_train, X_val, X_test, y_train, y_val, y_test, scaler

# 3. Model Architecture
def create_model():
    model = models.Sequential([
        layers.Conv1D(16, 3, activation='relu', input_shape=(13, 1)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Conv1D(32, 3, activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        
        layers.Flatten(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    
    model.compile(optimizer='adam',
                 loss='binary_crossentropy',
                 metrics=['accuracy'])
    
    return model

# 4. Visualization and Evaluation Functions
def plot_training_history(history):
    print("Plotting training history...")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Plot accuracy
    ax1.plot(history.history['accuracy'])
    ax1.plot(history.history['val_accuracy'])
    ax1.set_title('Model Accuracy')
    ax1.set_ylabel('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend(['Train', 'Validation'])
    
    # Plot loss
    ax2.plot(history.history['loss'])
    ax2.plot(history.history['val_loss'])
    ax2.set_title('Model Loss')
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend(['Train', 'Validation'])
    
    plt.tight_layout()
    plt.savefig('training_history1.png')
    print("Plotting training history successfully completed.")

def plot_confusion_matrix(y_true, y_pred):
    print("Generating confusion matrix...")
    cm = confusion_matrix(y_true, y_pred.round())
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig('confusion_matrix1.png')
    print("Confusion matrix generated.")

def evaluate_model(model, X_test, y_test):
    print("\n=== Model Evaluation ===")
    
    # Get model predictions
    print("Making predictions...")
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    
    # Calculate accuracy
    print("\nModel Performance:")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Loss: {test_loss:.4f}")
    
    # Print detailed report
    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred_classes))
    
    return y_pred    

def visualize_results(y_test, y_pred):
    print("\n=== Creating Visualizations ===")
    
    # 1. Confusion Matrix
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    cm = confusion_matrix(y_test, (y_pred > 0.5).astype(int))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    
    # 2. ROC Curve
    plt.subplot(1, 2, 2)
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    
    plt.tight_layout()
    plt.savefig('model_performance1.png')
    # plt.show()

    print("Visualizations created.")


# def plot_waveform_and_spectrogram(audio_file , type):
#     y, sr = librosa.load(audio_file)
#     plt.figure(figsize=(14, 5))
#     librosa.display.waveshow(y, sr=sr, alpha=0.6)
#     plt.title('Waveform of '+type+' audio')
#     plt.tight_layout()
#     plt.savefig('waveform of'+type+'.png')
#     # plt.show()
    
#     plt.figure(figsize=(14, 5))
#     S = librosa.feature.melspectrogram(y=y, sr=sr)
#     S_dB = librosa.power_to_db(S, ref=np.max)
#     librosa.display.specshow(S_dB, sr=sr, x_axis='time', y_axis='mel')
#     plt.colorbar(format='%+2.0f dB')
#     plt.title('Mel-Frequency Spectrogram of '+type+' audio')
#     # plt.show()
#     plt.tight_layout()
#     plt.savefig('spectogram of'+type+'.png')
#     return Audio(audio_file)

def plot_class_distribution(labels, aug):
    unique, counts = np.unique(labels, return_counts=True)
    # Map the numerical labels to names
    label_names = ["normal" if label == 0 else "depressed" for label in unique]
    plt.figure(figsize=(6, 5))
    sns.barplot(x=label_names, y=counts, palette="viridis")
    plt.title('Class Distribution')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig('class distribution ' + aug + '.png')

# 5. Main Execution
def main():
    # Load combined dataset directly
    print("Loading dataset...")
    dataset_path = r"D:\projects\clg work\Project\mfcc_features_more_detailed.csv"
    dataset_path1 = r"D:\projects\clg work\Project\combined_dataset_kaggle_data.csv"
    data = pd.read_csv(dataset_path)
    data1 = pd.read_csv(dataset_path1)
    
    # Separate features and labels
    features = data.drop('label', axis=1)
    labels = data['label'].values
    labels1 = data1['label'].values
    
    # Preprocess data
    print("Preprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler =preprocess_data(features,
                                                                             labels)
    
    # Create and compile model
    print("Creating model...")
    model = create_model()
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            patience=5,
            restore_best_weights=True,
            monitor='val_loss',
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=3,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=64,
        callbacks=callbacks,
        verbose=1
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Evaluate model
    print("\nEvaluating model...")
    # y_pred = model.predict(X_test)
    y_pred = evaluate_model(model, X_test, y_test)
    visualize_results(y_test, y_pred)


    plot_confusion_matrix(y_test, y_pred)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred.round()))

    print("\nclass distribution:")
    plot_class_distribution(labels , "after augmentation")
    plot_class_distribution(labels1 , "before augmentation")




    report_dict = classification_report(y_test, y_pred.round(), output_dict=True)
    df_report = pd.DataFrame(report_dict).transpose()

    # fig, ax = plt.subplots(figsize=(8, 6))
    # ax.axis('tight')
    # ax.axis('off')


    # plt.savefig("classification_report.png")


    # Save model and scaler
    print("\nSaving model and scaler...")
    model.save('depression_detection_model.keras')
    joblib.dump(scaler, 'scaler.pkl')
    print("Model and scaler saved successfully!")

if __name__ == "__main__":
    main()

