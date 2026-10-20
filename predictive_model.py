import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main():
    # Simulasi data transaksi sederhana
    data = {
        'order_amount': [1500000, 2300000, 500000, 4200000, 1100000],
        'discount_given': [0.05, 0.35, 0.10, 0.02, 0.40],
        'late_payment': [0, 1, 0, 1, 1] # 1 jika terlambat bayar
    }
    
    df = pd.DataFrame(data)
    
    # Pisahkan fitur dan target
    X = df[['order_amount', 'discount_given']]
    y = df['late_payment']
    
    # Latih model sederhana
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    print("Prediksi berhasil dijalankan!")
    print(f"Akurasi Model: {accuracy_score(y_test, predictions):.2f}")

if __name__ == "__main__":
    main()
