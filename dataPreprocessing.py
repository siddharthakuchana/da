import pandas as pd

# ---------------- DATASET ----------------
data = {
    'Age': [25, None, 35, 50, 23, None, 60, 55, 30, 48],
    'BP': [120, 140, None, 150, 110, 135, 160, None, 125, 145],
    'Cholesterol': [200, 240, 220, None, 180, 230, 270, 250, 210, None]
}

df = pd.DataFrame(data)
print("Original Dataset:\n", df)

# ---------------- MENU ----------------
while True:
    print("\n--- Data Preprocessing Menu ---")
    print("1. Handle Missing Values")
    print("2. Noise Detection & Removal")
    print("3. Remove Redundant Data (Duplicates)")
    print("4. Show Dataset")
    print("5. Exit")

    choice = int(input("Enter your choice (1-5): "))

    # ---------------- 1. Missing Values ----------------
    if choice == 1:
        print("\nHandling Missing Values...")
        df.fillna(df.mean(numeric_only=True), inplace=True)
        print("Missing values replaced with mean:\n", df)

    # ---------------- 2. Noise Removal ----------------
    elif choice == 2:
        print("\nRemoving Noise (Outliers)...")
        
        # Simple rule: remove values outside range
        df = df[(df['Age'] >= 20) & (df['Age'] <= 60)]
        df = df[(df['BP'] >= 100) & (df['BP'] <= 160)]
        df = df[(df['Cholesterol'] >= 150) & (df['Cholesterol'] <= 300)]

        print("Noise removed dataset:\n", df)

    # ---------------- 3. Remove Duplicates ----------------
    elif choice == 3:
        print("\nRemoving Duplicate Data...")
        df.drop_duplicates(inplace=True)
        print("Duplicates removed:\n", df)

    # ---------------- 4. Show Dataset ----------------
    elif choice == 4:
        print("\nCurrent Dataset:\n", df)

    # ---------------- EXIT ----------------
    elif choice == 5:
        print("Exiting program...")
        break

    else:
        print("Invalid choice ❌")
