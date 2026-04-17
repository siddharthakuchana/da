import pandas as pd
import matplotlib.pyplot as plt

# ---------------- DATASET ----------------
data = {
    'Day': [1,2,3,4,5,6,7],
    'Sales': [200,250,220,240,300,280,320],
    'Profit': [50,70,60,65,90,80,100]
}

df = pd.DataFrame(data)
print("Dataset:\n", df)

# ---------------- MENU ----------------
print("\nChoose Graph Type:")
print("1. Bar Graph (Sales)")
print("2. Column Graph (Profit)")
print("3. Line Graph (Sales Trend)")
print("4. Scatter Plot (Sales vs Profit)")
print("5. Show All Graphs")

choice = int(input("Enter your choice (1-5): "))

# ---------------- GRAPH SELECTION ----------------
if choice == 1:
    plt.bar(df['Day'], df['Sales'])
    plt.xlabel("Day")
    plt.ylabel("Sales")
    plt.title("Bar Graph - Sales per Day")
    plt.show()

elif choice == 2:
    plt.bar(df['Day'], df['Profit'])
    plt.xlabel("Day")
    plt.ylabel("Profit")
    plt.title("Column Graph - Profit per Day")
    plt.show()

elif choice == 3:
    plt.plot(df['Day'], df['Sales'], marker='o')
    plt.xlabel("Day")
    plt.ylabel("Sales")
    plt.title("Line Graph - Sales Trend")
    plt.show()

elif choice == 4:
    plt.scatter(df['Sales'], df['Profit'])
    plt.xlabel("Sales")
    plt.ylabel("Profit")
    plt.title("Scatter Plot - Sales vs Profit")
    plt.show()

elif choice == 5:
    # Create multiple graphs in one window
    plt.figure(figsize=(10,8))

    # Bar Graph
    plt.subplot(2,2,1)
    plt.bar(df['Day'], df['Sales'])
    plt.title("Bar - Sales")

    # Column Graph
    plt.subplot(2,2,2)
    plt.bar(df['Day'], df['Profit'])
    plt.title("Column - Profit")

    # Line Graph
    plt.subplot(2,2,3)
    plt.plot(df['Day'], df['Sales'], marker='o')
    plt.title("Line - Sales Trend")

    # Scatter Plot
    plt.subplot(2,2,4)
    plt.scatter(df['Sales'], df['Profit'])
    plt.title("Scatter - Sales vs Profit")

    plt.tight_layout()
    plt.show()

else:
    print("Invalid choice ❌")
