import pandas as pd

# Load the dataset
data = pd.read_csv('hospital_clean_rows.csv')

# Function to generate string response
def get_unique_values(data):
    unique_values = {}
    for col in data.columns:
        unique_values[col] = data[col].unique()
    response = ""
    for col, values in unique_values.items():
        response += f"{col}: {', '.join(map(str, values))}\n"
    return response

# Generate string response
string_response = get_unique_values(data)

#print(string_response)


# Calculate percentage of null values for each column
null_percentage = (data.isnull().mean() * 100).round(2)

# Generate string response
response = "Percentage of null values for each column:\n"
for col, percentage in null_percentage.items():
    response += f"{col}: {percentage}%\n"

print(response)