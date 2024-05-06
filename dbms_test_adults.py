import pandas as pd
from openai import OpenAI

client = OpenAI(api_key='your-api-keu')

# Read the datasets
clean_df = pd.read_csv('adults_clean.csv')
dirty_df = pd.read_csv('adults_dirty.csv')

# Convert the first column to string in both clean_df and dirty_df
clean_df['row_id'] = clean_df['row_id'].astype(str)
dirty_df['row_id'] = dirty_df['row_id'].astype(str)

prompt_text = "Consider a dataset with the following columns: row_id,age,workclass,education,maritalstatus,occupation,relationship,race,sex,hoursperweek,country,income.\nHere are 5 examples of clean rows: 0,31-50,Private,Prof-school,Never-married,Prof-specialty,Not-in-family,White,Female,40,United-States,MoreThan50K \n1,>50,Private,HS-grad,Married-civ-spouse,Craft-repair,Husband,White,Male,16,United-States,LessThan50K \n2,>50,Private,Some-college,Married-civ-spouse,Exec-managerial,Husband,White,Male,55,United-States,MoreThan50K \n3,22-30,Private,HS-grad,Never-married,Handlers-cleaners,Own-child,White,Male,40,United-States,LessThan50K.\n\n Errors can be common typos on a qwerty keyboard, missing values and implicitly missing values ex: age = 0, as well as values replaced with values from other columns. "
column_text = """The values that the columns can take are the following -  age: 18-21, 22-30, 31-50, <18, >50, 
workclass: Federal-gov, Local-gov, Private, Self-emp-inc, Self-emp-not-inc, State-gov, Without-pay,
education: 10th, 11th, 12th, 5th-6th, 7th-8th, 9th, Assoc-acdm, Assoc-voc, Bachelors, Doctorate, HS-grad, Masters, Preschool, Prof-school, Some-college,
maritalstatus: Divorced, Married-civ-spouse, Married-spouse-absent, Never-married, Separated, Widowed,
occupation: Adm-clerical, Craft-repair, Exec-managerial, Farming-fishing, Handlers-cleaners, Machine-op-inspct, Other-service, Priv-house-serv, Prof-specialty, Protective-serv, Sales, Tech-support, Transport-moving,
relationship: Husband, Not-in-family, Other-relative, Own-child, Unmarried, Wife,
race: Amer-Indian-Eskimo, Asian-Pac-Islander, Black, Other, White,
sex: Female, Male,
hoursperweek: 1, 10, 11, 12, 14, 15, 16, 17, 18-21, 2, 22, 23, 24, 25, 26, 27, 28, 3, 30, 32, 33, 35, 36, 37, 38, 39, 40, 42, 43, 44, 45, 46, 47, 48, 49, 5, 50, 52, 53, 55, 56, 57, 58, 6, 60, 63, 64, 65, 7, 70, 72, 75, 8, 80, 86, 88, 9, 90, 98, 99,
country: Canada, China, Cuba, Dominican-Republic, El-Salvador, England, Germany, Guatemala, Iran, Italy, Jamaica, Japan, Laos, Mexico, Nicaragua, Philippines, Poland, Portugal, Puerto-Rico, South, Taiwan, United-States, Vietnam,
income: LessThan50K, MoreThan50K"""

def evaluate_responses(response, dirty_row, clean_row):
    dirty_values = dirty_row.values
    clean_values = clean_row.values
    response_values = response.choices[0].text.strip().replace("Output: ", "").split(',')
    
    # Initialize counts for each cell
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    cleaned_correctly = 0
    cleaned_incorrectly = 0
    
    # Compare each cell individually
    for dirty_val, clean_val, response_val in zip(dirty_values, clean_values, response_values):
        if dirty_val == clean_val:
            if response_val == clean_val:
                true_positives += 1
            elif response_val !=clean_val:
                false_negatives += 1 #program says the data is dirty  
                print(f"False negative! Dirty val=cleaned val: {dirty_val} Cleaned val: {response_val}")
    
        
        elif dirty_val!=clean_val:
            if response_val!=dirty_val:
                true_negatives+=1
                if response_val == clean_val: #our program cleans the same way
                    cleaned_correctly+=1
                else:
                    cleaned_incorrectly+=1
                    print(f"Cleaned incorrectly! Clean value: {clean_val} Our response: {response_val}")
            elif response_val==dirty_val: #program does not identify dirty data
                false_positives+=1
                print(f"False positive! Dirty val: {dirty_val} Cleaned val=response val: {response_val}")

    return true_positives, true_negatives, false_positives, false_negatives, cleaned_correctly, cleaned_incorrectly


true_positives_total = 0
true_negatives_total = 0
false_positives_total = 0
false_negatives_total = 0
cleaned_correctly_total = 0
cleaned_incorrectly_total = 0

for idx, dirty_row in dirty_df.iterrows():
    response = client.completions.create(
        model="gpt-3.5-turbo-instruct",
        prompt=prompt_text + column_text + f"for each row in the file, clean the row if there are errors and respond as a comma separated row with each entry corresponding to the clean attribute in the input row. Double check your answer before responding. Input:  {dirty_row.to_string()}",
        max_tokens=60,
        temperature=0)
    
    # Find the corresponding clean row
    clean_row = clean_df.iloc[idx]
    
    # Evaluate responses
    tp, tn, fp, fn, cc, ci= evaluate_responses(response,dirty_row, clean_row)
    
    # Accumulate totals
    true_positives_total += tp
    true_negatives_total += tn
    false_positives_total += fp
    false_negatives_total += fn
    cleaned_correctly_total += cc
    cleaned_incorrectly_total += ci
    if(idx%50 == 0):
        print(f"We are on iteration: {idx}")


# Output the results
print(f"True Positives: {true_positives_total}")
print(f"True Negatives: {true_negatives_total}")
print(f"False Positives: {false_positives_total}")
print(f"False Negatives: {false_negatives_total}")
print(f"Cleaned correctly: {cleaned_correctly_total}")
print(f"Cleaned incorrectly: {cleaned_incorrectly_total}")

precision = true_positives_total/ (true_positives_total + false_positives_total)
recall = true_positives_total/ (true_positives_total + false_negatives_total)
cleaning_accuracy = cleaned_correctly_total/(cleaned_correctly_total+cleaned_incorrectly_total)
total_accuracy = (cleaned_correctly_total + true_positives_total)/ (true_positives_total + true_negatives_total + false_positives_total + false_negatives_total)

print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"Cleaning accuracy: {cleaning_accuracy}")
print(f"Total accuracy: {total_accuracy}")
