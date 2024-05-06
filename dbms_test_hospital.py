import pandas as pd
from openai import OpenAI
import numpy as np

client = OpenAI(api_key='your-api-key')

# Read the datasets
clean_df = pd.read_csv('hospital_clean_rows.csv')
dirty_df = pd.read_csv('hospital_dirty.csv')

# Convert the first column to string in both clean_df and dirty_df
clean_df['ProviderNumber'] = clean_df['ProviderNumber'].astype(str)
dirty_df['ProviderNumber'] = dirty_df['ProviderNumber'].astype(str)
clean_df['ZipCode'] = clean_df['ZipCode'].astype(str)
dirty_df['ZipCode'] = dirty_df['ZipCode'].astype(str)
clean_df['PhoneNumber'] = clean_df['PhoneNumber'].astype(str)
dirty_df['PhoneNumber'] = dirty_df['PhoneNumber'].astype(str)

prompt_text= "Consider a dataset with the following columns: ProviderNumber,HospitalName,Address1,Address2,Address3,City,State,ZipCode,CountyName,PhoneNumber,HospitalType,HospitalOwner,EmergencyService,Condition,MeasureCode,MeasureName,Score,Sample,Stateavg\nHere are some examples of clean rows:\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-card-2,surgery patients who were taking heart drugs called beta blockers before coming to the hospital who were kept on the beta blockers during the period just before and after their surgery,,,al_scip-card-2\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-1,surgery patients who were given an antibiotic at the right time (within one hour before surgery) to help prevent infection,,,al_scip-inf-1\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-2,surgery patients who were given the  right kind  of antibiotic to help prevent infection,,,al_scip-inf-2\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-3,surgery patients whose preventive antibiotics were stopped at the right time (within 24 hours after surgery),,,al_scip-inf-3\n\nfor each row of the following rows, check if there is an error in the row and give a clean version of the row.\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-card-2,surgery patients who were taking heart drugs caxxed beta bxockers before coming to the hospitax who were kept on the beta bxockers during the period just before and after their surgery,,,al_scip-card-2\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-1,surgery patients who were given an antibiotic at the right time (within one hour before surgery) to help prevent infection,,,al_scip-inf-1\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-2,surgery patients who were given the  right kind  of antibiotic to help prevent infection,,,al_scip-inf-2\n10018,callahan eye foundation hospital,1720 university blvd,,,birminghxm,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-3,surgery patients whose preventive antibiotics were stopped at the right time (within 24 hours after surgery),,,al_scip-inf-3\n10018,callahan eye foundation hospital,1720 university blvd,,,birmingham,al,35233,jefferson,2053258100,acute care hospitals,voluntary non-profit - private,yes,surgical infection prevention,scip-inf-4,all heart surgery patients whose blood sugar (blood glucose) is kept under good control in the days right after surgery,,,al_scip-inf-4n\n Errors can be common typos on a qwerty keyboard, missing values and implicitly missing values ex: age = 0, as well as values replaced with values from other columns"
column_text = """Stateavg column values: al_scip-card-2, al_scip-inf-1, al_scip-inf-2, al_scip-inf-3, al_scip-inf-4, al_scip-inf-6, al_scip-vte-1, al_scip-vte-2, al_ami-1, al_ami-2, al_ami-3, al_ami-4, al_ami-5, al_ami-7a, al_ami-8a, al_hf-1, al_hf-2, al_hf-3, al_hf-4, al_pn-2, al_pn-3b, al_pn-4, al_pn-5c, al_pn-6, al_pn-7, al_cac-2, al_cac-3, ak_hf-1, ak_hf-2, ak_hf-3, ak_hf-4, ak_pn-2, ak_pn-3b, ak_pn-4, ak_pn-5c, ak_pn-6, ak_pn-7, ak_scip-card-2, ak_scip-inf-1, ak_scip-inf-2, ak_scip-inf-3, ak_scip-inf-4, ak_scip-inf-6, ak_scip-vte-1, ak_scip-vte-2, ak_ami-1, ak_ami-2, al_cac-1. 
MeasureCode column values: scip-card-2, scip-inf-1, scip-inf-2, scip-inf-3, scip-inf-4, scip-inf-6, scip-vte-1, scip-vte-2, ami-1, ami-2, ami-3, ami-4, ami-5, ami-7a, ami-8a, hf-1, hf-2, hf-3, hf-4, pn-2, pn-3b, pn-4, pn-5c, pn-6, pn-7, cac-2, cac-3, cac-1,
Score column is empty for 16.7 percent of values, Sample column is empty for 6.0 percent of values."""

def get_unique_values(data):
    unique_values = {}
    for col in data.columns:
        unique_values[col] = sorted(data[col].fillna('Null value').unique()) # Fill NaN values with 'Null value'
    response = ""
    for col, values in unique_values.items():
        response += f"{col}: {', '.join(map(str, values))}\n"
    return response

def evaluate_responses(response, dirty_row, clean_row):
    dirty_values = dirty_row.values
    dirty_values= np.where(pd.isnull(dirty_values), 'Null value', dirty_values)
    clean_values = clean_row.values
    clean_values= np.where(pd.isnull(clean_values), 'Null value', clean_values)
    #print(dirty_values)
    #print(clean_values)
    response_values = response.choices[0].text.strip().replace("Output: ", "").split(',')
    response_values = ["Null value" if x == '' else x for x in response_values]
    #print(response_values)
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
        prompt=prompt_text + column_text + f"for each row in the file, clean the row if there are errors and respond as a comma separated row with each entry corresponding to the clean attribute in the input row. Double check your answer before responding. No line breaks please. Finish writing. Input:  {dirty_row.to_string()}",
        max_tokens=150,
        temperature=0)
    
    #print(response.choices[0].text)
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
