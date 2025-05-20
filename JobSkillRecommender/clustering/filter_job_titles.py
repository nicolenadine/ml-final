import pandas as pd
import numpy as np

data = pd.read_csv('../data/raw/postings.csv')


# df['title'].to_csv("job_titles.txt", index=False, header=False)

def split_file(input_file, lines_per_file=3000):
    with open(input_file, 'r') as file:
        lines = file.readlines()

    total_lines = len(lines)
    num_files = (total_lines + lines_per_file - 1) // lines_per_file  # ceil division

    for i in range(num_files):
        chunk = lines[i * lines_per_file : (i + 1) * lines_per_file]
        output_file = f"{input_file}_part{i+1}.txt"
        with open(output_file, 'w') as out_file:
            out_file.writelines(chunk)
        print(f"Wrote {len(chunk)} lines to {output_file}")

# Example usage:
#split_file("filtered_titles_GPTraw.txt")


# combine files back  together and drop duplicates
output_file = "filter_job_titles_refined.txt"

# with open(output_file, 'w') as outfile:
#     for i in range(1, 12):  # 1 through 11
#         part_file = f"filtered.txt_part{i}.txt"
#         with open(part_file, 'r') as infile:
#             outfile.write(infile.read())
#             outfile.write('\n')  # Ensure newline between parts
#
# print(f"All parts combined into {output_file}")

# Step 2: Load output file containing list of CS/DS job titles into pandas
with open("filter_job_titles_refined.txt", "r") as f:
    lines = [line.strip() for line in f if line.strip()]  # removes blank lines

df = pd.DataFrame(lines, columns=["Job Title"])


print("Number of job titles:", len(df["Job Title"]))

#Drop duplicates and blank lines
df = df[df["Job Title"] != ""]  # Remove empty rows
unique_titles = df["Job Title"].drop_duplicates()

# Save the unique list to a file
unique_titles.to_csv("unique_job_titles.txt", index=False, header=False)

print("Number of unique job titles:", len(unique_titles))


print("Unique job titles saved to unique_job_titles.txt")

valid_titles = set(unique_titles)


filtered_df = data[data["title"].isin(valid_titles)]

print(filtered_df.info())

# save new dataset that has clearly non-tech job titles filtered out
filtered_df.to_csv("csds_filtered.csv", index=False)




# -------------   SAVED OUTPUT FOR REPORTING LATER   -------------

# postings.csv info
'''
RangeIndex: 123849 entries, 0 to 123848
Data columns (total 31 columns):
 #   Column                      Non-Null Count   Dtype  
---  ------                      --------------   -----  
 0   job_id                      123849 non-null  int64  
 1   company_name                122130 non-null  object 
 2   title                       123849 non-null  object 
 3   description                 123842 non-null  object 
 4   max_salary                  29793 non-null   float64
 5   pay_period                  36073 non-null   object 
 6   location                    123849 non-null  object 
 7   company_id                  122132 non-null  float64
 8   views                       122160 non-null  float64
 9   med_salary                  6280 non-null    float64
 10  min_salary                  29793 non-null   float64
 11  formatted_work_type         123849 non-null  object 
 12  applies                     23320 non-null   float64
 13  original_listed_time        123849 non-null  float64
 14  remote_allowed              15246 non-null   float64
 15  job_posting_url             123849 non-null  object 
 16  application_url             87184 non-null   object 
 17  application_type            123849 non-null  object 
 18  expiry                      123849 non-null  float64
 19  closed_time                 1073 non-null    float64
 20  formatted_experience_level  94440 non-null   object 
 21  skills_desc                 2439 non-null    object 
 22  listed_time                 123849 non-null  float64
 23  posting_domain              83881 non-null   object 
 24  sponsored                   123849 non-null  int64  
 25  work_type                   123849 non-null  object 
 26  currency                    36073 non-null   object 
 27  compensation_type           36073 non-null   object 
 28  normalized_salary           36073 non-null   float64
 29  zip_code                    102977 non-null  float64
 30  fips                        96434 non-null   float64
dtypes: float64(14), int64(2), object(15)
'''


# postings.csv columns
'''
Index(['job_id', 'company_name', 'title', 'description', 'max_salary',
       'pay_period', 'location', 'company_id', 'views', 'med_salary',
       'min_salary', 'formatted_work_type', 'applies', 'original_listed_time',
       'remote_allowed', 'job_posting_url', 'application_url',
       'application_type', 'expiry', 'closed_time',
       'formatted_experience_level', 'skills_desc', 'listed_time',
       'posting_domain', 'sponsored', 'work_type', 'currency',
       'compensation_type', 'normalized_salary', 'zip_code', 'fips'],
      dtype='object')
'''

# Number of job titles: 9299   <- in filter_job_titles_refined.txt
# Number of unique job titles: 5736  <- saved in unique_job_titles.txt


# data set after filtering out non cs/ds roles  <- saved in data_csds_filtered.csv
'''
<class 'pandas.core.frame.DataFrame'>
Index: 11888 entries, 13 to 123843
Data columns (total 31 columns):
 #   Column                      Non-Null Count  Dtype  
---  ------                      --------------  -----  
 0   job_id                      11888 non-null  int64  
 1   company_name                11768 non-null  object 
 2   title                       11888 non-null  object 
 3   description                 11888 non-null  object 
 4   max_salary                  3265 non-null   float64
 5   pay_period                  3551 non-null   object 
 6   location                    11888 non-null  object 
 7   company_id                  11769 non-null  float64
 8   views                       11691 non-null  float64
 9   med_salary                  286 non-null    float64
 10  min_salary                  3265 non-null   float64
 11  formatted_work_type         11888 non-null  object 
 12  applies                     4595 non-null   float64
 13  original_listed_time        11888 non-null  float64
 14  remote_allowed              2659 non-null   float64
 15  job_posting_url             11888 non-null  object 
 16  application_url             5960 non-null   object 
 17  application_type            11888 non-null  object 
 18  expiry                      11888 non-null  float64
 19  closed_time                 116 non-null    float64
 20  formatted_experience_level  8738 non-null   object 
 21  skills_desc                 185 non-null    object 
 22  listed_time                 11888 non-null  float64
 23  posting_domain              5296 non-null   object 
 24  sponsored                   11888 non-null  int64  
 25  work_type                   11888 non-null  object 
 26  currency                    3551 non-null   object 
 27  compensation_type           3551 non-null   object 
 28  normalized_salary           3551 non-null   float64
 29  zip_code                    8779 non-null   float64
 30  fips                        8119 non-null   float64
dtypes: float64(14), int64(2), object(15)
memory usage: 2.9+ MB
'''