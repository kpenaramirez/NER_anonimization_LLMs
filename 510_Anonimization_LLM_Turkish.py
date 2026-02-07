from transformers import AutoTokenizer, AutoModelForTokenClassification
from transformers import pipeline
import pandas as pd
import numpy as np
import glob

#English trained model:
tokenizer = AutoTokenizer.from_pretrained("dslim/bert-base-NER")
model = AutoModelForTokenClassification.from_pretrained("dslim/bert-base-NER")
#Spanish trained model plus cased (normalization fixed)
model = AutoModelForTokenClassification.from_pretrained("skimai/spanberta-base-cased-ner-conll02")
tokenizer = AutoTokenizer.from_pretrained("skimai/spanberta-base-cased-ner-conll02")


#Testing single name
nlp = pipeline("ner", model=model, tokenizer=tokenizer)
example = "My name is Wolfgang and I live in Berlin"
#Results
ner_results = nlp(example)
print(ner_results)

#Entry datasets
## UA database
df = pd.read_csv("Gender.csv")
## Spaniard female
df = pd.read_csv("female_names.csv")
## Spaniard male
df = pd.read_csv("male_names.csv")
## USA names 1880-2021
files_sorted = sorted(glob.glob( 'names_USA/*.txt' ))
with open( 'Names_English_sorted.txt', 'w' ) as result:
    for file_ in files_sorted:
        for line in open( file_, 'r' ):
            result.write( line )
df = pd.read_csv("Names_English.txt", sep=",")
df.columns =['name','gender','freq']
df.to_csv("Names_English.csv")
##

print(df.head(10))
items = df.name
#Random test
item = np.random.choice(items,5)
# item = np.random.choice(items, p = [0.2, 0.5, 0.2, 0.1])
ner_results = nlp(item[1])
print(ner_results)

#Massive test with different capitalizations
newnames = df.name
newnames_list = list(newnames)
newnames_list?
#Full caps
#for i in range(len(newnames_list)):
#    newnames_list[i] = newnames_list[i].upper()
#Full lower
#for i in range(len(newnames_list)):
#    newnames_list[i] = newnames_list[i].lower()
#First letter
for i in range(len(newnames_list)):
    newnames_list[i] = newnames_list[i].capitalize()

print(newnames_list[0:10])
newnames_list?
new_all = nlp(newnames_list)

#Avoid empty elements
new_all = list(filter(None, new_all))
#Get the result as dataframe
group = []
score = []
name = []
for i in range(len(new_all)):
    group.append(new_all[i][0]['entity_group'])
    score.append(new_all[i][0]['score'])
    name.append(new_all[i][0]['word'])

#Getting and saving the results
df = pd.DataFrame({'name':name, 'group':group, 'score':score})
print(df)
df.to_csv("final.csv")
df.groupby("group").size()
