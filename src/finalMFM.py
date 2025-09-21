import datasets

import pandas as pd

import numpy as np

import matplotlib

import IPython.display as display

print("ready!")





from datasets import load_dataset

# Load the complete dataset (Arabic dataset from huggface)

dataset =load_dataset("KFUPM-JRCAI/arabic-generated-abstracts")



# inspect the dataset structure (columns, data types).

# 1] Keys (subset)

print("Data Keys:")

print(dataset.keys())





# create a variable for each data set, to get on it without the need of write the full code in each print or method.



by_polishing = dataset["by_polishing"]

from_title = dataset["from_title"]

from_title_and_content = dataset["from_title_and_content"]



total_rows = sum(split.num_rows for split in dataset.values())

print("Total rows in Full Dataset:", total_rows)





# 5] Subsets data types:

print("Data Subsets data types:","\n")

print("by_polishing:" , by_polishing.features,"\n")

print("from_title:" , from_title.features,"\n")

print("from_title_and_content:" , from_title_and_content.features,"\n")





# 6] Sample from Subsets:

print("Data Subsets Sample:","\n")



print("by_polishing 1: " ,by_polishing[0])     # first row

print("by_polishing 2: " ,by_polishing[:2])    # first and second row



print("from_title 1: " ,from_title[0])     # first row

print("from_title 2: " ,from_title[:2])    # first and second row



print("from_title_and_content: 1: " ,from_title_and_content[0])     # first row

print("from_title_and_content: 2" ,from_title_and_content[:2])    # first and second row





# after we get all information about the dataset, now we need to convert the dataset tables into data frame

# convert tables to dataframe using method 1 because we get dataset from huggface

by_polishing_df = by_polishing.to_pandas()

from_title_df = from_title.to_pandas()

from_title_and_content_df = from_title_and_content.to_pandas()

# to pring all dataset togther

dataset_df = pd.concat([by_polishing_df, from_title_df, from_title_and_content_df], ignore_index=True)



# save dataframes into excel file with csv type, using .to_csv() function which comes from pandas library, and now we can use it because we alredy convert dataset into pandas variable.

by_polishing_df.to_csv("by_polishing.csv", index=False, encoding="utf-8-sig")

from_title_df.to_csv("from_title.csv", index=False, encoding="utf-8-sig")

from_title_and_content_df.to_csv("from_title_and_content.csv", index=False, encoding="utf-8-sig")

dataset_df.to_csv("dataset.csv", index=False, encoding="utf-8-sig")



print("Now all dataframes saved as excel file with csv datatype in our project folder")



# read data from excel نقرأها من ملف الاكسل

by_polishing = pd.read_csv('by_polishing.csv' ,encoding="utf-8-sig")

from_title = pd.read_csv('from_title.csv',encoding="utf-8-sig")

from_title_and_content = pd.read_csv('from_title_and_content.csv',encoding="utf-8-sig")

dataset = pd.read_csv('dataset.csv',encoding="utf-8-sig")



# print tables طباعة الجداول كلها 

print(by_polishing)

print(from_title)

print(from_title_and_content)





print("اول صف:\n")

print("by_polishing:\n", by_polishing.head(0))

print("from_title:\n", from_title.head(0))

print("from_title_and_content:\n", from_title_and_content.head(0))





print("معلومات الأعمدة وأنواع البيانات:\n")

print("by_polishing:\n", by_polishing.dtypes)

print("from_title:\n", from_title.dtypes)

print("from_title_and_content:\n", from_title_and_content.dtypes)





print("وصف كامل للبيانات:\n")

#السبب اننا ما حطينا دالة (info) جوت الطباعة لانها دالة تطبع من نفسها

print("\nby_polishing:\n")

by_polishing.info()



print("\nfrom_title:\n")

from_title.info()



print("\nfrom_title_and_content:\n")

from_title_and_content.info()



# human او AI اصنف العواميد في الجدول هل هي

# the reason way we create anthor dataframe is because we want to add the 'Lable' column.



#by polishing method

human_df_by_polishing = pd.DataFrame({

    'text': by_polishing['original_abstract'],

    'label': 'human',

    'genration_method':'by_polishing'

})



ai_df_by_polishing = pd.DataFrame({

    'text': pd.concat([

        by_polishing['allam_generated_abstract'],

        by_polishing['jais_generated_abstract'],

        by_polishing['llama_generated_abstract'],

        by_polishing['openai_generated_abstract']

    ]),

    'label': 'ai',

    'genration_method':'by_polishing'

})

#___________________________________________________________

#from title method

human_df_from_title = pd.DataFrame({

    'text': from_title['original_abstract'],

    'label': 'human',

    'genration_method':'from_title'

})



ai_df_from_title = pd.DataFrame({

    'text': pd.concat([

        from_title['allam_generated_abstract'],

        from_title['jais_generated_abstract'],

        from_title['llama_generated_abstract'],

        from_title['openai_generated_abstract']

    ]),

    'label': 'ai',

     'genration_method':'from_title'

})

#___________________________________________________________

#from_title_and_content method

human_df_from_title_and_content = pd.DataFrame({

    'text': from_title_and_content['original_abstract'],

    'label': 'human',

    'genration_method':'from_title_and_content'

})



ai_df_from_title_and_content = pd.DataFrame({

    'text': pd.concat([

        from_title_and_content['allam_generated_abstract'],

        from_title_and_content['jais_generated_abstract'],

        from_title_and_content['llama_generated_abstract'],

        from_title_and_content['openai_generated_abstract']

    ]),

    'label': 'ai',

    'genration_method':'from_title_and_content'

})





# Dataframe merge



# دمج البيانات  human or ai /وحده DataFrame دمجها بـ

merged_df_dataset = pd.concat([human_df_by_polishing, ai_df_by_polishing,human_df_from_title, ai_df_from_title,human_df_from_title_and_content, ai_df_from_title_and_content], ignore_index=True)

merged_df_by_polishing = pd.concat([human_df_by_polishing, ai_df_by_polishing], ignore_index=True)

merged_df_from_title = pd.concat([human_df_from_title, ai_df_from_title], ignore_index=True)

merged_df_from_title_and_content = pd.concat([human_df_from_title_and_content, ai_df_from_title_and_content], ignore_index=True)



# Datafram Print



# عرض أول صفوف من البيانات الجديدة

print("\nالبيانات هل هي من كتابة الانسان ام الذكاء الاصطناعي:")

# عدد الصفوف سار الضعف 5 مرات مثلا بدلا من 2851 اصبح 2581*5=14255؟

#لانو الاعمدة بدال ما كانت جمب بعض سارت تحت بعض يعني بدال ما تكون اعمدة سارت صفوف لاننا عدلنا ترتيب شكل الجدول



print("Full Dataset")

display.display(merged_df_dataset)



print("by_polishing")

display.display(merged_df_by_polishing)



print("\nfrom_title")

display.display(merged_df_from_title)



print("\nfrom_title_and_content")

display.display(merged_df_from_title_and_content)





#number of human vs AI

print("\nnumber of human vs AI\n")



print("Full Dataset")

print(merged_df_dataset['label'].value_counts())



print("\nby_polishing")

print(merged_df_by_polishing['label'].value_counts())



print("\nfrom_title")

print(merged_df_from_title['label'].value_counts())



print("\nfrom_title_and_content")

print(merged_df_from_title_and_content['label'].value_counts())





#number of missing values

print("\nnumber of missing values\n")

print("Full Dataset")

print(dataset.isnull().sum())





#number of duplicates:

print("\nnumber of duplicates:\n")

print("Full Dataset")

print(dataset.duplicated().sum())



#number of inconsistencies:

print("\n inconsistencies:\n")

print("\ndataset")

print(merged_df_dataset['label'].unique())



import re



def extract_strange(text):

    if pd.isna(text):

        return []

    # هنا ما نسمح إلا بالعربي (\u0600-\u06FF) + الأرقام (0-9) + المسافات

    return re.findall(r"[^0-9\u0600-\u06FF\s]", str(text))



all_strange_chars = set()

total_strange_count = 0

for t in merged_df_dataset["text"].dropna():

    chars = extract_strange(t)

    all_strange_chars.update(chars)

    total_strange_count += len(chars)







# عرض النتائج

print("\n الرموز الغريبة الموجودة (بدون تكرار):")

print(all_strange_chars)



print("\n مجموع عدد الرموز الغريبة في النصوص:")

print(total_strange_count)



print(" القيم الفريدة في عمود label:")

print(merged_df_dataset['label'].unique())



print("\n توزيع القيم في عمود label:")

print(merged_df_dataset['label'].value_counts())





# إنشاء عمود جديد لطول النص

merged_df_dataset['text_length'] = merged_df_dataset['text'].astype(str).str.len()



print("\n إحصائيات طول النصوص:")

print(merged_df_dataset['text_length'].describe())



# مثال: نعرض أطول وأقصر النصوص

print("\n أقصر نص:")

print(merged_df_dataset.loc[merged_df_dataset['text_length'].idxmin(), 'text'])



print("\n أطول نص:")

print(merged_df_dataset.loc[merged_df_dataset['text_length'].idxmax(), 'text'])



















