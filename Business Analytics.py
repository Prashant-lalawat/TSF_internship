print("....Algorithm........")
##import required libaries
##read the file
##important information about the file 
##showing duplicate values
##finding the co-realtion
## plotting the graph

print("........CODE.............")

import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt

df = pd.read_csv('C:/Users/HP/Desktop/Internship/SampleSuperstore.csv')
print(df)
print(df.info())
print(df.describe())
print(df.duplicated().sum())
print(df.drop_duplicates(keep='first', inplace = True))
print(df)

corr = df.corr()
print(corr)
print(sb.countplot(df['Ship Mode']))
print(sb.countplot(df['Segment']))
print(sb.countplot(df['State'],order=(df['State'].value_counts().head(30)).index))
print(plt.xticks(rotation=90))

plt.figure(figsize=(40,25))
sb.barplot(df['Sub-Category'], df['Profit'])

state = df.groupby("State")[["Sales","Profit"]].sum().sort_values(by = "Sales",
                                                                  ascending = False)

plt.figure(figsize=(50,50))
state[:25].plot(kind = "bar", color = ["k", "yellow"],edgecolor = "#000000")
plt.title("Profit Or Loss & Sales of top 25 States")
plt.xlabel("States")
plt.ylabel("Total profit / loss and sales")
plt.grid(True)
state[25:].plot(kind = "bar", color = ["k","yellow"],edgecolor = "#000000")
plt.title("Profit Or Loss & sales of the least economic states")
plt.xlabel("States")
plt.ylabel("Total profit / loss and sales")
plt.grid(True)

df.groupby('Sub-Category')['Profit', 'Sales'].sum().plot.bar(color = ['b','k'])

pd.DataFrame(df.groupby('State').sum())['Discount'].sort_values(ascending = True)

pd.DataFrame(df.groupby('State').sum())['Profit'].sort_values(ascending = True)

plt.figure(figsize = (12,6))
sb.lineplot('Discount' , 'Profit', data = df, color = 'g', label = 'Discount')
plt.legend()
