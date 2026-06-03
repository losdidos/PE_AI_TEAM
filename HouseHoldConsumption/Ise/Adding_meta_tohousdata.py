import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



def Maak_whederdataHous_met_Meta(nummer):
    mun = str(nummer)
    df = pd.read_csv(f"sprint3/data/whederdataHous_{nummer}.csv")

    df["Occupancy"] = metaData[mun]["Occupancy"]
    df["Appliances Owned"] = metaData[mun]["Appliances Owned"]
    df["Detached"] = metaData[mun]["Detached"]
    df["Semi-detached"] = metaData[mun]["Semi-detached"]
    df["Mid-terrace"] = metaData[mun]["Mid-terrace"]
    df["Size"] = metaData[mun]["Size"]

    print(f"////////////sprint3/data/whederdataHous_{mun}.csv////////////")
    print(df.head())

    df.to_csv(f"sprint3/data/whederdataHous_{mun}_metMetaData.csv")
    return df













metaData = {
    "1": {
        "Occupancy": 2,
        "Appliances Owned": 35,

        "Detached":1,
        "Semi-detached":0,
        "Mid-terrace":0,

        "Size": 4


    },



    "2": {
        "Occupancy": 4,
        "Appliances Owned": 15,

        "Detached":0,
        "Semi-detached":1,
        "Mid-terrace":0,

        "Size": 3},



    "3": {
        "Occupancy": 2,
        "Appliances Owned": 27,

        "Detached":1,
        "Semi-detached":0,
        "Mid-terrace":0,

        "Size": 3},



    "5": {
        "Occupancy": 4,
        "Appliances Owned": 44,

        "Detached":0,
        "Semi-detached":0,
        "Mid-terrace":1,

        "Size": 4},



    "6": {
        "Occupancy": 2,
        "Appliances Owned": 49,

        "Detached":1,
        "Semi-detached":0,
        "Mid-terrace":0,

        "Size": 4},



    "7": {
        "Occupancy": 4,
        "Appliances Owned": 25,

        "Detached":1,
        "Semi-detached":0,
        "Mid-terrace":0,

        "Size": 3},



    "8": {
        "Occupancy": 2,
        "Appliances Owned": 35,

        "Detached":1,
        "Semi-detached":0,
        "Mid-terrace":0,

        "Size": 2},

}

"""
" Detached			"
" Semi-detached	"

" Mid-terrace		"



"""



csv = {1,2,3,5,6,7,8}


dfs = []

for nummer in csv:
    dfs.append(Maak_whederdataHous_met_Meta(nummer))



df_all = pd.concat(dfs)


print(df_all.head())

df_all.to_csv(f"sprint3/data/whederdataHous_all_met_metadata.csv")





















