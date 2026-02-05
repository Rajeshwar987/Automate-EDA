# Interactive GenAI-Powered EDA Tool

A lightweight, interactive Exploratory Data Analysis (EDA) script for
**any CSV dataset**, enhanced with an optional **GenAI-generated
business summary** using the OpenAI API.

This tool is designed for quick, first-pass understanding of unfamiliar
datasets before deeper analysis or modeling.

------------------------------------------------------------------------

## Features

-   Works with any CSV file
-   Automatic encoding fallback (`utf-8`, `latin1`, `cp1252`)
-   Cleans column name whitespace issues
-   Basic categorical and numerical statistics
-   Multiple visualizations (histogram, boxplot, bar chart, scatter
    plot, correlation heatmap)
-   Non-blocking plot handling
-   Optional 2‑sentence GenAI business insight

------------------------------------------------------------------------

## GenAI Prompt Used

The script dynamically constructs the following prompt at runtime:

    Analyze this dataset: {rows} rows x {columns} columns. First columns: {column_list}. Provide a sharp 2-sentence business insight.

This allows the system to remain domain‑agnostic and work with any
dataset structure.

------------------------------------------------------------------------

## Requirements

Install dependencies:

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

## OpenAI API Setup (Required for GenAI Summary)

Set your API key in the same terminal session before running the script.

------------------------------------------------------------------------

## How to Run

``` bash
python Auto_EDA.py
```

Provide the CSV file path when prompted.

------------------------------------------------------------------------

## What the Script Outputs

1.  Dataset shape and columns
2.  Missing value count
3.  Top categorical distribution
4.  Descriptive statistics for numeric column
5.  Visual plots
6.  Optional GenAI summary

------------------------------------------------------------------------

## Purpose

This tool is meant for fast dataset understanding, not deep analytics.
It helps you quickly grasp structure, trends, and potential business
context before proceeding to advanced analysis.
