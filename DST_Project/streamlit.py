import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np

file = st.file_uploader("Upload file", type=["csv"])

if file is not None:
    df = pd.read_csv(file)

    n_row = st.slider(
        "Choose number of rows",
        min_value=5,
        max_value=len(df),
        step=1
    )

    column_view = st.multiselect(
        "Select columns to show",
        df.columns.to_list(),
        default=df.columns.to_list()
    )

    numerial_col = df.select_dtypes(include=np.number).columns.to_list()

    st.write(df.loc[:n_row, column_view])

    x_column = st.selectbox("Select column on x-axis:", numerial_col)

    # Histogram
    fig = px.histogram(
        df,
        x=x_column,
        nbins=30,
        title='Distribution of Average Salaries for Software Engineer Position'
    )
    st.plotly_chart(fig)

    # Salary by Experience Level
    filtered_df = df[df['experience_level'] != 'Not Specified']
    figy = px.box(
        df,
        x='experience_level',
        y=x_column,
        category_orders={'experience_level': ['Entry', 'Mid', 'Senior']},
        title='Salary Distribution by Experience Level'
    )
    st.plotly_chart(figy)

    # Top Hiring Companies
    top_companies = (
        df['Company']
        .value_counts()
        .head(10)
        .reset_index()
    )
    top_companies.columns = ['Company', 'Job Postings']
    fig = px.bar(
        top_companies,
        x='Company',
        y='Job Postings',
        title='Top 10 Companies Hiring Software Engineers',
        color_discrete_sequence=['#FF5733']
    )
    st.plotly_chart(fig)

    # Top Paying Locations on Average
    geo_df = (
        df[df['clean_location'] != 'Remote']
        .groupby('clean_location')['salary_avg']
        .mean()
        .reset_index()
    )
    geo_df = geo_df.sort_values('salary_avg', ascending=False).head(20)
    fig = px.bar(
        geo_df,
        x='salary_avg',
        y='clean_location',
        orientation='h',
        title='Top 20 Locations by Average Salary',
        color='salary_avg',
        color_continuous_scale='viridis'
    )
    fig.update_layout(
        yaxis={'categoryorder': 'total ascending'},
        xaxis_title='Average Salary ($)',
        yaxis_title='Location'
    )
    st.plotly_chart(fig)

    # Remote Jobs vs On-Site
    remote_counts = df['Is_Remote'].value_counts()
    labels = ['On-site' if x == 0 else 'Remote' for x in remote_counts.index]
    fig = px.pie(
        values=remote_counts,
        names=labels,
        title='Distribution of Remote vs On-site Positions',
        color=labels,
        color_discrete_map={
            'On-site': '#FF7F50',
            'Remote': '#87CEFA'
        }
    )
    st.plotly_chart(fig)
