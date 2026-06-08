import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def tab3(df):
    st.header("Graphs")
    st.title("Crime Data Graph")

    graph_df = df.copy()
    
    # Split data in years
    graph_df["Year"] = graph_df["Month"].astype(str).str[:4]

    months = {
        "01" : "Jan",
        "02" : "Feb",
        "03" : "Mar",
        "04" : "Apr",
        "05" : "May",
        "06" : "June",
        "07" : "July",
        "08" : "Aug",
        "09" : "Sep",
        "10" : "Oct",
        "11" : "Nov",
        "12" : "Dec"
    }

    # Split data in months
    graph_df["Month"] = graph_df["Month"].astype(str).str[5:].replace(months)

    # Use the name of the months instead of numbers with dictionary
    graph_df["Month"] = pd.Categorical(
        graph_df["Month"], 
        categories=list(months.values()), 
        ordered=True
    )

    # Add columns instead of sidebars to prevent sidebar being visible in every tab.
    col1, col2 = st.columns(2)

    # Selectbox for years
    with col1:
        years = sorted(graph_df["Year"].unique())
        chosen_year = st.selectbox("Choose Year:", options=["All"] + years)

    #Multiselect box for crimes
    with col2:
        crimes = sorted(graph_df['Crime type'].unique())
        selected_crimes = st.multiselect(
            "Select Crime Type(s):", 
            options=crimes, 
            default=[crimes[0]] 
        )
    
    if selected_crimes:
        graph_df = graph_df[graph_df['Crime type'].isin(selected_crimes)]


    if not graph_df.empty:
        
        #If no year is chosen, show all years crime count
        if chosen_year == "All":
            grouped_df = graph_df.groupby(['Year', 'Crime type'])['Crime_Count'].sum().reset_index()
            x_column = "Year"
            x_label = "Year"
            chart_title = "Crime Count by Year"
        
        #If a specfic year is chosen, show monthly crime counts 
        else:
            graph_df = graph_df[graph_df['Year'] == chosen_year]  
            grouped_df = graph_df.groupby(['Month', 'Crime type'])['Crime_Count'].sum().reset_index()
            x_column = "Month"
            x_label = "Month"
            chart_title = f"Crime Count by Month in {chosen_year}"

        # Create figure for plots
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Add seaborn bar plot
        sns.barplot(
            data=grouped_df, 
            x=x_column,      # choose month or year according to chosen year 
            y="Crime_Count", 
            hue="Crime type", 
            ax=ax,
            palette="viridis" 
        )
        # Add labels and titles to the graph
        ax.set_title(chart_title, fontsize=16, pad=15)
        ax.set_xlabel(x_label, fontsize=12)
        ax.set_ylabel("Total Crimes", fontsize=12)
        
        # Add legend and move to upper left
        plt.legend(title="Crime Type", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        
        # Show the plot in dashboard
        st.pyplot(fig)