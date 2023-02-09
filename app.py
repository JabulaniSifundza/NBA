import streamlit as st
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import lxml

st.title("NBA Statistics")

st.subheader(
    "This app web srapes NBA statistics to visualize them starting from 1950 to 2022"
)
st.write("Data source: https://www.basketball-reference.com/")

st.sidebar.header('Select a year, team and positions')
# Creating dropdown in streamlit
selected_year = st.sidebar.selectbox("Year", list(reversed(range(1950, 2022))))


# web scraping NBA player stats and data pre-processing
@st.cache
def load_data(year):
    url = f"https://www.basketball-reference.com/leagues/NBA_{str(year)}_per_game.html"
    html = pd.read_html(url, header=0)
    df = html[0]
    raw = df.drop(df[df.Age == "Age"].index) # deletes repeating info
    raw = raw.fillna(0)
    return raw.drop(['Rk'], axis=1)

playerstats = load_data(selected_year)

# sidebar team selector
sorted_unique_team = sorted(playerstats.Tm.unique())
selected_team = st.sidebar.multiselect('Team', sorted_unique_team, sorted_unique_team)

# sidebar position selector
unique_pos = ['C', 'PF', 'PG', 'SG', 'SF']
selected_pos = st.sidebar.multiselect('Position', unique_pos, unique_pos)

# Filtering data based on input selection
# Selecting data in Pandas
# We use the name of the dataframe (playerstats) and inside the List we enter our condition to filter data
df_selected_team = playerstats[(playerstats.Tm.isin(selected_team)) & (playerstats.Pos.isin(selected_pos))]


st.header('Display Player Stats of the Selected Team(s) and Positions')
st.write(f"Data Dimensions: {str(df_selected_team.shape[0])} rows and {str(df_selected_team.shape[1])} columns")
st.dataframe(df_selected_team)

# Downloading NBA player stats Data
def csvfiledownload(df):
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    return f"<a href='data:file/csv;base64,{b64}' download='playerstats.csv'>Download CSV File</a>"

st.markdown(csvfiledownload(df_selected_team), unsafe_allow_html=True)

# Making a heatmap
if st.button('Intercorrelation Heatmap'):
    st.header('Intercorrelation Matrix Heatmap')
    df_selected_team.to_csv('output.csv', index=False)
    df = pd.read_csv('output.csv')
    
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("white"):
        f, ax = plt.subplots(figsize=(7,5))
        ax = sns.heatmap(corr, mask=mask, vmax=1, square=True)
    st.pyplot(f)

st.header('Visualize Individual Player Statistics')
    
fullname = st.text_input("Enter NBA Player's Fullname 👇🏾", placeholder="Player Fullname")

def getplayerdata(fullname):
    if len(fullname) < 1:
        st.write("Please enter a Player's fullname to start")
    else:
        firstchar = fullname.split()
    firstname, lastname = firstchar

    lastname_initial = lastname[0].lower()

    five_char = lastname[:5].lower()
    two_char = firstname[:2].lower()

    # print(f"{five_char}{two_char}")
    # print(two_char)
    # print(five_char)
    # print(lastname_char)
    url = f"https://www.basketball-reference.com/players/{lastname_initial}/{five_char}{two_char}01.html"
    html = pd.read_html(url, header = 0)
    df = html[0]
    cleanup = df.drop(df[df.Age == "Age"].index)
    cleanup = cleanup.fillna(0)
    cleanup = cleanup.drop(['Lg'], axis=1)
    cleanup = cleanup.drop(cleanup.loc[cleanup['Season'] == 0].index)
    # cleanup['Age'] = cleanup['Age'].apply(lambda x: int(x))
    # cleanup['MP'] = cleanup['MP'].apply(lambda x: round(float(x)))
    # cleanup['FG'] = cleanup['FG'].apply(lambda x: round(float(x), 1))
    return cleanup
try:
    individual_stats = getplayerdata(fullname)
except UnboundLocalError:
    st.write("Please enter a Player's fullname to start")
    

st.dataframe(individual_stats)
seasons = individual_stats['Season']
# Assists, Points and Offensive Rebounds
labels = individual_stats['Season'].tolist()
# [_ for _ in individual_stats['Season']]
df_assists = individual_stats['AST'].tolist()
df_points = individual_stats['PTS'].tolist()
# df_offensive_stats = [df_assists, df_points, df_off_rebounds]
lab = np.arange(len(labels))
width = 0.4
fi, ax1 = plt.subplots(figsize=(22,10))
rects1 = ax1.bar(lab - width / 2, df_assists, width, label = 'Assists', align='center')
rects2 = ax1.bar(lab + width / 2, df_points, width, label = 'Points', align='center')

ax1.set_ylabel('Season Average')
ax1.set_title('Offensive Production')
ax1.set_xticks(lab, labels)
ax1.legend()
ax1.bar_label(rects1, padding=8)
ax1.bar_label(rects2, padding=8)
# fi.tight_layout()
st.subheader(f"{fullname} Offensive Statistics")
st.pyplot(fi)

# Defensive production
df_blocks = individual_stats['BLK'].tolist()
df_steals = individual_stats['STL'].tolist()
lab_size = np.arange(len(labels))
fig2, ax2 = plt.subplots(figsize=(22,10))
rects3 = ax2.bar(lab_size - width / 2, df_blocks, width, label = 'Blocks', align = 'center')
rects4 = ax2.bar(lab_size + width / 2, df_steals, width, label = 'Steals', align = 'center')
ax2.set_ylabel('Season Average')
ax2.set_title('Defensive Production')
ax2.set_xticks(lab_size, labels)
ax2.legend()
ax2.bar_label(rects3, padding=8)
ax2.bar_label(rects4, padding=8)
# fi.tight_layout()
st.subheader(f"{fullname} Defensive Statistics")
st.pyplot(fig2)

#Rebounding
df_ORB = individual_stats['ORB'].tolist()
df_DRB = individual_stats['DRB'].tolist()
lab2_size = np.arange(len(labels))
fig3, ax3 = plt.subplots(figsize=(22,10))
rects5 = ax3.bar(lab2_size - width / 2, df_ORB, width, label = 'Offensive Rebounding', align = 'center')
rects6 = ax3.bar(lab2_size + width / 2, df_DRB, width, label = 'Defensive Rebounding', align = 'center')
ax3.set_ylabel('Season Average')
ax3.set_title('Rebounding')
ax3.set_xticks(lab2_size, labels)
ax3.legend()
ax3.bar_label(rects5, padding=8)
ax3.bar_label(rects6, padding=8)
# fi.tight_layout()
st.subheader(f"{fullname} Rebounding")
st.pyplot(fig3)




selected_season = st.selectbox("Select Season", seasons)
df_season_stats = individual_stats.loc[individual_stats['Season'] == selected_season]
# Portion of 2 Pointers and 3 Pointers
# Showing Season stats
df_field_goals = df_season_stats.FG
df_3Pointer = df_season_stats['3P'].values[0]
df_2Pointer = df_season_stats['2P'].values[0]
# df_Free_Throws = df_season_stats['FT'].values[0]
df_shots = [df_2Pointer, df_3Pointer]
# st.write(df_field_goals, df_2Pointer, df_3Pointer)
explode = (0, 0.1)
colors = ['#17408B', '#C9082A']
fig, ax = plt.subplots()
ax.set_title(f"Types of Shots Taken During {selected_season} Season")
ax = plt.pie(df_shots, labels=["2 Pointers", "3 Pointers"], explode=explode, autopct='%1.1f%%', colors=colors)
st.pyplot(fig)
