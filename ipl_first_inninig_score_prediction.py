#library 
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler


# detting into data and analysis
df =pd.read_csv('ipl.csv')
print(df.head())
print(df.shape)
print(df.info())
print(df.describe())
print(df.isnull().sum())

# making new variable for forcasting and splitting
df['year']=pd.to_datetime(df.date,format="%Y-%m-%d").dt.year
df.drop('date',axis=1,inplace=True)
df.head()


# encoding venue
le=LabelEncoder()
df.venue=le.fit_transform(df.venue)
encoded_venue=sorted(df.venue.unique())
array=le.inverse_transform(encoded_venue)
encoded_dict_venue={v:k for v,k in enumerate(array)}
print('encoded venue  \n {}'.format(encoded_dict_venue))

#dropping unneceddary column 
df.drop(['date','mid','batsman','bowler',],axis=1,inplace=True)
remove=df.loc[df['overs']<5].index
df.drop(remove,axis=0,inplace=True)

# refining consistent team 
df.bat_team.replace('Deccan Chargers','Sunrisers Hyderabad',inplace=True)
df['bat_team'].replace('Delhi Daredevils','Delhi Capitals',inplace=True)
df.bowl_team.replace('Deccan Chargers','Sunrisers Hyderabad',inplace=True)
df.bowl_team.replace('Delhi Daredevils','Delhi Capitals',inplace=True)

consistent_teams = ['Kolkata Knight Riders', 'Chennai Super Kings', 'Rajasthan Royals',
                    'Mumbai Indians', 'Kings XI Punjab', 'Royal Challengers Bangalore',
                    'Delhi Capitals', 'Sunrisers Hyderabad']
df = df[(df['bat_team'].isin(consistent_teams)) & (df['bowl_team'].isin(consistent_teams))]

#encoding batting team
bat_team_encoder=LabelEncoder()
df.bat_team=bat_team_encoder.fit_transform(df.bat_team)
encoded_bat=sorted(df.bat_team.unique())
bat=bat_team_encoder.inverse_transform(encoded_bat)
encoded_dict_bat={v:k for v,k in enumerate(bat)}
print('encoded botting team  \n {}'.format(encoded_dict_bat))

#encoding bowling team
bowl_team_encoder=LabelEncoder()
df.bowl_team=bowl_team_encoder.fit_transform(df.bowl_team)
encoded_bowl=sorted(df.bowl_team.unique())
bowl=bowl_team_encoder.inverse_transform(encoded_bowl)
encoded_dict_bowl={v:k for v,k in enumerate(bowl)}
print('encoded bowling team  \n {}'.format(encoded_dict_bowl))

# splitting the data for predicting new season score
x=df.drop('total',axis=1)
x_train=x[x['year']!=2017]
x_train.head()
x_train=x_train.drop('year',axis=1)
x_test=x[x['year']==2017]
x_test=x_test.drop('year',axis=1)
x=df.drop(['year','total'],axis=1)
y=df.iloc[:,10:]
y_train=y[y['year']!=2017]
y_train=y_train.drop('year',axis=1)
y_test=y[y['year']==2017]
y_test=y_test.drop('year',axis=1)
y=y.drop('year',axis=1)

x_col=x.columns
#scaling the input variable for better performance 
scaler = StandardScaler()
scaler.fit(x)
x=scaler.transform(x)
x_train=scaler.transform(x_train)
x_test=scaler.transform(x_test)
x=pd.DataFrame(x,columns=x_col)
x_train=pd.DataFrame(x_train,columns=x_col)
x_test=pd.DataFrame(x_test,columns=x_col)

# gradient boostiiiiiiing
regressor=GradientBoostingRegressor(max_depth=4,learning_rate=0.1,min_samples_leaf=103,n_estimators=200,)
regressor.fit(x_train,y_train)
regressor.score(x_test,y_test)

# saving the model
filename = 'first-innings-score-gradient-boost.pkl'
pickle.dump(regressor, open(filename, 'wb'))
