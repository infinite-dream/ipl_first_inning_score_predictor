
from flask import Flask,render_template,request
import pickle
import numpy as np
from ipl_first_inninig_score_prediction import le,bowl_team_encoder,bat_team_encoder,scaler

filename='first-innings-score-KNeighbour.pkl'
regressor=pickle.load(open(filename,'rb'))


app=Flask(__name__)
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    temp=list()
    if request.methods=="POST":
        venue=request.form['venue']
        venue=le.transform(venue)
        temp=temp+venue
         
        batting_team=request.form['batting-team']
        team=bat_team_encoder.transform(batting_team)
        temp=temp+team
        
        bowling_team=request.form['bowling-team']
        team=bowl_team_encoder.transfrom(bowling_team)
        temp=temp+team
       
        overs=float(request.form['overs'])
        run=int(request.form['runs'])
        wickets=int(request.form['wickets'])
        runs_in_prev5=int(request.form['runs_in_prev_5'])
        wickets_in_prev5=int(request.form['wickets_in_prev_5'])
        
        temp=temp+overs+run+wickets+runs_in_prev5+wickets_in_prev5
        temp=scaler.transform(temp)
        data=np.array([temp])
        prediction=int(regressor,predict(data)[0])
        
        return render_template("result.html",lower_limit=prediction-15 ,upper_limit=prediction+15)
    2
    if __name__=="__main__":
        app.run(debug=True)
        
        
        
        
        
        
    