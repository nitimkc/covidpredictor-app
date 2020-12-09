from flask import Flask, render_template, request
import pickle
import re
import numpy as np
from scipy import stats

# load the model
with open(f"model/best_model.pkl", 'rb') as f:
    model = pickle.load(f) 
with open(f"model/best_model_prob.pkl", 'rb') as f:
    prob = pickle.load(f) 
with open(f"model/best_model_score.pkl", 'rb') as f:
    score = pickle.load(f) 
# with open(f"model/column_means.pkl", 'rb') as f:
#     col_means = pickle.load(f) 
col_means = {'cough':None, 'fever':None, 'sore_throat':None, 'shortness_of_breath':None, 'head_ache':None, 'Sixties': 0.123189725, 'Gender': 0.49188364, 'contact': 0.63471705}

# instantiate Flask
# app = Flask(__name__, template_folder='templates')
app = Flask('covid_predictor', template_folder='templates')

# use Python decorators to decorate a function to map URL to a function
@app.route('/') 

# if user visits "/", then Flask will render "main.html" on the web browser
#                          Flask will look for the html file in templates folder
# "render_template" function renders the template and expects it to be stored 
# in the Templates folder on the same level as the "app.py" file
def show_predict_covid_form():
    return render_template('main.html')

# "/results" is now mapped to "results" function defined below (line 18 to 30)    
@app.route('/results', methods=['GET', 'POST'])

def results():
    form = request.form
    if request.method == 'GET':
        show_predict_covid_form()

    if request.method == 'POST':

        # gather input from web form using request.Form, which is a dictionary object
        cough = request.form['cough']
        fever = request.form['fever']
        sorethroat = request.form['sorethroat']
        shortnessofbreath = request.form['shortnessofbreath']
        headache = request.form['headache']
        sixtiesplus = request.form['sixtiesplus']
        gender = request.form['gender']
        contact = request.form['contact']
        
        # convert input elements into list and dictionary
        # input_vals = ['No','Yes','Yes','Yes','Yes','Above 60','Unknown','Yes',]
        input_vals = [cough, fever, sorethroat, shortnessofbreath, headache, sixtiesplus, gender, contact,]
        input_vars = [k for k in col_means]
        record = dict(zip(input_vars, input_vals))      

        # for pretty display
        display_vars = ['Cough', 'Fever', 'Sore throat', 'Shortness of breath', 'Headache',
                        'Age ', 'Gender', 'Contact with known carrier',]
        display_record = dict(zip(display_vars, input_vals))
        
        # process record to fit model requirement             
        maps = {'Yes':1.0, 'No':0.0, 'Male':1.0, 'Female':0.0, 'Above 60':1.0, 'Below 60':0.0}
        dummy_vars = [k for k,v in col_means.items() if v!=None]

        processed_record = dict(zip(record.keys(), [maps.get(j,j)  for i,j in record.items() if i in record.keys()])) 
        X = {}
        for i in processed_record: 
            if i in dummy_vars:
                if processed_record[i] in [0,1]:               # if contains values other than 0 and 1
                    X[i] = processed_record[i]
                    X[i+'_1'] = 0.0                # create new dummy column, 1=missing in original
                else:
                    X[i] = col_means[i]            # fill value other than 0 and 1 with mean
                    X[i+'_1'] = 1.0   
            else:
                X[i] = processed_record[i]
        X = list(X.values())
        
        if sum(X[:5])==0:
            predicted_covid_prob = False
            prob_percentile = False
            model_name = False
            model_score = False
            obs_posrate = False
            est_posrate = False
            policy_advice = False
        else:
            X = np.array(X).reshape(1,-1)
            
            # pass X to predict y
            y = model.predict_proba( X )[:,1]
            y_percentile = np.round( stats.percentileofscore(prob[:, 1], y),1 )
            
            predicted_covid_prob = '% '.join(map(str, np.append(np.round(y*100, 1), '') ))
            prob_percentile = str(y_percentile)
            model_name = re.sub(r"(\w)([A-Z])", r"\1 \2", score['name'])
            model_score = dict((k, score[k]) for k in ('sensitivity', 'specificity', 'accuracy', 'auc'))
            for k,v in model_score.items():
                model_score[k] = str(np.round(v*100,1))+'%'
            obs_posrate = str(np.round(score['positiverate']*100,1))+'%'
            est_posrate = (score['positiverate']-1+.99)/(.85-1+.99)
            est_posrate = str(np.round(est_posrate*100,1))+'%'
            policy_advice = np.where( y_percentile>(1-10000/30000), "Test", "Do Not Test")

        # pass input variables and "predicted_prob" to the "render_template" function
        # display the predicted value on the webpage by rendering the "resultsform.html" file
        return render_template('main.html', 
                                original_input=display_record,
                                prediction_prob=predicted_covid_prob,
                                prediction_prob_percentile=prob_percentile,
                                model_name=model_name,
                                model_score=model_score,
                                obs_posrate=obs_posrate,
                                est_posrate=est_posrate,
                                policy_advice=policy_advice)

# app.run() will start running the app on the host “localhost” on the port number 9999
# "debug": - during development the Flask server can reload the code without restarting the app
#          - also outputs useful debugging information
# visiting http://localhost:9999/ will render the "predictorform.html" page.
# if __name__ == "main":
app.run("localhost", "9999", debug=True)