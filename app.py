from flask import Flask, render_template, request
import pickle
import re
import pandas as pd
import numpy as np
from scipy import stats

# load the reqd info
with open(f"model/best_model.pkl", 'rb') as f:
    model = pickle.load(f) 
with open(f"model/X_test.pkl", 'rb') as f:
    test_data = pickle.load(f) 
with open(f"model/best_model_score.pkl", 'rb') as f:
    score = pickle.load(f) 
with open(f"model/column_means.pkl", 'rb') as f:
    col_means = pickle.load(f) 

# instantiate Flask
app = Flask(__name__, template_folder='templates')
# app = Flask('covid_predictor', template_folder='templates')

# use Python decorators to decorate a function to map URL to a function
@app.route('/') 

# "render_template" function renders the template and expects it to be stored 
# in the Templates folder on the same level as the "app.py" file
def show_predict_covid_form():
    return render_template('main.html')

# "/results" is now mapped to "results" function defined below
@app.route('/results', methods=['GET', 'POST'])

def results():
    form = request.form
    if request.method == 'GET':
        show_predict_covid_form()

    if request.method == 'POST':
        
        # gather input from web form using request.Form
        rates = [request.form['tpr'], request.form['tnr']]
        test_capacity, n_patient = int(request.form['test_capacity']), int(request.form['n_patient'])
        input_vals = [request.form['cough'], request.form['fever'], request.form['sorethroat'], request.form['shortnessofbreath'], 
                request.form['headache'], request.form['sixtiesplus'], request.form['gender'], float(request.form['apt7'])/100,]

        # rates = ['85.0', '90.0']
        # test_capacity , n_patient = 10000, 30000
        # input_vals = ['No','Yes','Yes','Yes','Yes','Above 60','Unknown',.25,]
        # input_vals = ['Yes','No','No','No','No','Below 60','Female',.25,]

        rates  = [ float(i) if float(i) < 1 else float(i)/100 for i in rates]
        tpr , tnr = rates[0], rates [1] 
        n_patient  = test_capacity if test_capacity > n_patient  else n_patient

        input_vars = [k for k in col_means]
        record = dict(zip(input_vars, input_vals))      

        # for pretty display
        display_vars = ['Cough', 'Fever', 'Sore throat', 'Shortness of breath', 'Headache',
                        'Age ', 'Gender', 'Average positive test % in the last 7 days',]
        display_record = dict(zip(display_vars, input_vals))
        display_test_info = dict(zip(['Test sensitivity', 'Test specificity', 'Estimated testing capacity', 'Estimated no. of patients'],
                                     [str(np.round(tpr*100,1))+'%', str(np.round(tnr*100,1))+'%', test_capacity, n_patient]))
        
        # process record to fit model requirement             
        maps = {'No':0.0, 'Yes':1.0, 'Male':0.0, 'Female':1.0, 'Below 60':0.0, 'Above 60':1.0 }
        dummy_vars = [k for k,v in col_means.items() if v!=None]
        dummy_vars_missing = [i+'_1' for i in dummy_vars]

        processed_record = dict(zip(record.keys(), [maps.get(j,j)  for i,j in record.items() if i in record.keys()])) 
        for i in processed_record:
            if processed_record[i] =='Unknown':
                processed_record[i] = col_means[i] 
        
        # add_dummy columns
        for i,j in zip(dummy_vars,dummy_vars_missing):
            print(i,j)
            if processed_record[i] not in [0,1]:
                processed_record.update({j:1.0}) # replace dummy col with value 1
            else:
                processed_record.update({j:0.0}) # replace dummy col with value 0

        X = list(processed_record.values())
        X_test = pd.DataFrame(test_data, columns=[k for k,v in processed_record.items()])

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
            print(processed_record)
            print(X)
            cond1 = X_test['Ave_Pos_Past7d']>=(float(input_vals[-1])-1)
            cond2 = X_test['Ave_Pos_Past7d']<=(float(input_vals[-1])+1)
            X_test = X_test[(cond1)&(cond2)]

            # pass X to predict y
            y = model.predict_proba( X )[:,1]
            y_prob = model.predict_proba(X_test)[:,1]
            y_percentile = np.round( stats.percentileofscore(y_prob, y),1 )
            
            predicted_covid_prob = '% '.join(map(str, np.append(np.round(y*100, 1), '') ))
            prob_percentile = str(y_percentile)
            model_name = re.sub(r"(\w)([A-Z])", r"\1 \2", score['name'])
            model_score = dict((k, score[k]) for k in ('sensitivity', 'specificity', 'accuracy', 'AUC'))
            for k,v in model_score.items():
                model_score[k] = str(np.round(v*100,1))+'%'
            
            obs_posrate = str(np.round(score['positiverate']*100,1))+'%'
            est_posrate = (score['positiverate']-1+tnr)/(tpr-1+tnr)
            est_posrate = str(np.round(est_posrate*100,1))+'%'
            policy_advice = np.where( y_percentile/100>1-(test_capacity/n_patient), "Test", "Do Not Test")

        # pass input variables and "predicted_prob" to the "render_template" function
        # display the predicted value on the webpage by rendering the "resultsform.html" file
        return render_template('main.html', 
                                original_patient_input=display_record,
                                original_test_input=display_test_info,
                                prediction_prob=predicted_covid_prob,
                                prediction_prob_percentile=prob_percentile,
                                model_name=model_name,
                                model_score=model_score,
                                obs_posrate=obs_posrate,
                                est_posrate=est_posrate,
                                policy_advice=policy_advice,
                                processed_record=processed_record)

# app.run() will start running the app on the host “localhost” on the port number 9999
# "debug": - during development the Flask server can reload the code without restarting the app
#          - also outputs useful debugging information
# visiting http://localhost:9999/ will render the "predictorform.html" page.
if __name__ == "main":
    app.run("localhost", "9999", debug=True)