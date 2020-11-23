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

# instantiate Flask
app = Flask(__name__, template_folder='templates')
# app = Flask('covid_predictor', template_folder='templates')

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
        abroad = request.form['abroad']
        
        # convert input elements into list and dictionary
        # input_vars = ['No','Yes','Yes','Yes','Yes','Above 60','Male','Yes','No',]
        input_vars = [cough, fever, sorethroat, shortnessofbreath, headache, sixtiesplus, gender, contact, abroad,]
        reg_vars = ['cough', 'fever', 'sorethroat', 'shortnessofbreath', 'headache']
        dummy_vars = ['sixtiesplus', 'gender', 'contact', 'abroad',]
        record = dict(zip(reg_vars+dummy_vars, input_vars))
                            
        display_vars = ['Cough', 'Fever', 'Sore throat', 'Shortness of breath', 'Headache',
                        'Age ', 'Gender', 'Contact with known carrier', 'Recent travel abroad',]
        display_record = dict(zip(display_vars, input_vars))
        
        map_reg_vals = {'Yes':1, 'No':0}
        map_dummy_vals = {'Yes':[1,0], 'No':[0,0],'Unknown':[0,1],
                            'Below 60':[0,0], 'Above 60':[1,0], 
                            'Male':[0,0], 'Female':[1,0],}

        reg_X = [map_reg_vals.get(j,j)  for i,j in record.items() if i in reg_vars]
        dummy_X = [map_dummy_vals.get(j,j)  for i,j in record.items() if i in dummy_vars]
        X = reg_X + dummy_X

        if sum(X[:5])==0:
            predicted_covid_prob = False
            prob_percentile = False
            model_name = False
            model_score = False
        else:
            X_all = []
            for i in X:
                if type(i)==list:
                    X_all.extend(i)
                else:
                    X_all.append(i)

            # prepare X for sklearn model
            X_int = np.array(X_all)
            if len(X_int.shape) == 1:
                X_int = X_int.reshape(1,-1)
            
            # pass X to predict y
            y = model.predict_proba( X_int )[:,1]
            y_percentile = np.round( stats.percentileofscore(prob[:, 1], y),1 )
            
            predicted_covid_prob = '% '.join(map(str, np.append(np.round(y*100, 1), '') ))
            prob_percentile = str(y_percentile)
            model_name = re.sub(r"(\w)([A-Z])", r"\1 \2", score['name'])
            model_score = score #'% '.join(map(str, np.append(np.round(score['auc']*100, 2), '') ))
            model_score = dict((k, score[k]) for k in ('sensitivity', 'specificity', 'accuracy', 'auc'))
            for k,v in model_score.items():
                model_score[k] = '% '.join(map(str, np.append(np.round(v*100, 1), '') ))

        # pass input variables and "predicted_prob" to the "render_template" function
        # display the predicted value on the webpage by rendering the "resultsform.html" file
        return render_template('main.html', 
                                original_input=display_record,
                                prediction_prob=predicted_covid_prob,
                                prediction_prob_percentile=prob_percentile,
                                model_name=model_name,
                                model_score=model_score)

# app.run() will start running the app on the host “localhost” on the port number 9999
# "debug": - during development the Flask server can reload the code without restarting the app
#          - also outputs useful debugging information
# visiting http://localhost:9999/ will render the "predictorform.html" page.
if __name__ == "main":
    app.run("localhost", "9999", debug=True)