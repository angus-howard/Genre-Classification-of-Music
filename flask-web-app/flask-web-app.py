
import os
import pickle
import librosa_script as lib
import numpy as np
from flask import Flask, render_template, request
from flask_dropzone import Dropzone
import plotly
import plotly.graph_objs as go
import json

basedir = os.path.abspath(os.path.dirname(__file__))


app = Flask(__name__)
#app._static_folder = os.path.join(basedir, '/templates/static')
app.config.update(
    UPLOADED_PATH=os.path.join(basedir, 'uploads'),
    # Flask-Dropzone config:
    DROPZONE_ALLOWED_FILE_TYPE='audio',
    DROPZONE_MAX_FILE_SIZE=30,
    DROPZONE_MAX_FILES=1,
    DROPZONE_REDIRECT_VIEW='completed'  # set redirect view
)

dropzone = Dropzone(app)

with open('/home/ahoward/mysite/flask-web-app/red_svc_model.pkl', 'rb') as mdl:
    model = pickle.load(mdl)

conversion_dict = {  0:'ELECTRO',
                     1:'ROCK',
                     2:'POP',
                     3:'FOLK',
                     4:'HIP-HOP',
                     5:'JAZZ',
                     6:'CLASSICAL',
                     7:'BLUES',
                     8:'SOUL-RNB',
                     9:'COUNTRY'
                    }

@app.route('/', methods=['POST', 'GET'])
def upload():
    global filename
    if request.method == 'POST':
        f = request.files.get('file')
        filename = f.filename
        f.save(os.path.join(app.config['UPLOADED_PATH'], f.filename))
    return render_template('index.html')

@app.route('/about', methods=['GET'])
def about():
    return render_template('generic.html')

@app.route('/uploaded')
def completed():
    return render_template('uploaded.html')

@app.route('/predict', methods=['GET'])
def predict():
    global filename
    features = lib.compute_features('/home/ahoward/mysite/flask-web-app/uploads/{}'.format(filename)).values.reshape(1,-1)
    os.remove('/home/ahoward/mysite/flask-web-app/uploads/{}'.format(filename))
    # @after_this_request
    # def remove_file():
    #     try:
    #         os.remove('./uploads/{}'.format(filename))
    #     except:
    #         pass
    pred = conversion_dict[model.predict(features)[0]]
    probas = dict(zip(conversion_dict.values(),model.predict_proba(features)[0]))
    barh = create_bar(probas)
    return render_template('predict.html', pred = pred, plot = barh)

def create_bar(proba_dict):
    bar = [go.Bar(
                    x = list(proba_dict.keys()),
                    y = np.around(list(proba_dict.values()),4),
                    opacity = 0.7,
                    marker = dict(
                                color = 'rgb(228,85,79)'
                                )
                )]

    layout = go.Layout(
                        margin = dict(
                                    t = 0,
                                    l = 0,
                                    r = 100,
                                    ),
                        paper_bgcolor ='rgba(0,0,0,0)',
                        plot_bgcolor = 'rgba(0,0,0,0)',
                        height = 450,
                        width = 750,
                        xaxis=dict(
                                    showticklabels = True,
                                    showgrid =False,
                                    tickfont = dict(
                                                    color = "rgba(255, 255, 255, 0.7)",
                                                    family = 'Helvetica',
                                                    size = 12
                                                    ),
                                    ticklen = 8
                                    ),
                        yaxis=dict(
                                    showticklabels = False,
                                    showgrid = False
                                    )
                        )

    fig = go.Figure(data=bar, layout=layout)

    graphJSON = json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)

    return graphJSON

if __name__ == '__main__':
    app.run(debug=True)
