import os
from datetime import timedelta

from flask import Flask, render_template, request, url_for

from models.SPADE.inference import inference


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'GenerativeAI_course_is_the_best'
    app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=365)

    @app.route('/')
    def stock():
        return render_template('main_page.html', title='Segment Understanding Knowledgeable Imaging',
                               text="SegmentUKI")

    @app.route("/generate", methods=['GET', 'POST'])
    def generation():
        generation_done = False
        p = None
        if request.method == 'POST':
            uploaded_file = request.files['file']
            if uploaded_file.filename != '':
                modes = ['min', 'med', 'max']
                epochs = {'ade': ['30', '150', '200'], 'facades': ['200', '400', '700']}
                dataset_type = request.form['Trained on dataset']
                res, p = inference(uploaded_file, uploaded_file.filename, model_type=request.form['Model type'],
                                   dataset_type=dataset_type,
                                   model_version=epochs[dataset_type][modes.index(request.form['Trained epochs'])])
                generation_done = True
                p = p.lstrip('api/static/')
                p = url_for('static', filename=p)
                return render_template("generation.html", result=p, generation_done=generation_done)

        return render_template("generation.html", result=p, generation_done=generation_done)

    @app.errorhandler(404)
    def not_found(error):
        info = 'Anonymous'
        er_txt = '404 not found: Wrong request: no such web-address!'
        return render_template('error.html', title='Error',
                               text=er_txt, useracc=info)

    @app.errorhandler(500)
    def error_serv(error):
        er_text = 'You are trying to break down the server. Don`t do that thing!'
        return render_template('error.html', title='Error', text=er_text)

    port = int(os.environ.get('PORT', 8080))
    app.run(host='127.0.0.1', port=port, debug=True)
    return app
