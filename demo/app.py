import os

from flask import Flask, jsonify, render_template, request
from flask_bootstrap import Bootstrap
from flask_wtf import FlaskForm
from wtforms import SubmitField, TextAreaField

from medspellchecker.tool.distilbert_candidate_ranker import RuDistilBertCandidateRanker
from medspellchecker.tool.medspellchecker import MedSpellchecker

static_folder_path = os.path.join(os.path.dirname(__file__), 'static')
app = Flask(__name__, static_folder=static_folder_path)

# Flask-WTF requires an enryption key - the string can be anything
app.config['SECRET_KEY'] = 'C2HWGVoMGfNTBsrYQg8EcMrdTimkZfAb'

Bootstrap(app)


class AnamnesisTextFrom(FlaskForm):
    anamnesis_text = TextAreaField(id="anamnesis_text", render_kw={'rows': 10})
    submit = SubmitField('Correct', id="anamnesis_submit",
                         render_kw={'class': 'btn btn-outline-primary',
                                    'rows': 10,
                                    'font-size': '18px'})


candidate_ranker = RuDistilBertCandidateRanker()
medspellchecker = MedSpellchecker(candidate_ranker)


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == "POST":
        corrected_text = medspellchecker.fix_text(request.form["anamnesis_text"])
        return jsonify(data={'message': corrected_text})

    form = AnamnesisTextFrom()
    return render_template('index.html', form=form)


@app.errorhandler(500)
def internal_server_error(e):
    return render_template('500.html'), 500


if __name__ == '__main__':
    app.run()
