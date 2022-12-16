from flask import Flask, url_for, render_template, redirect, session, Response, send_from_directory
from ensembles import RandomForestMSE, GradientBoostingMSE
import pandas as pd
import sys
import io
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from flask_forms import FittingForm, PredictForm


app = Flask(__name__)
app.config['SECRET_KEY'] = 'hello'

menu = [{'name': 'Главная', 'url': 'index'},
        {'name': 'Случайный лес', 'url': 'get_rf'},
        {'name': 'Градиентный бустинг', 'url': 'get_gb'},
        {'name': 'О себе', 'url': 'about_me'}]
datasets = {}

@app.route('/')
@app.route('/index')
def get_index():
    return render_template('index.html', menu=menu)

@app.route('/plot')
def plot_png():
    acc_train = session['acc_train']
    acc_val = session['acc_val']

    fig, ax = plt.subplots(figsize=(6, 4), dpi=500)
    ax.set_title('RMSE от числа итераций')
    ax.set_xlabel('Число итераций')
    ax.set_ylabel('RMSE')
    ax.plot(acc_train, label='train')
    if acc_val is not None:
        ax.plot(acc_val, label='val')
    ax.legend()
    ax.grid()
    fig.tight_layout()
    output = io.BytesIO()
    FigureCanvasAgg(fig).print_png(output)

    return Response(output.getvalue(), mimetype='img/png')

@app.route('/get_rf', methods=['GET', 'POST'])
def get_rf():
    fitting_form = FittingForm()
    if fitting_form.validate_on_submit():
        train_data = pd.read_csv(fitting_form.train_data.data)
        val_data = pd.read_csv(fitting_form.val_data.data)
        target_name = fitting_form.target_name.data
        datasets['train_data'] = train_data
        datasets['val_data'] = val_data
        datasets['target_name'] = target_name

        n_estimators = fitting_form.n_estimators.data
        max_depth = fitting_form.max_depth.data
        feature_subsampling = fitting_form.feature_subsample_size.data

        session['model_type'] = 'Случайный лес'
        session['model_params'] = {'n_estimators': n_estimators, 'max_depth': max_depth, 'feature_subsampling': feature_subsampling}

        return redirect(url_for('get_predict'))


    return render_template('fitting.html', form=fitting_form, gb=False, title='Случайный лес', menu=menu)

@app.route('/get_gb', methods=['GET', 'POST'])
def get_gb():
    fitting_form = FittingForm()
    if fitting_form.validate_on_submit():
        train_data = pd.read_csv(fitting_form.train_data.data)
        val_data = pd.read_csv(fitting_form.val_data.data)
        target_name = fitting_form.target_name.data
        datasets['train_data'] = train_data
        datasets['val_data'] = val_data
        datasets['target_name'] = target_name

        n_estimators = fitting_form.n_estimators.data
        max_depth = fitting_form.max_depth.data
        feature_subsampling = fitting_form.feature_subsample_size.data
        learning_rate = fitting_form.learning_rate.data

        session['model_type'] = 'Градиентный бустинг'
        session['model_params'] = {'n_estimators': n_estimators, 'max_depth': max_depth, 'feature_subsampling': feature_subsampling, 'learning_rate': learning_rate}

        return redirect(url_for('get_predict'))

    return render_template('fitting.html', form=fitting_form, gb=True, title='Градиентный бустинг', menu=menu)

@app.route('/about_me')
def get_about():
    return render_template('about_me.html', menu=menu, title='Тебе крышка')

@app.route('/prediction', methods=['GET', 'POST'])
def get_predict():
    train_data = datasets['train_data']
    val_data = datasets['val_data']
    target_name = datasets['target_name']

    X_train = train_data.drop([target_name], axis=1).values
    y_train = train_data[target_name].values
    if val_data is not None:
        X_val = val_data.drop([target_name], axis=1).values
        y_val = val_data[target_name].values
    else:
        X_val = None
        y_val = None

    n_estimators = session['model_params']['n_estimators']
    max_depth = session['model_params']['max_depth']
    feature_subsampling = session['model_params']['feature_subsampling']
    if 'learning_rate' in session:
        learning_rate = session['model_params']['learning_rate']
    if session['model_type'] == 'Градиентынй бустинг':
        model = GradientBoostingMSE(n_estimators=n_estimators, max_depth=max_depth,
                                    feature_subsample_size=feature_subsampling,
                                    learning_rate=learning_rate)
    else:
        model = RandomForestMSE(n_estimators=n_estimators, max_depth=max_depth,
                                feature_subsample_size=feature_subsampling)

    if X_val is not None:
        acc_train, acc_val, time_md = model.fit(X_train, y_train, X_val, y_val, trace=True)
    else:
        acc_train, time_md = model.fit(X_train, y_train, trace=True)
        acc_val = None
    predict_form = PredictForm()
    session['acc_train'] = acc_train
    session['acc_val'] = acc_val

    if predict_form.validate_on_submit():
        data_test = pd.read_csv(predict_form.test_data.data)
        y_pred = pd.DataFrame({target_name: model.predict(data_test.values)})
        file_name = predict_form.test_data.name + '_predict.csv'
        path = os.path.join(os.getcwd(), 'tmp/')
        if not os.path.exists(path):
            os.mkdir(path)
        y_pred.to_csv(os.path.join(path, file_name))
        return send_from_directory(path, file_name, as_attachment=True)


    return render_template('prediction.html', acc_train=acc_train,
    acc_val=acc_val, menu=menu, title=session['model_type'], form=predict_form)