from flask import Flask, url_for, render_template, redirect, session, Response, send_from_directory
from ensembles import RandomForestMSE, GradientBoostingMSE
import pandas as pd
import sys, io, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib.backends.backend_agg import FigureCanvasAgg
import matplotlib.pyplot as plt

from flask_forms import FittingForm, PredictForm

app = Flask(__name__)
app.config['SECRET_KEY'] = 'hello'

menu = [{'name': 'Главная', 'url': '/index'},
        {'name': 'Случайный лес', 'url': '/get_rf'},
        {'name': 'Градиентный бустинг', 'url': '/get_gb'},
        {'name': 'О себе', 'url': '/about_me'}]
errors = []
datasets = {}
Session = {}

@app.route('/')
@app.route('/index')
def get_index():
    return render_template('index.html', menu=menu)

@app.route('/plot')
def plot_png():
    try:
        acc_train = Session['acc_train']
        acc_val = Session['acc_val']

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
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        output = io.BytesIO()
        return Response(output.getvalue(), mimetype='img/png')

def check_train_val(train_data, val_data, target_name):
    cur_errors = []

    try:
        cols_train = train_data.select_dtypes(include=np.number).columns.tolist()
        if target_name in train_data.columns.values:
            if len(cols_train) == train_data.shape[1]:
                if val_data is not None:
                    cols_val = val_data.select_dtypes(include=np.number).columns.tolist()
                    app.logger.info(list(train_data.columns.values))
                    if list(train_data.columns.values) == list(val_data.columns.values):
                        if cols_train == cols_val:
                            pass
                        else:
                            cur_errors.append('Типы признаков у обучающей и валидационной выборки должны совпадать!')
                    else:
                        cur_errors.append('Множество признаков для обучения и валидации должны совпадать!')
            else:
                cur_errors.append('Данная реализация умеет рабтать только с числовыми признаками. В обучающей выборке есть признаки, отличные от числовых!')
        else:
            cur_errors.append('Такой целевой переменной нет в обучающей выборке!')

        return cur_errors
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        cur_errors.append('Извините, ошибка на стороне сервера. Попробуйте другие данные.')
        return cur_errors

def check_train_test(train_data, test_data, target_name):
    cur_errors = []

    try:
        cols_train_names = train_data.drop(columns=[target_name]).columns.tolist()
        cols_train_types = train_data.drop(columns=[target_name]).select_dtypes(include=np.number).columns.tolist()

        cols_test_names = test_data.columns.tolist()
        cols_test_types = test_data.select_dtypes(include=np.number).columns.tolist()

        if target_name not in cols_test_names:
            if cols_train_names == cols_test_names:
                if cols_train_types == cols_test_types:
                    pass
                else:
                    cur_errors.append('Типы признаков у обучающей и тестовой выборки должны совпадать!')
            else:
                cur_errors.append('Тестовые данные не подходят по формату к тренировочным. Проверьте совпадение имен колонок.')
        else:
            cur_errors.append('Целевая переменная не должна присутсвовать в тестовой выборке')
        return cur_errors
    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        cur_errors.append('Извините, ошибка на стороне сервера. Попробуйте другие данные.')
        return cur_errors

@app.route('/get_rf', methods=['GET', 'POST'])
def get_rf():
    try:
        global errors
        fitting_form = FittingForm()
        if fitting_form.validate_on_submit():
            train_data = pd.read_csv(fitting_form.train_data.data)
            if fitting_form.val_data.data is not None:
                val_data = pd.read_csv(fitting_form.val_data.data)
            else:
                val_data = None
            target_name = fitting_form.target_name.data
            datasets['train_data'] = train_data
            datasets['val_data'] = val_data
            datasets['target_name'] = target_name

            errors = check_train_val(train_data, val_data, target_name)
            if len(errors) != 0:
                return redirect(url_for('get_rf'))

            n_estimators = fitting_form.n_estimators.data
            max_depth = fitting_form.max_depth.data
            feature_subsampling = fitting_form.feature_subsample_size.data

            Session['model_type'] = 'Случайный лес'
            Session['model_train'] = None
            Session['model_params'] = {'n_estimators': n_estimators, 'max_depth': max_depth, 'feature_subsampling': feature_subsampling}
            errors = []
            return redirect(url_for('get_predict'))


        return render_template('fitting.html', form=fitting_form, gb=False, title='Случайный лес', menu=menu, errors=errors)

    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for('get_rf'))


@app.route('/get_gb', methods=['GET', 'POST'])
def get_gb():
    try:
        global errors
        fitting_form = FittingForm()
        if fitting_form.validate_on_submit():
            train_data = pd.read_csv(fitting_form.train_data.data)
            if fitting_form.val_data.data is not None:
                val_data = pd.read_csv(fitting_form.val_data.data)
            else:
                val_data = None
            target_name = fitting_form.target_name.data
            datasets['train_data'] = train_data
            datasets['val_data'] = val_data
            datasets['target_name'] = target_name

            errors = check_train_val(train_data, val_data, target_name)
            if len(errors) != 0:
                return redirect(url_for('get_gb'))

            n_estimators = fitting_form.n_estimators.data
            max_depth = fitting_form.max_depth.data
            feature_subsampling = fitting_form.feature_subsample_size.data
            learning_rate = fitting_form.learning_rate.data

            Session['model_type'] = 'Градиентный бустинг'
            Session['model_train'] = None
            Session['model_params'] = {'n_estimators': n_estimators, 'max_depth': max_depth, 'feature_subsampling': feature_subsampling, 'learning_rate': learning_rate}

            errors = []
            return redirect(url_for('get_predict'))

        return render_template('fitting.html', form=fitting_form, gb=True, title='Градиентный бустинг', menu=menu, errors=errors)

    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for('get_gb'))

@app.route('/about_me')
def get_about():
    return render_template('about_me.html', menu=menu, title='Тебе крышка')

@app.route('/prediction', methods=['GET', 'POST'])
def get_predict():
    try:
        global errors
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

        n_estimators = Session['model_params']['n_estimators']
        max_depth = Session['model_params']['max_depth']
        feature_subsampling = Session['model_params']['feature_subsampling']
        learning_rate = 0.1
        if 'learning_rate' in Session:
            learning_rate = Session['model_params']['learning_rate']

        if Session['model_train'] == None:
            if Session['model_type'] == 'Градиентный бустинг':
                model = GradientBoostingMSE(n_estimators=n_estimators, max_depth=max_depth,
                                            feature_subsample_size=feature_subsampling,
                                            learning_rate=learning_rate)
            else:
                model = RandomForestMSE(n_estimators=n_estimators, max_depth=max_depth,
                                        feature_subsample_size=feature_subsampling)

            Session['model_train'] = model

        if X_val is not None:
            acc_train, acc_val, time_md = Session['model_train'].fit(X_train, y_train, X_val, y_val, trace=True)
        else:
            acc_train, time_md = Session['model_train'].fit(X_train, y_train, trace=True)
            acc_val = None
        predict_form = PredictForm()
        Session['acc_train'] = acc_train
        Session['acc_val'] = acc_val

        if predict_form.validate_on_submit():
            data_test = pd.read_csv(predict_form.test_data.data)

            errors = check_train_test(train_data, data_test, target_name)

            if len(errors) != 0:
                return redirect(url_for('get_predict'))

            y_pred = pd.DataFrame({target_name: Session['model_train'].predict(data_test.values)})
            file_name = predict_form.test_data.name + '_predict.csv'
            path = os.path.join(os.getcwd(), 'tmp/')
            if not os.path.exists(path):
                os.mkdir(path)
            y_pred.to_csv(os.path.join(path, file_name), index=False)
            return send_from_directory(path, file_name, as_attachment=True)

        ch_p = ['Тип ансамбля', 'Число деревьев', 'Число признаков']
        ch_v = [Session['model_type'], n_estimators, feature_subsampling]
        if Session['model_type'] == 'Градиентный бустинг':
            ch_p.append('Темп обучения')
            ch_v.append(learning_rate)
        params = pd.DataFrame({
            'Подбираемый параметр': ch_p,
            'Значение': ch_v
        })

        return render_template('prediction.html', acc_train=acc_train,
        acc_val=acc_val, menu=menu, title=Session['model_type'], form=predict_form, errors=errors, params=params)

    except Exception as exc:
        app.logger.info('Exception: {0}'.format(exc))
        return redirect(url_for('get_predict'))
