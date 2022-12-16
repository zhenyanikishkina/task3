from flask_wtf import FlaskForm
from flask_wtf.file import FileAllowed
from wtforms import StringField, IntegerField, SelectField, SubmitField
from wtforms import FloatField
from flask_wtf.file import FileField, FileAllowed, FileRequired
from wtforms.validators import DataRequired, Optional, NumberRange


ensembles_types = [
    ('RandomForest', 'Случайный лес'),
    ('GradientBoosting', 'Градиентный бустинг')
]

class FittingForm(FlaskForm):
    ensemble_type = SelectField('Тип ансамбля', choices=ensembles_types)
    n_estimators = IntegerField('Число деревьев', validators=[DataRequired(),
                                NumberRange(min=1, max=1000)])
    max_depth = IntegerField('Максимальная глубина', validators=[DataRequired(),
                             NumberRange(min=1, max=50)])
    feature_subsample_size = IntegerField('Число признаков', validators=
                                          [DataRequired(),
                                          NumberRange(min=1, max=50)])
    learning_rate = FloatField('Learning rate', default=0.1)
    train_data = FileField('Обучающая выборка', validators=[FileRequired(),
                           FileAllowed(['csv'], 'Допускается только формат .csv')
                           ])
    target_name = StringField('Целевая переменная')
    val_data = FileField('Валидационная выборка', validators=[Optional(),
                         FileAllowed(['csv'], 'Допускается только формат .csv')])
    submit = SubmitField('Обучить')

class PredictForm(FlaskForm):
    test_data = FileField('Тестовая выборка', validators=[FileRequired(),
                           FileAllowed(['csv'], 'Допускается только формат .csv')
                           ])
    submit = SubmitField('Предсказать')