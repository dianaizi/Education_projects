import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC


def filter_data(df):
    columns_to_drop = [
        'id',
        'url',
        'region',
        'region_url',
        'price',
        'manufacturer',
        'image_url',
        'description',
        'posting_date',
        'lat',
        'long'
    ]
    return df.drop(columns_to_drop, axis=1)


def delete_outliners(df):
    data = df['year']
    q25 = data.quantile(0.25)
    q75 = data.quantile(0.75)
    iqr = q75 - q25
    boundaries = (q25 - 1.5 * iqr, q75 + 1.5 * iqr)
    df.loc[data < boundaries[0], 'year'] = round(boundaries[0])
    df.loc[data > boundaries[1], 'year'] = round(boundaries[1])
    return df


def add_new_features(df):
    df.loc[:, 'short_model'] = df['model'].apply(lambda x: x.lower().split(' ')[0] if not pd.isna(x) else x)
    df.loc[:, 'age_category'] = df['year'].apply(lambda x: 'new' if x > 2013 else ('old' if x < 2006 else 'average'))
    return df


def main():
    df = pd.read_csv('homework.csv')

    X = df.drop(['price_category'], axis=1)
    y = df['price_category']

    preparation = Pipeline(steps=[
        ('filter', FunctionTransformer(filter_data)),
        ('deletion', FunctionTransformer(delete_outliners)),
        ('adding_features', FunctionTransformer(add_new_features))
    ])

    numerical_features = make_column_selector(dtype_include=['int64', 'float64'])
    categorical_features = make_column_selector(dtype_include=object)

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ('numerical', numerical_transformer, numerical_features),
        ('categorical', categorical_transformer, categorical_features)
    ])

    models = (
        LogisticRegression(solver='liblinear'),
        RandomForestClassifier(),
        SVC()
    )

    best_score = .0
    best_pipe = None
    for model in models:
        pipe = Pipeline(steps=[
            ('preparation', preparation),
            ('preprocessor', preprocessor),
            ('classifier', model)
        ])
        score = cross_val_score(pipe, X, y, cv=4, scoring='accuracy')
        print(f'model: {type(model).__name__}, acc_mean: {score.mean():.4f}, acc_std: {score.std():.4f}')
        if score.mean() > best_score:
            best_score = score.mean()
            best_pipe = pipe

    print(f'model: {type(best_pipe.named_steps["classifier"]).__name__}, accuracy: {best_score:4f}')
    joblib.dump(best_pipe, 'homework.pkl')


if __name__ == '__main__':
    main()
