import sys
import pytest
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

project_dir = Path(__file__).resolve().parents[3]
sys.path.append(str(project_dir))

try:
    from ml_project.src.models.train_model import train
    from ml_project.src.models.predict_model import predict
    from ml_project.entities import read_training_pipeline_params
except ModuleNotFoundError:
    project_dir = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_dir))
    from ml_project.src.models.train_model import train
    from ml_project.src.models.predict_model import predict
    from ml_project.entities import read_training_pipeline_params


COMPLETE_CONFIG_PATH = './tests/test_configs/complete_config.yaml'


@pytest.mark.parametrize(
    'model_type',
    [
        pytest.param('lightgbm', id='lightgbm'),
        pytest.param('catboost', id='catboost')
    ]
)
def test_train_predict(model_type, real_dataset):
    params = read_training_pipeline_params(COMPLETE_CONFIG_PATH)
    params.train_params.model = model_type
    train_dataset, val_dataset = train_test_split(real_dataset, test_size=0.2, random_state=0)
    feature_list = [x for x in real_dataset.columns if x != 'target']
    model = train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        params=params,
        features=feature_list,
        categorical_features=[],
        target='target',
    )

    if model_type == 'lightgbm':
        assert model.fitted_
    else:
        assert model.is_fitted()

    val_dataset['predict'] = predict(
        model, val_dataset[feature_list]
    )
    val_score = roc_auc_score(val_dataset['target'], val_dataset['predict'])

    assert 'predict' in val_dataset.columns
    assert val_dataset['predict'].isna().sum() == 0
    assert 0.967 - .05 < val_score < 1
