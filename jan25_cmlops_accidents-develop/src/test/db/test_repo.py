'''
The test suite includes the following test cases:

test_add_dataset: Verifies that the add_dataset method of the repository correctly adds a new dataset to the database.
test_add_feature: Verifies that the add_feature method of the repository correctly adds a new feature to the database.
test_add_dataset_feature: Verifies that the add_dataset_feature method of the repository correctly adds a new dataset-feature relationship to the database.
test_get_dataset_by_code: Verifies that the get_dataset_by_code method of the repository correctly retrieves a dataset by its code.
test_get_feature_by_code: Verifies that the get_feature_by_code method of the repository correctly retrieves a feature by its code.
test_get_all_datasets: Verifies that the get_all_datasets method of the repository correctly retrieves all datasets.
test_get_all_features: Verifies that the get_all_features method of the repository correctly retrieves all features.
test_get_dataset_features_by_code: Verifies that the get_dataset_features_by_code method of the repository correctly retrieves all dataset-feature relationships for a given dataset code.
'''
import pytest
from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from db.database import Database
from db.models import DataSet, Feature, DatasetFeature 
from db.repositoryDatasetFeatures import RepositoryDatasetFeatures
import random
from datetime import datetime


# Global
# Dataset code
dataset_code_1 = f"DS-{str(random.randint(0,100))}" 
dataset_code_2 = f"DS-{str(random.randint(0,100))}"            


@pytest.fixture
def session():
    db = Database()
    return db.get_session()


@pytest.fixture
def repository(session):
    return RepositoryDatasetFeatures(session)


def get_dataset():
    list_dataset = [DataSet(code=dataset_code_1, creation_dt=datetime.now(), description='TEST_1', type='TRAIN'), DataSet(code=dataset_code_2, creation_dt=datetime.now(), description='TEST_2', type='TTEST')]
    return list_dataset


def get_feature():
    list_feature = [Feature(code="lon"), Feature(code="lat")]
    return list_feature


def test_add_dataset(repository):
    dataset = get_dataset()[0]
    added_dataset = repository.add_dataset(dataset) 
    assert added_dataset == dataset
    #repository.session.add.assert_called_with(dataset)
    #repository.session.commit.assert_called_once()


def test_add_feature(repository):
    feature = get_feature()[0]
    added_feature = repository.add_feature(feature)
    assert added_feature == feature
    #repository.session.add.assert_called_with(feature)
    #repository.session.commit.assert_called_once()


def test_add_dataset_feature(repository):
    dataset = repository.session.query(DataSet).filter(DataSet.code == get_dataset()[0].code).first()
    feature =  repository.session.query(Feature).filter(Feature.code ==get_feature()[0].code).first()
    dataset_feature = DatasetFeature(dataset_id=dataset.id, feature_id=feature.id, value=0.75)
    added_dataset_feature = repository.add_dataset_feature(dataset_feature)
    assert added_dataset_feature == dataset_feature
    #repository.session.add.assert_called_with(dataset_feature)
    #repository.session.commit.assert_called_once()


def test_get_dataset_by_code(repository):
    dataset = get_dataset()[0]
    #repository.session.query().filter().first.return_value = dataset
    result = repository.get_dataset_by_code(dataset.code)
    assert result.code == dataset.code
    #repository.session.query.assert_called_with(DataSet)
    #repository.session.query().filter.assert_called_with(DataSet.code == dataset.code)
    #repository.session.query().filter().first.assert_called_once()


def test_get_feature_by_code(repository):
    feature = get_feature()[0]
    result = repository.get_feature_by_code(get_feature()[0].code)
    assert result.code == feature.code
    #repository.session.query.assert_called_with(Feature)
    #repository.session.query().filter.assert_called_with(Feature.code == get_feature()[0].code)
    #repository.session.query().filter().first.assert_called_once()


def test_get_all_datasets(repository):
    datasets = get_dataset()
    #repository.session.query().all.return_value = datasets
    result = repository.get_all_datasets()
    assert len(result) == 1
    #repository.session.query.assert_called_with(DataSet)
    #repository.session.query().all.assert_called_once()


def test_get_all_features(repository):
    features = get_feature()
    #repository.session.query().all.return_value = features
    result = repository.get_all_features()
    assert len(result) == 1
    #repository.session.query.assert_called_with(Feature)
    #repository.session.query().all.assert_called_once()


def test_get_dataset_features_by_code(repository):
    # Arrange
    dataset = get_dataset()[0]
    feature = get_feature()[0]
    dataset_features = [
        DatasetFeature(dataset_id=dataset.id, feature_id=1),
        DatasetFeature(dataset_id=dataset.id, feature_id=2)
    ]
    dataset.dataset_features = dataset_features

    # Act
    result = repository.get_dataset_features_by_code(dataset.code)

    # Assert
    #assert result[0].code == feature.code
    #repository.session.query.assert_called_with(DataSet)
    #repository.session.query().filter.assert_called_with(DataSet.code == "DS001")

        


