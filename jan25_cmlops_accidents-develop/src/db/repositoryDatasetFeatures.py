from sqlalchemy.orm import Session
from sqlalchemy.exc import NoResultFound
from typing import List, Optional
from db.models import DataSet, DatasetFeature, Feature
#from datetime import datetime


class RepositoryDatasetFeatures:


    def __init__(self, session: Session):
        self.session = session

    # Create
    def add_dataset(self, dataset: DataSet) -> DataSet:
        self.session.add(dataset)
        self.session.commit()
        return dataset

    def add_feature(self, feature: Feature) -> Feature:
        self.session.add(feature)
        self.session.commit()
        return feature

    def add_dataset_feature(self, dataset_feature: DatasetFeature) -> DatasetFeature:
        self.session.add(dataset_feature)
        self.session.commit()
        return dataset_feature

    # Read
    def get_dataset_by_code(self, dataset_code: str) -> Optional[DataSet]:
        return self.session.query(DataSet).filter(DataSet.code == dataset_code).first()

    def get_feature_by_code(self, feature_code: int) -> Optional[Feature]:
        return self.session.query(Feature).filter(Feature.code == feature_code).first()

    def get_all_datasets(self) -> List[DataSet]:
        return self.session.query(DataSet).all()

    def get_all_features(self) -> List[Feature]:
        return self.session.query(Feature).all()
    
    def get_dataset_features_by_code(self, dataset_code: str) -> List[DatasetFeature]:
        """
        Récupère toutes les DatasetFeatures associées à un DataSet spécifique
        identifié par son code.

        :param dataset_code: Le code du DataSet
        :return: Liste de DatasetFeature
        """
        dataset = self.session.query(DataSet).filter(DataSet.code == dataset_code).first()
        
        if dataset:
            return dataset.dataset_features
        return []


    # Update
    def update_dataset(self, dataset_id: int, updated_data: dict) -> Optional[DataSet]:
        dataset = self.get_dataset(dataset_id)
        if dataset:
            for key, value in updated_data.items():
                setattr(dataset, key, value)
            self.session.commit()
            return dataset
        return None

    def update_feature(self, feature_id: int, updated_data: dict) -> Optional[Feature]:
        feature = self.get_feature(feature_id)
        if feature:
            for key, value in updated_data.items():
                setattr(feature, key, value)
            self.session.commit()
            return feature
        return None

    # Delete
    def delete_dataset(self, dataset_id: int) -> bool:
        try:
            dataset = self.get_dataset(dataset_id)
            if dataset:
                self.session.delete(dataset)
                self.session.commit()
                return True
            return False
        except NoResultFound:
            return False

    def delete_feature(self, feature_id: int) -> bool:
        try:
            feature = self.get_feature(feature_id)
            if feature:
                self.session.delete(feature)
                self.session.commit()
                return True
            return False
        except NoResultFound:
            return False

    def delete_dataset_feature(self, dataset_id: int, feature_id: int) -> bool:
        dataset_feature = self.session.query(DatasetFeature).filter_by(dataset_id=dataset_id, feature_id=feature_id).first()
        if dataset_feature:
            self.session.delete(dataset_feature)
            self.session.commit()
            return True
        return False