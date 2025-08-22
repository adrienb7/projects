from sqlalchemy.orm import declarative_base, relationship
from sqlalchemy import Column, Integer, String, TIMESTAMP, Boolean, Float, Sequence, ForeignKey
from sqlalchemy.ext.associationproxy import association_proxy
from datetime import datetime

Base = declarative_base()

seq_dataset_id = Sequence("dataset_id_seq")
seq_feature_id = Sequence("feature_id_seq")


class DataSet(Base):
    __tablename__ = "te_datasets"

    id = Column(Integer, seq_dataset_id, primary_key=True, index=True)
    description = Column(String, index=True)
    creation_dt = Column(TIMESTAMP(timezone=False), nullable=False, default=datetime.now())
    modification_dt = Column(TIMESTAMP(timezone=False), nullable=False, default=datetime.now())
    is_new = Column(Boolean, nullable=False, default=True)
    code = Column(String, nullable=False)
    type = Column(String, nullable=False)

    # Association avec les features
    dataset_features = relationship("DatasetFeature", back_populates="dataset", cascade="all, delete-orphan")
    features = association_proxy("dataset_features", "feature")

    def __repr__(self):
        return f"<DataSet(id={self.id}, code='{self.code}')>"


#  - Entity Features
class Feature(Base):
    __tablename__ = 'te_features'
    id = Column(Integer, seq_feature_id, primary_key=True)
    code = Column(String, nullable=False)

    # Association avec les datasets
    feature_datasets = relationship("DatasetFeature", back_populates="feature", cascade="all, delete-orphan")
    datasets = association_proxy("feature_datasets", "dataset")

    def __repr__(self):
        return f"<Feature(id={self.id}, code='{self.code}')>"


# - Association One To Many Table Dataset --> Featrures
class DatasetFeature(Base):
    __tablename__ = 'ta_dataset_features'

    dataset_id = Column(Integer, ForeignKey('te_datasets.id'), primary_key=True)
    feature_id = Column(Integer, ForeignKey('te_features.id'), primary_key=True)
    value = Column(Float, nullable=False)  # valeur associée à ce lien

    # Relations vers les entités
    dataset = relationship("DataSet", back_populates="dataset_features")
    feature = relationship("Feature", back_populates="feature_datasets")

    def __repr__(self):
        return (f"<DatasetFeature(dataset_id={self.dataset_id}, "
                f"feature_id={self.feature_id}, value='{self.value}')>")


# https://docs.sqlalchemy.org/en/21/orm/basic_relationships.html#one-to-many
# 