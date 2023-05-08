from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from data.config import DATABASES


def create_session(database):
    db_engine = DATABASES[database]['ENGINE']
    db_name = DATABASES[database]['NAME']
    db_user = DATABASES[database]['USER']
    db_password = DATABASES[database]['PASSWORD']
    db_host = DATABASES[database]['HOST']
    db_port = DATABASES[database]['PORT']

    engine_url = f'{db_engine}://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}'
    engine = create_engine(engine_url, echo=False)
    Session = sessionmaker(bind=engine, expire_on_commit=False)
    session = Session()

    return session