from app.services.check_environment_service import CheckDatasetsEnvironment


class CheckEnvironmentController:
    @classmethod
    def check_datasets_environment(cls):
        CheckDatasetsEnvironment.check_origin_datasets_path()