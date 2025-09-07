import os

import yaml


class Configs:
    def __init__(
        self, env: str | None = None, config_paths: dict | None = None
    ) -> None:
        rootpath = os.path.dirname(os.path.dirname(__file__))
        default_config_paths = {"gcs": os.path.join(rootpath, "config", "gcs.yml")}
        if config_paths is None:
            self.config_paths = default_config_paths
        else:
            self.config_paths = config_paths
        self.cloud = {}
        self.cloud["gcs"] = self.load_cloud_config(env=env, config_key="gcs")

    def load_cloud_config(self, env: str, config_key: str) -> dict:
        print(f"loading gcs configs for {env}")
        cloud_fpath = self.config_paths[config_key]
        with open(cloud_fpath) as f:
            config_cloud = yaml.safe_load(f)
        cloud_shared = config_cloud["shared"]
        if env is not None:
            cloud_env = config_cloud[env]
        else:
            cloud_env = {}
        return {**cloud_shared, **cloud_env}


if __name__ == "__main__":
    configs = Configs()
    print(configs.cloud)
