import yaml


def get_config(config_path):
    with open(config_path) as file:
        yaml_data = yaml.safe_load(file)

    return yaml_data
