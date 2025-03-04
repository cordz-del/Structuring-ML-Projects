import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# Example usage:
if __name__ == "__main__":
    config = load_config("config.yaml")
    print(config)
