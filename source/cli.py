import click

from .config import get_config
from .train import train_model
from .inference import inference_from_data


@click.group()
def cli():
    pass


@click.command()
@click.option("--config-path", help="Path to the configuration of the ml experiment")
def train(config_path):
    """
    Train and evaluate a machine learning model for the Titanic dataset.
    """
    if not config_path:
        click.echo("Please provide the path to the configuration yml file.")
        return

    click.echo("Starting training model...")
    config = get_config(config_path)
    train_model(config)
    click.echo("Model training and evaluation completed.")


@click.command()
@click.option("--data", help="custom data to do some inference on")
@click.option("--config-path", help="Path to the configuration of the ml experiment")
def inference(data, config_path):
    """
    Train and evaluate a machine learning model for the Titanic dataset.
    """
    if not config_path:
        click.echo("Please provide the path to the configuration yml file.")
        return

    click.echo("Starting inference over model...")
    config = get_config(config_path)
    predicted_results = inference_from_data(config)

    click.echo("Model inference complete.")
    click.echo(predicted_results)


cli.add_command(train)
cli.add_command(inference)

if __name__ == "__main__":
    cli()
