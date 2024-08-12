import click
import pkg_resources
from tiff_utils.tiff_converter import convert_to_tiff

VERSION = pkg_resources.get_distribution('tiff-utils').version

@click.group()
def cli():
    pass

@cli.command()
def version():
    click.echo(VERSION)

@cli.command(name='convert-to-tiff')
@click.argument('read-path', type=click.Path(exists=True), required=True)
@click.argument('write-path', type=click.Path(), required=True)
@click.option('--subresolutions', type=int, default=4)
@click.option('--grid-size', type=int, default=1)
@click.option('--max-workers', type=int, default=None)
def convert_to_tiff_cli(read_path, write_path, subresolutions, grid_size, max_workers):
    convert_to_tiff(read_path, write_path, subresolutions, grid_size, max_workers)
    
    
if __name__ == '__main__':
    
    # Run the CLI
    cli()