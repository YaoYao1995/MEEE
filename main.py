import click
from examples.instrument import run_example_local

@click.group()
def cli():
    pass

@cli.command(
    name='run_local',
    context_settings={'ignore_unknown_options': True})
@click.argument("example_module_name", required=True, type=str)
@click.argument('example_argv', nargs=-1, type=click.UNPROCESSED)
def run_example_local_cmd(example_module_name, example_argv):
    return run_example_local(example_module_name, example_argv)

cli.add_command(run_example_local_cmd)
cli.add_command(run_example_local_cmd, name='launch_example_local')

def main():
    return cli()

if __name__ == "__main__":
    main()
