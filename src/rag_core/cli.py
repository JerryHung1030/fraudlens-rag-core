"""
===============================================================================
    Module Name: cli.py
    Description: Command-line interface for interacting with the RAG API.
    Author: Jerry, Ken, SJ
    Last Updated: 2025-06-23
    Version: 1.0.0
    Notes: 無
===============================================================================
"""
import click
import json
import requests
from typing import Optional

API_BASE_URL = "http://localhost:8000/api/v1"


@click.group()
def cli():
    """rag-core-x CLI - A command line interface for rag-core-x API"""
    pass


@cli.command()
@click.option('--project-id', required=True, help='Project ID for the RAG task')
@click.option('--input-file', type=click.Path(exists=True), required=True, help='JSON file containing input data')
@click.option('--reference-file', type=click.Path(exists=True), required=True, help='JSON file containing reference data')
@click.option('--direction', default='forward', type=click.Choice(['forward', 'backward']), help='RAG direction')
@click.option('--rag-k', default=5, type=int, help='Number of top-k results')
@click.option('--cof-threshold', default=0.6, type=float, help='Confidence threshold')
def rag(project_id: str, input_file: str, reference_file: str, direction: str, rag_k: int, cof_threshold: float):
    """Submit a RAG task to the API"""
    try:
        # Read input and reference data
        with open(input_file, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        with open(reference_file, 'r', encoding='utf-8') as f:
            reference_data = json.load(f)

        # Prepare request payload
        payload = {
            "project_id": project_id,
            "scenario": {
                "direction": direction,
                "rag_k": rag_k,
                "cof_threshold": cof_threshold
            },
            "input_data": input_data,
            "reference_data": reference_data
        }

        # Submit request
        response = requests.post(f"{API_BASE_URL}/rag", json=payload)
        response.raise_for_status()
        
        # Print job ID
        result = response.json()
        click.echo(f"Task submitted successfully. Job ID: {result.get('job_id')}")
        
    except requests.exceptions.RequestException as e:
        click.echo(f"Error submitting task: {str(e)}", err=True)
    except json.JSONDecodeError:
        click.echo("Error: Invalid JSON in input or reference file", err=True)
    except Exception as e:
        click.echo(f"Unexpected error: {str(e)}", err=True)


@cli.command()
@click.argument('job_id')
def status(job_id: str):
    """Check the status of a RAG task"""
    try:
        response = requests.get(f"{API_BASE_URL}/rag/{job_id}/status")
        response.raise_for_status()
        result = response.json()
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        click.echo(f"Error checking status: {str(e)}", err=True)


@cli.command()
@click.argument('job_id')
def result(job_id: str):
    """Get the results of a completed RAG task"""
    try:
        response = requests.get(f"{API_BASE_URL}/rag/{job_id}/result")
        response.raise_for_status()
        result = response.json()
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        click.echo(f"Error getting results: {str(e)}", err=True)


@cli.command()
@click.option('--project-id', help='Filter tasks by project ID')
@click.option('--clean-old/--no-clean-old', default=False, help='Clean old completed tasks')
def list_tasks(project_id: Optional[str], clean_old: bool):
    """List all RAG tasks"""
    try:
        params = {}
        if project_id:
            params['project_id'] = project_id
        if clean_old:
            params['clean_old'] = 'true'
            
        response = requests.get(f"{API_BASE_URL}/rag", params=params)
        response.raise_for_status()
        result = response.json()
        click.echo(json.dumps(result, indent=2, ensure_ascii=False))
    except requests.exceptions.RequestException as e:
        click.echo(f"Error listing tasks: {str(e)}", err=True)


if __name__ == '__main__':
    cli()
