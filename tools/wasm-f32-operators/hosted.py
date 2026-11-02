"""Audit-only provider identity and authenticated, bounded original artifacts."""
import json
import os
from pathlib import Path
import re
import urllib.error
import urllib.request


def require(ok, message):
    if not ok:
        raise ValueError(message)


def identity(actual):
    require(re.fullmatch('[0-9a-f]{40}', actual), 'Exact source commit required')
    if os.environ.get('GITHUB_ACTIONS') == 'true':
        require(actual == os.environ.get('GITHUB_SHA'), 'GitHub source differs')
        return dict(provider='GitHub Actions', source=actual)
    require(Path('/kaggle/working').is_dir() and Path('/kaggle/input').is_dir() and
            os.environ.get('KAGGLE_KERNEL_RUN_TYPE') in ('Batch', 'Interactive'), 'Hosted exact source required')
    require(actual == os.environ.get('CRISPEMBED_CAPTURE_SOURCE') and
            re.fullmatch('[0-9a-f]{64}', os.environ.get('CRISPEMBED_BOOTSTRAP_SHA', '')),
            'Kaggle submitted source/script differs')
    return dict(provider='Kaggle CPU', source=actual,
                kernel_run_type=os.environ['KAGGLE_KERNEL_RUN_TYPE'],
                executed_script_sha256=os.environ['CRISPEMBED_BOOTSTRAP_SHA'])


def api(path):
    request = urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/' + path,
        headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN'], 'Accept': 'application/vnd.github+json',
                 'X-GitHub-Api-Version': '2022-11-28'})
    with urllib.request.urlopen(request, timeout=90) as response:
        data = response.read(2 * 1024**2 + 1)
    require(len(data) <= 2 * 1024**2, 'Artifact service metadata budget exceeded')
    return json.loads(data)


def verify_service(recorded, current):
    if current['provider'] == 'GitHub Actions':
        require(recorded.get('provider') == 'GitHub Actions' and
                recorded.get('head_sha') == current['source'] and
                recorded.get('id') == int(os.environ['GITHUB_RUN_ID']) and
                recorded.get('run_attempt') == int(os.environ['GITHUB_RUN_ATTEMPT']),
                'GitHub preparation service changed')
    else:
        require(recorded == current, 'Kaggle preparation source/script changed')


def artifact(run, name, destination, maximum):
    listing = api('actions/runs/' + str(run) + '/artifacts?per_page=100')
    rows = [row for row in listing['artifacts'] if row['name'] == name and not row['expired']]
    require(listing['total_count'] <= 100 and len(rows) == 1, 'Missing/ambiguous frozen artifact')
    row = rows[0]
    require(row['workflow_run']['id'] == run and 0 < row['size_in_bytes'] <= maximum, 'Artifact run/extent differs')

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *args, **kwargs):
            return None

    request = urllib.request.Request('https://api.github.com/repos/CrispStrobe/CrispEmbed/actions/artifacts/' +
        str(row['id']) + '/zip', headers={'Authorization': 'Bearer ' + os.environ['GH_TOKEN']})
    try:
        response = urllib.request.build_opener(NoRedirect).open(request, timeout=90)
    except urllib.error.HTTPError as error:
        require(error.code == 302, 'Artifact download service rejected')
        location = error.headers['Location']
        require(location.startswith('https://'), 'Signed artifact storage must use HTTPS')
        # A fresh request to storage has no GitHub Authorization header.
        try:
            response = urllib.request.urlopen(location, timeout=180)
        except Exception:
            raise ValueError('Signed artifact transfer failed') from None
    with response, destination.open('xb') as output:
        count = 0
        while block := response.read(1024**2):
            count += len(block)
            require(count <= row['size_in_bytes'], 'Artifact exceeded exact service extent')
            output.write(block)
    require(count == row['size_in_bytes'], 'Truncated frozen artifact')
    # The caller checks both immutable service digest and complete ZIP SHA
    # before parsing any bytes. This helper never substitutes another artifact.
    return {key: row[key] for key in ('id', 'name', 'size_in_bytes', 'digest')}
