import re
import time
from datetime import date

import requests
from github import Github

client = Github('...')

contributors_regex = r'(?P<contributors>(\d+,?\d*|∞))\s*\<\/span\>\s*contributor'

data_path = r'..\data\data.csv'

headers = ['full_name',
           'stars_count',
           'forks_count',
           'contributors_count',
           'commits_count',
           'days_count',
           'is_org',
           'readme_path',
           'topics']


def contributors_count(repo_name, retry_count=5, sleep_duration=5):
    raw_html = requests.get(f'https://github.com/{repo_name}')

    # Sometimes this fails unexpectedly (maybe the request fails) so we retry
    for _ in range(retry_count):
        match = re.search(contributors_regex, raw_html.text)

        if match:
            return match.group('contributors').replace(',', '')
        else:
            time.sleep(sleep_duration)

    raise Exception('Error reading contributors count!')


def days_count(repo_created_at):
    parts = repo_created_at.strftime('%Y %m %d').split(' ')
    parts = list(map(lambda s: int(s), parts))
    created_date = date(*parts)
    current_date = date(2019, 1, 12)
    days_count = current_date - created_date

    return days_count.days


def write_readme(readme, i):
    readme_name = fr'readme\{i}.txt'
    with open(fr'..\data\{readme_name}', 'w') as f:
        f.write(readme.content)

    return readme_name


def main(start=0):
    repos = client.search_repositories(
        query='stars:>=5000 is:public', sort='stars', order='desc')

    if start == 0:
        with open(data_path, 'a') as f:
            f.write(','.join(headers))
            f.write('\n')

    for i in range(start, 1000):
        repo = repos[i]
        print(f'{i}: {repo.full_name}')

        with open(data_path, 'a', encoding='utf-8') as f:
            data = []
            data.append(repo.full_name)
            data.append(str(repo.stargazers_count))
            data.append(str(repo.forks_count))
            data.append(str(contributors_count(repo.full_name)))
            data.append(str(repo.get_commits().totalCount))
            data.append(str(days_count(repo.created_at)))
            data.append('0' if repo.organization == None else '1')
            data.append(write_readme(repo.get_readme(), i))
            data.append(' '.join(repo.get_topics()))

            f.write(','.join(data))
            f.write('\n')


if __name__ == '__main__':
    main()
